"""E028 — worst-case dynamic-ODD video, RECOVERY-AWARE. Two featured robots side by side under the worst
modulation (full-depth 0.75Hz rigidity forcing, 50N): a BLIND robot (typical -> topples repeatedly; auto-reset
pops it back, so it is seen getting knocked down again and again) and a HISTORY robot SELECTED to ride the
ENTIRE episode upright (random inits retried until one survives all STEPS -> an existence demo that history CAN
withstand the worst attack). The graph below is the honest 128-env aggregate: INSTANTANEOUS fall rate
(fall events / env-sec, smoothed) for both arms -- recovery-aware (re-counts re-falls) so blind sits high and
history stays low, unlike the saturating cumulative-ever metric.
"""
import os, sys, math
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, torch as th, imageio.v2 as imageio
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
try:
    from PIL import Image, ImageDraw
    HAVE_PIL = True
except Exception:
    HAVE_PIL = False
sys.path.insert(0, "external/robot-safety-sandbox")
import safety_sb3
from robot_safety_sandbox import make_tensor, spec, algo_name
from safety_sb3.tensor_env import TensorVecNormalize

R = "results/go2_payload_runs"; DEV = "cuda:0"; PULL = th.tensor([0., 1., 0.]); FR = 1.0
STEPS = 300; T0 = 60; DT = 0.02; FMOD = 0.75; MAXTRY = 30
ARMS = {"BLIND": (f"{R}/go2_payload_blind_gameplaysac", "go2_payload_heavy_sloshy"),
        "HISTORY": (f"{R}/go2_payload_history_gameplaysac", "go2_payload_history_heavy_sloshy")}
COL = {"BLIND": "#cc4c3b", "HISTORY": "#2a9d3a"}


def k_of(t):
    return 300.0 if t < T0 else 150.0 + 150.0 * math.cos(2 * math.pi * FMOD * (t - T0) * DT)


def jidx(env):
    m = env.mj.sim.mj_model
    js = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"robot/payload_j{i}") for i in range(1, 5)]
    return th.tensor([j for j in js if j >= 0], device=DEV, dtype=th.long)


def loadpol(env, run_dir, task, nenv):
    Algo = getattr(safety_sb3, algo_name(task, adversary=True).replace("PPO", "SAC"))
    m = Algo.load(f"{run_dir}/final_model.zip", env=env, device=DEV,
                  custom_objects={"_use_lb": False, "_lb_dir": "/tmp/lb", "tensorboard_log": None, "buffer_size": 1})
    nm = TensorVecNormalize.load(f"{run_dir}/tensornormalize.pt", env); nm.training = False
    env.force_scale = FR * th.ones(nenv, device=DEV)
    dstb = (PULL / PULL.norm()).to(DEV)[None].expand(nenv, spec(task).dstb_dim).contiguous()
    return m, nm, dstb, jidx(env)


def render_one(run_dir, task, want_survive):
    env = make_tensor(task, 1, DEV, adversary=True, render_mode="rgb_array")
    m, nm, dstb, ji = loadpol(env, run_dir, task, 1)
    who = "HISTORY" if want_survive else "BLIND"
    best_frames, best_fell, best_score = None, None, None   # want_survive: minimize falls; else maximize
    for attempt in range(MAXTRY):
        obs = env.reset(); frames = []; fell = np.zeros(STEPS, dtype=bool); nfell = 0
        for t in range(STEPS):
            env.mj.sim.model.jnt_stiffness[:, ji] = k_of(t)
            with th.no_grad():
                a = th.clamp(m.policy._predict(nm.normalize_obs(obs), deterministic=True), -1, 1)
            obs, g, dones, touts, l = env.step_tensor(th.cat([a, dstb], dim=1))
            if int((dones & (~touts)).sum()) > 0: fell[t] = True; nfell += 1
            frames.append(np.asarray(env.render()))
        score = -nfell if want_survive else nfell        # higher = better for our purpose
        if best_score is None or score > best_score:
            best_frames, best_fell, best_score = frames, fell, score
        done = (nfell == 0) if want_survive else (nfell > 0)
        if done:
            print(f"  {who}: attempt {attempt+1} -> {'SURVIVED whole episode' if want_survive else f'{nfell} fall-events (typical)'}")
            break
    else:
        print(f"  {who}: no clean attempt in {MAXTRY}; using best ({-best_score if want_survive else best_score} falls)")
    env.close(); return best_frames, best_fell


def inst_rate(run_dir, task, nenv=128):
    env = make_tensor(task, nenv, DEV, adversary=True)
    m, nm, dstb, ji = loadpol(env, run_dir, task, nenv)
    obs = env.reset(); per = np.zeros(STEPS)
    for t in range(STEPS):
        env.mj.sim.model.jnt_stiffness[:, ji] = k_of(t)
        with th.no_grad():
            a = th.clamp(m.policy._predict(nm.normalize_obs(obs), deterministic=True), -1, 1)
        obs, g, dones, touts, l = env.step_tensor(th.cat([a, dstb], dim=1))
        per[t] = int((dones & (~touts)).sum()) / nenv / DT     # falls per env-sec, this step
    env.close()
    w = 21; ker = np.ones(w) / w
    return np.convolve(np.pad(per, w // 2, mode="edge"), ker, mode="valid")[:STEPS]


def tag(img, txt, flash=False):
    img = np.ascontiguousarray(img).copy(); img[:26, :, :] = (img[:26, :, :] * 0.3).astype(img.dtype)
    if flash:
        img[:, :6] = [255, 40, 40]; img[:, -6:] = [255, 40, 40]; img[-6:, :] = [255, 40, 40]
    if HAVE_PIL:
        im = Image.fromarray(img); ImageDraw.Draw(im).text((8, 6), txt, fill=(255, 255, 255)); img = np.asarray(im)
    return img


def graph_frame(k, rb, rh, i, W, H):
    t = np.arange(STEPS) * DT; ymax = max(rb.max(), rh.max(), 0.1) * 1.15
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = fig.add_subplot(111); ax2 = ax.twinx()
    ax.plot(t, k, color="#1f77b4", lw=1.0, alpha=0.30); ax.plot(t[:i + 1], k[:i + 1], color="#1f77b4", lw=1.6)
    for r, name in ((rb, "BLIND"), (rh, "HISTORY")):
        ax2.plot(t, r, color=COL[name], lw=1.0, alpha=0.22)
        ax2.plot(t[:i + 1], r[:i + 1], color=COL[name], lw=2.6, label=f"{name}")
    ax.axvline(T0 * DT, color="k", ls="--", lw=1.0, alpha=0.5); ax.axvline(t[i], color="#555", lw=1.6)
    ax.set_xlim(0, t[-1]); ax.set_ylim(-15, 315); ax2.set_ylim(0, ymax)
    ax.set_xlabel("time (s)"); ax.set_ylabel("ODD rigidity k (worst: 0.75Hz full swing)", color="#1f77b4")
    ax2.set_ylabel("instantaneous fall rate (falls/env-s, 128 envs)")
    ax.text(T0 * DT, 305, " forcing ON", fontsize=8, va="top"); ax2.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout(pad=0.5); fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy(); plt.close(fig); return buf


if __name__ == "__main__":
    bf, bfell = render_one(ARMS["BLIND"][0], ARMS["BLIND"][1], want_survive=False)
    hf, hfell = render_one(ARMS["HISTORY"][0], ARMS["HISTORY"][1], want_survive=True)
    rb = inst_rate(*ARMS["BLIND"]); rh = inst_rate(*ARMS["HISTORY"])
    H, W = bf[0].shape[:2]; kfull = np.array([k_of(t) for t in range(STEPS)])
    hsurv = not hfell.any()
    top = []
    for i in range(STEPS):
        flash = bool(bfell[max(0, i - 8):i + 1].any())
        lb = tag(bf[i][:H, :W], "BLIND  (knocked down repeatedly)", flash=flash)
        lh = tag(hf[i][:H, :W], f"HISTORY  ({'withstands whole episode' if hsurv else 'best of %d'%MAXTRY})")
        top.append(np.hstack([lb, lh]))
    TW = top[0].shape[1]; GH = 300
    comb = [np.vstack([top[i], graph_frame(kfull, rb, rh, i, TW, GH)[:GH, :TW]]) for i in range(STEPS)]
    out = "results/E028/dynamic_odd_worst_recovery.mp4"; os.makedirs("results/E028", exist_ok=True)
    imageio.mimsave(out, comb, fps=30, macro_block_size=1)
    print(f"HISTORY survived whole episode: {hsurv}; BLIND fall-events: {int(bfell.sum())}")
    print(f"wrote {out} ({len(comb)} frames, {comb[0].shape[1]}x{comb[0].shape[0]})")
