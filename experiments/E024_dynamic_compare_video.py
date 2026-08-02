"""E024 — HONEST dynamic-ODD comparison video (supersedes E022, which showed only 2 envs -> a misleading
single anecdote). Layout: BLIND 16-robot grid | HISTORY 16-robot grid on top; a shared graph below plots the
ODD rigidity change k(t) AND the cumulative fall-fraction (fraction of 128 envs that have failed by time t)
for BOTH arms. The fall-fraction curves are the actual aggregate metric, so the picture matches the statistic:
after the mid-episode jump, BLIND's fraction climbs while HISTORY's stays low. Moving cursor synced to frames.
"""
import os, sys
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
STEPS = 220; T_JUMP = 110; DT = 0.02
ARMS = {"BLIND": (f"{R}/go2_payload_blind_gameplaysac", "go2_payload_heavy_sloshy"),
        "HISTORY": (f"{R}/go2_payload_history_gameplaysac", "go2_payload_history_heavy_sloshy")}
COL = {"BLIND": "#cc4c3b", "HISTORY": "#2a9d3a"}


def jidx(env):
    m = env.mj.sim.mj_model
    js = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"robot/payload_j{i}") for i in range(1, 5)]
    return th.tensor([j for j in js if j >= 0], device=DEV, dtype=th.long)


def load(run_dir, task, nenv, render):
    env = make_tensor(task, nenv, DEV, adversary=True, **({"render_mode": "rgb_array"} if render else {}))
    if render:
        try: env.mj.cfg.viewer.max_extra_envs = max(1, nenv - 1)
        except Exception: pass
    Algo = getattr(safety_sb3, algo_name(task, adversary=True).replace("PPO", "SAC"))
    m = Algo.load(f"{run_dir}/final_model.zip", env=env, device=DEV,
                  custom_objects={"_use_lb": False, "_lb_dir": "/tmp/lb", "tensorboard_log": None, "buffer_size": 1})
    nm = TensorVecNormalize.load(f"{run_dir}/tensornormalize.pt", env); nm.training = False
    env.force_scale = FR * th.ones(nenv, device=DEV)
    dstb = (PULL / PULL.norm()).to(DEV)[None].expand(nenv, spec(task).dstb_dim).contiguous()
    return env, m, nm, dstb, jidx(env)


def rollout(run_dir, task, nenv, render):
    env, m, nm, dstb, ji = load(run_dir, task, nenv, render)
    obs = env.reset(); frames = []; ever = th.zeros(nenv, dtype=th.bool, device=DEV); frac = []
    for t in range(STEPS):
        env.mj.sim.model.jnt_stiffness[:, ji] = 0.0 if t >= T_JUMP else 300.0
        with th.no_grad():
            a = th.clamp(m.policy._predict(nm.normalize_obs(obs), deterministic=True), -1, 1)
        obs, g, dones, touts, l = env.step_tensor(th.cat([a, dstb], dim=1))
        ever |= (dones & (~touts)); frac.append(float(ever.float().mean()))
        if render: frames.append(np.asarray(env.render()))
    env.close()
    return frames, np.array(frac)


def tag(img, txt):
    img = np.ascontiguousarray(img).copy(); img[:26, :, :] = (img[:26, :, :] * 0.3).astype(img.dtype)
    if HAVE_PIL:
        im = Image.fromarray(img); ImageDraw.Draw(im).text((8, 6), txt, fill=(255, 255, 255)); img = np.asarray(im)
    return img


def graph_frame(k, fb, fh, i, W, H):
    t = np.arange(STEPS) * DT
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = fig.add_subplot(111); ax2 = ax.twinx()
    ax.plot(t, k, color="#1f77b4", lw=1.0, alpha=0.30)
    ax.plot(t[:i + 1], k[:i + 1], color="#1f77b4", lw=2.2)
    for f, name in ((fb, "BLIND"), (fh, "HISTORY")):
        ax2.plot(t, f, color=COL[name], lw=1.0, alpha=0.25)
        ax2.plot(t[:i + 1], f[:i + 1], color=COL[name], lw=2.4, label=f"{name} fell")
    ax.axvline(T_JUMP * DT, color="k", ls="--", lw=1.0, alpha=0.5)
    ax.axvline(t[i], color="#555", lw=1.6)
    ax.set_xlim(0, t[-1]); ax.set_ylim(-15, 315); ax2.set_ylim(-0.02, 1.02)
    ax.set_xlabel("time (s)"); ax.set_ylabel("ODD rigidity k (300 rigid / 0 sloshy)", color="#1f77b4")
    ax2.set_ylabel("cumulative fall fraction (128 envs)")
    ax.text(T_JUMP * DT, 305, " ODD JUMP rigid->sloshy", fontsize=8, va="top")
    ax2.legend(loc="upper left", fontsize=8, framealpha=0.9)
    fig.tight_layout(pad=0.5); fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy(); plt.close(fig)
    return buf


if __name__ == "__main__":
    grids, fracs = {}, {}
    for arm, (rd, task) in ARMS.items():
        grids[arm], _ = rollout(rd, task, 16, True)          # 16-env visual grid
        _, fracs[arm] = rollout(rd, task, 128, False)        # 128-env honest fall-fraction curve
        print(f"{arm}: final fall-fraction over 128 envs = {fracs[arm][-1]:.2f}")
    H, W = grids["BLIND"][0].shape[:2]
    top = [np.hstack([tag(grids["BLIND"][i][:H, :W], f"BLIND  ({fracs['BLIND'][-1]*100:.0f}% fell)"),
                      tag(grids["HISTORY"][i][:H, :W], f"HISTORY  ({fracs['HISTORY'][-1]*100:.0f}% fell)")])
           for i in range(STEPS)]
    TW = top[0].shape[1]; GH = 300
    comb = [np.vstack([top[i], graph_frame(np.array([0. if t >= T_JUMP else 300. for t in range(STEPS)]),
                                           fracs["BLIND"], fracs["HISTORY"], i, TW, GH)[:GH, :TW]]) for i in range(STEPS)]
    out = "results/E024/dynamic_odd_compare.mp4"; os.makedirs("results/E024", exist_ok=True)
    imageio.mimsave(out, comb, fps=30, macro_block_size=1)
    print(f"wrote {out} ({len(comb)} frames, {comb[0].shape[1]}x{comb[0].shape[0]})")
