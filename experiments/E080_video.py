"""E080 videos — square & sine load waves, 3 panels: STAND-ONLY | BIDIRECTIONAL HANDOFF | REST-ONLY.
Trunk tint: gray->red with W in stand mode; BLUE while in rest mode. Graph: W(t) + boundary, EMA V_stand with
both thresholds, per-arm heights. Mirrors experiments/E080_bidirectional.py guard exactly.
"""
import os, sys, io, contextlib, math, mujoco
from _paths import _ART
import torch as th
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "external/robot-safety-sandbox"); sys.path.insert(0, "experiments")
import numpy as np, imageio.v2 as imageio
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    from PIL import Image, ImageDraw
    HAVE_PIL = True
except Exception:
    HAVE_PIL = False
from robot_safety_sandbox import make_tensor, spec
from robot_safety_sandbox.eval.policies import load_twin
from _value_util import stand_value

DEV, DT, STEPS = "cuda:0", 0.02, 1000
EPS_DN, K_DN, EPS_UP, K_UP, REFRACT, ALPHA = -0.05, 5, 0.15, 25, 50, 0.1
W_C = 130.0
CK = "results/go2_weight_runs/{r}/checkpoints/model_49999872_steps.zip"
ARMS = ["STAND-ONLY", "HANDOFF", "REST-ONLY"]
COL = {"STAND-ONLY": "#e74c3c", "HANDOFF": "#1a5276", "REST-ONLY": "#1e8449"}
GRAY, RED, BLUE = np.array([0.5, 0.5, 0.5, 1.0]), np.array([0.85, 0.1, 0.1, 1.0]), np.array([0.15, 0.35, 0.9, 1.0])


def W_of(sched, t):
    s = t * DT
    if sched == "square":
        return 40.0 if (int(s // 3.0) % 2 == 0) else 220.0
    return 130.0 + 110.0 * math.sin(2 * math.pi * s / 8.0)


def rollout(sched, arm, nenv, render):
    with contextlib.redirect_stdout(io.StringIO()):
        env = make_tensor("go2_weight_rest_hi_at_0", nenv, DEV, adversary=True,
                          **({"render_mode": "rgb_array"} if render else {}))
        if render:
            try: env.mj.cfg.viewer.max_extra_envs = max(1, nenv - 1)
            except Exception: pass
        m_s, n_s = load_twin(CK.format(r="E075_recal/go2_weight_stand_hi_adv"), DEV, quiet=True)
        m_r, n_r = load_twin(CK.format(r="go2_weight_rest_hi_adv"), DEV, quiet=True)
    inner = env.mj; mm = inner.sim.mj_model; tm = inner.sim.model
    base_id = mujoco.mj_name2id(mm, mujoco.mjtObj.mjOBJ_BODY, "robot/base_link")
    trunk = [g for g in range(mm.ngeom) if mm.geom_bodyid[g] == base_id]
    env.force_scale = 0.2 * th.ones(nenv, device=DEV)
    dstb = th.zeros(nenv, spec("go2_weight_rest_hi_at_0").dstb_dim, device=DEV); dstb[:, 1] = 1.0
    obs = env.reset()
    in_rest = th.zeros(nenv, dtype=th.bool, device=DEV)
    if arm == "REST-ONLY": in_rest[:] = True
    vbar = th.zeros(nenv, device=DEV); below = th.zeros(nenv, device=DEV); above = th.zeros(nenv, device=DEV)
    refr = th.zeros(nenv, device=DEV)
    frames, hs, Vs, rf = [], [], [], []
    for t in range(STEPS):
        W = W_of(sched, t)
        env.base_load = th.tensor([0., 0., -W], device=DEV)[None].expand(nenv, 3).contiguous()
        inner._weight_W[:] = W; inner._weight_h[:] = 0.25
        V = stand_value(env, m_s, n_s)
        vbar = (1 - ALPHA) * vbar + ALPHA * V
        if arm == "HANDOFF":
            below = th.where(vbar < EPS_DN, below + 1, th.zeros_like(below))
            above = th.where(vbar > EPS_UP, above + 1, th.zeros_like(above))
            can = refr <= 0
            go_dn = (~in_rest) & (below >= K_DN) & can
            go_up = in_rest & (above >= K_UP) & can
            in_rest = th.where(go_dn, th.ones_like(in_rest), in_rest)
            in_rest = th.where(go_up, th.zeros_like(in_rest), in_rest)
            refr = th.where(go_dn | go_up, th.full_like(refr, REFRACT), refr - 1)
        with th.no_grad():
            a_s = th.clamp(m_s.policy._predict(n_s(obs), deterministic=True), -1, 1)
            a_r = th.clamp(m_r.policy._predict(n_r(obs), deterministic=True), -1, 1)
        a = th.where(in_rest.unsqueeze(-1), a_r, a_s)
        obs, g, dones, touts, l = env.step_tensor(th.cat([a, dstb], dim=1))
        d = inner.scene["robot"].data
        hs.append(float(d.root_link_pos_w[:, 2].mean()))
        Vs.append(float(vbar.mean())); rf.append(float(in_rest.float().mean()))
        if render:
            tint = GRAY + (RED - GRAY) * min(1.0, W / 250.0)
            host = BLUE if float(in_rest.float().mean()) > 0.5 else tint
            rgba = th.tensor(tint, device=DEV, dtype=tm.geom_rgba.dtype)[None, None, :].expand(nenv, len(trunk), 4).clone()
            rgba[in_rest] = th.tensor(BLUE, device=DEV, dtype=tm.geom_rgba.dtype)
            tm.geom_rgba[:, trunk, :] = rgba
            for gg in trunk: mm.geom_rgba[gg] = host
            frames.append(np.asarray(env.render()))
    env.close()
    return frames, np.array(hs), np.array(Vs), np.array(rf)


def tag(img, txt):
    img = np.ascontiguousarray(img).copy(); img[:26] = (img[:26] * 0.3).astype(img.dtype)
    if HAVE_PIL:
        im = Image.fromarray(img); ImageDraw.Draw(im).text((8, 6), txt, fill=(255, 255, 255)); img = np.asarray(im)
    return img


def graph_frame(i, sched, data, W, H):
    t = np.arange(STEPS) * DT
    Wv = np.array([W_of(sched, k) for k in range(STEPS)])
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = fig.add_subplot(111); ax2 = ax.twinx()
    ax2.plot(t, Wv, color="#222", lw=1.0, alpha=0.3)
    ax2.plot(t[:i + 1], Wv[:i + 1], color="#222", lw=2.0, label="load W(t)")
    ax2.axhline(W_C, color="#c0392b", ls="--", lw=1.2, alpha=0.7)
    ax2.text(0.1, W_C + 6, "W_c (boundary)", fontsize=8, color="#c0392b")
    for arm in ARMS:
        ax.plot(t, data[arm]["h"], color=COL[arm], lw=1.0, alpha=0.25)
        ax.plot(t[:i + 1], data[arm]["h"][:i + 1], color=COL[arm], lw=2.4, label=arm)
    V = data["HANDOFF"]["V"]
    ax.plot(t[:i + 1], 0.32 + V[:i + 1] * 0.25, color="#1a5276", ls="--", lw=1.5, label="V̄_stand (scaled)")
    ax.axhline(0.32 + EPS_DN * 0.25, color="r", ls=":", lw=1.0)
    ax.axhline(0.32 + EPS_UP * 0.25, color="#1e8449", ls=":", lw=1.0)
    ax.axvline(t[i], color="#555", lw=1.5)
    ax.set_xlim(0, t[-1]); ax.set_ylim(0.05, 0.45); ax2.set_ylim(0, 260)
    ax.set_xlabel("time (s)"); ax.set_ylabel("base height (m) / V̄"); ax2.set_ylabel("W (N)")
    ax.legend(loc="upper left", fontsize=7.5, ncol=2, framealpha=0.9); ax2.legend(loc="upper right", fontsize=8)
    fig.tight_layout(pad=0.5); fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy(); plt.close(fig)
    return buf


if __name__ == "__main__":
    od = os.path.expanduser(_ART + "/E080-bidirectional")
    for sched in ("square", "sine"):
        data, grids = {}, {}
        for arm in ARMS:
            grids[arm], h, V, rf = rollout(sched, arm, 16, True)
            data[arm] = {"h": h, "V": V, "rf": rf}
            print(f"{sched} {arm}: rendered {len(grids[arm])} frames")
        Hh, Ww = grids["STAND-ONLY"][0].shape[:2]
        SUB = 2                                                     # save every 2nd frame -> ~17s @30fps
        idxs = list(range(0, STEPS, SUB))
        top = [np.hstack([tag(grids[a][i][:Hh, :Ww], a) for a in ARMS]) for i in idxs]
        TW = top[0].shape[1]; GH = 320
        comb = [np.vstack([top[k], graph_frame(idxs[k], sched, data, TW, GH)[:GH, :TW]]) for k in range(len(idxs))]
        vp = f"{od}/bidirectional_{sched}.mp4"
        imageio.mimsave(vp, comb, fps=30, macro_block_size=1)
        print("video ->", vp)
