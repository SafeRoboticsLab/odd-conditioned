"""E072 — VIDEO + FIGURE of the certified weight-ladder handoff (E071 demo).

Video: STAND-ONLY | HANDOFF (ours) | REST-ONLY robot grids on the W(t) ramp, trunks TINTED gray->red with the
growing load; the handoff arm's trunk flips BLUE the moment it switches to the REST mode. Synced graph below:
mean base height per arm, W(t), the handoff arm's V_stand trace with the eps trigger line, switch-time band.
Figure: standalone timeline (W(t), V_stand contraction crossing eps, heights, switch distribution) — the
"weight increase and mode switch" story in one plot.
"""
import os, sys, io, contextlib, mujoco
from _paths import _ART
import torch as th
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "external/robot-safety-sandbox")
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
sys.path.insert(0, "experiments")
from _value_util import stand_value

DEV, DT = "cuda:0", 0.02
STEPS, RAMP_START, RAMP_LEN, W_MAX = 500, 100, 300, 300.0
PULL_SCALE = 0.2
EPS, HYST = 0.015, 5
CK = "results/go2_weight_runs/go2_weight_{m}_adv/checkpoints/model_49999872_steps.zip"
TASK = "go2_weight_rest_at_0"                    # permissive cfg; W driven per step
ARMS = ["STAND-ONLY", "HANDOFF", "REST-ONLY"]
COL = {"STAND-ONLY": "#f1948a", "HANDOFF": "#1a5276", "REST-ONLY": "#7dcea0"}
GRAY, RED, BLUE = np.array([0.5, 0.5, 0.5, 1.0]), np.array([0.85, 0.10, 0.10, 1.0]), np.array([0.15, 0.35, 0.9, 1.0])


def W_of(t):
    if t < RAMP_START: return 0.0
    if t < RAMP_START + RAMP_LEN: return (t - RAMP_START) / RAMP_LEN * W_MAX
    return W_MAX


def rollout(arm, nenv, render):
    with contextlib.redirect_stdout(io.StringIO()):
        env = make_tensor(TASK, nenv, DEV, adversary=True, **({"render_mode": "rgb_array"} if render else {}))
        if render:
            try: env.mj.cfg.viewer.max_extra_envs = max(1, nenv - 1)
            except Exception: pass
        m_stand, n_stand = load_twin(CK.format(m="stand"), DEV, quiet=True)
        m_rest, n_rest = load_twin(CK.format(m="rest"), DEV, quiet=True)
    inner = env.mj; mm = inner.sim.mj_model; tm = inner.sim.model
    base_id = mujoco.mj_name2id(mm, mujoco.mjtObj.mjOBJ_BODY, "robot/base_link")
    trunk_geoms = [g for g in range(mm.ngeom) if mm.geom_bodyid[g] == base_id]
    env.force_scale = PULL_SCALE * th.ones(nenv, device=DEV)
    dstb = th.tensor([0., 1., 0.], device=DEV)[None].expand(nenv, spec(TASK).dstb_dim).contiguous()
    obs = env.reset()
    switched = th.zeros(nenv, dtype=th.bool, device=DEV); below = th.zeros(nenv, device=DEV)
    switch_t = th.full((nenv,), -1.0, device=DEV)
    frames, hs, Vs, swf = [], [], [], []
    for t in range(STEPS):
        W = W_of(t)
        env.base_load = th.tensor([0., 0., -W], device=DEV)[None].expand(nenv, 3).contiguous()
        inner._weight_W = th.full((nenv,), W, device=DEV)
        V = stand_value(env, m_stand, n_stand)
        if arm == "HANDOFF":
            below = th.where(V < EPS, below + 1.0, th.zeros_like(below))
            new = (below >= HYST) & ~switched
            switch_t[new] = t * DT
            switched |= new
        elif arm == "REST-ONLY":
            switched[:] = True
        with th.no_grad():
            a_s = th.clamp(m_stand.policy._predict(n_stand(obs), deterministic=True), -1, 1)
            a_r = th.clamp(m_rest.policy._predict(n_rest(obs), deterministic=True), -1, 1)
        a = th.where(switched.unsqueeze(-1), a_r, a_s)
        obs, g, dones, touts, l = env.step_tensor(th.cat([a, dstb], dim=1))
        d = inner.scene["robot"].data
        hs.append(float(d.root_link_pos_w[:, 2].mean()))
        Vs.append(float(V.mean())); swf.append(float(switched.float().mean()))
        if render:                                          # trunk tint: gray->red with load; blue once switched
            tint = GRAY + (RED - GRAY) * (W / W_MAX)
            rgba = th.tensor(tint, device=DEV, dtype=tm.geom_rgba.dtype)[None, None, :].expand(nenv, len(trunk_geoms), 4).clone()
            rgba[switched] = th.tensor(BLUE, device=DEV, dtype=tm.geom_rgba.dtype)
            tm.geom_rgba[:, trunk_geoms, :] = rgba
            # the renderer reads the HOST model's colors -> flip the whole panel BLUE once the majority
            # has switched (the unmistakable mode-switch marker); else the load tint.
            host_tint = BLUE if float(switched.float().mean()) > 0.5 else tint
            for gg in trunk_geoms: mm.geom_rgba[gg] = host_tint
            frames.append(np.asarray(env.render()))
    env.close()
    return frames, np.array(hs), np.array(Vs), np.array(swf), switch_t.cpu().numpy()


def tag(img, txt):
    img = np.ascontiguousarray(img).copy(); img[:26] = (img[:26] * 0.3).astype(img.dtype)
    if HAVE_PIL:
        im = Image.fromarray(img); ImageDraw.Draw(im).text((8, 6), txt, fill=(255, 255, 255)); img = np.asarray(im)
    return img


def graph_frame(i, data, Wv, W, H):
    t = np.arange(STEPS) * DT
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = fig.add_subplot(111); ax2 = ax.twinx()
    for arm in ARMS:
        ax.plot(t, data[arm]["h"], color=COL[arm], lw=1.0, alpha=0.25)
        ax.plot(t[:i + 1], data[arm]["h"][:i + 1], color=COL[arm], lw=2.6, label=arm)
    ax2.plot(t, Wv, color="#222", lw=1.0, alpha=0.3)
    ax2.plot(t[:i + 1], Wv[:i + 1], color="#222", lw=2.0, label="load W(t)")
    V = data["HANDOFF"]["V"]
    ax.plot(t, 0.10 + V * 1.5, color="#1a5276", ls="--", lw=1.0, alpha=0.25)
    ax.plot(t[:i + 1], 0.10 + V[:i + 1] * 1.5, color="#1a5276", ls="--", lw=1.8, label="V_stand (scaled)")
    ax.axhline(0.10 + EPS * 1.5, color="r", ls=":", lw=1.2, alpha=0.7)
    st = data["HANDOFF"]["switch_t"]; st = st[st >= 0]
    if len(st):
        lo, hi = np.percentile(st, 25), np.percentile(st, 75)
        ax.axvspan(lo, hi, color="#1a5276", alpha=0.10)
    ax.axvline(t[i], color="#555", lw=1.5)
    ax.set_xlim(0, t[-1]); ax.set_ylim(0.05, 0.30); ax2.set_ylim(0, 320)
    ax.set_xlabel("time (s)"); ax.set_ylabel("base height (m)  /  V_stand"); ax2.set_ylabel("load W (N)")
    ax.text(0.02, 0.95, "red dotted = trigger ε · shaded = switch window", transform=ax.transAxes, fontsize=8, va="top")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9); ax2.legend(loc="lower right", fontsize=8)
    fig.tight_layout(pad=0.5); fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy(); plt.close(fig)
    return buf


def timeline_figure(data, Wv, out):
    plt.rcParams.update({"font.size": 14})
    t = np.arange(STEPS) * DT
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True, gridspec_kw={"height_ratios": [1, 1.4]})
    a1.plot(t, Wv, color="#222", lw=2.5, label="load W(t)")
    a1.set_ylabel("W (N)")
    ax1b = a1.twinx()
    ax1b.plot(t, data["HANDOFF"]["V"], color="#1a5276", lw=2.5, label="V_stand(x, W)")
    ax1b.axhline(EPS, color="r", ls=":", lw=2, label="trigger ε")
    ax1b.axhline(0, color="#999", lw=0.8)
    ax1b.set_ylabel("V_stand", color="#1a5276")
    st = data["HANDOFF"]["switch_t"]; st = st[st >= 0]
    for a in (a1, a2):
        if len(st): a.axvspan(np.percentile(st, 25), np.percentile(st, 75), color="#1a5276", alpha=0.10)
    a1.legend(loc="upper left", fontsize=11); ax1b.legend(loc="center right", fontsize=11)
    a1.set_title("Certified weight-ladder handoff: the certificate contracts as the load grows, the filter switches specs")
    for arm in ARMS:
        a2.plot(t, data[arm]["h"], color=COL[arm], lw=2.8, label=arm)
    a2.set_ylabel("base height (m)"); a2.set_xlabel("time (s)"); a2.legend(fontsize=11); a2.grid(alpha=0.3)
    if len(st):
        a2.annotate("mode switch\n(V_stand < ε)", xy=(np.median(st), 0.17), xytext=(np.median(st) + 1.2, 0.24),
                    fontsize=12, arrowprops=dict(arrowstyle="->", color="#1a5276"), color="#1a5276")
    a2.annotate("STAND: fight the pull", xy=(1.0, 0.27), fontsize=11, color="#c0392b")
    a2.annotate("certified REST (h=0.10)", xy=(7.5, 0.115), fontsize=11, color="#1e8449")
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight"); print("figure ->", out)


if __name__ == "__main__":
    data = {}
    Wv = np.array([W_of(t) for t in range(STEPS)])
    grids = {}
    for arm in ARMS:
        grids[arm], h, V, swf, st_r = rollout(arm, 16, True)
        _, h128, V128, swf128, st = rollout(arm, 128, False)
        data[arm] = {"h": h128, "V": V128, "swf": swf128, "switch_t": st}
        print(f"{arm}: h_end={h128[-1]:.2f} switched={float((st>=0).mean() if len(st) else 0):.2f}"
              f" medianW@switch={W_of(int(np.median(st[st>=0])/DT)) if (st>=0).any() else float('nan'):.0f}N")
    out_dir = os.path.expanduser(_ART + "/E071-handoff")
    os.makedirs(out_dir, exist_ok=True)
    H, Wd = grids["STAND-ONLY"][0].shape[:2]
    top = [np.hstack([tag(grids[a][i][:H, :Wd], f"{a}") for a in ARMS]) for i in range(STEPS)]
    TW = top[0].shape[1]; GH = 330
    comb = [np.vstack([top[i], graph_frame(i, data, Wv, TW, GH)[:GH, :TW]]) for i in range(STEPS)]
    vp = f"{out_dir}/weight_ladder_handoff.mp4"
    imageio.mimsave(vp, comb, fps=30, macro_block_size=1)
    print("video ->", vp, f"({len(comb)} frames)")
    timeline_figure(data, Wv, f"{out_dir}/handoff_timeline.png")
