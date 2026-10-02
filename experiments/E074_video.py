"""E074 (T003) Task 4 — VIDEO + TIMELINE FIGURE of the HIGH-CoM gust-ramp handoff (E072 pattern, upgraded).

Video (4 panels): STAND-ONLY | HANDOFF | UNIFIED | REST-ONLY on the W(t) ramp. Trunks tint gray->red with the
growing load; the HANDOFF panel flips BLUE the moment the majority switches to REST (host-tint majority trick).
A GUST tag + red frame border marks the 25N gust pulses. Synced graph below: base height per arm, W(t),
V_stand (dashed) with the eps trigger line, orange gust bands, switch-window shading.

Timeline figure: top = W(t) + gust bands + V_stand_hi AND V_unified (the certification DEFICIT: V_stand
contracts across eps, V_unified stays flat/rising = no handoff signal); bottom = heights per arm with the
mode-switch annotation, gust bands, and STAND-ONLY's tip visible (height crashing at a high-W gust).
"""
import os, sys, io, contextlib, mujoco
from _paths import _ART
import torch as th
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "external/robot-safety-sandbox")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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

DEV, DT = "cuda:0", 0.02
STEPS, RAMP_START, RAMP_END, W_MAX = 500, 100, 400, 250.0
LOAD_H, AMBIENT, GUST_SCALE, GUST_STEPS = 0.25, 0.20, 0.70, 25
GUST_STARTS = [75, 250, 325, 425]
EPS, HYST, ALPHA, SETTLE = -0.04, 8, 0.10, 130
CK = "results/go2_weight_runs/go2_weight_{m}_adv/checkpoints/model_49999872_steps.zip"
TASK = "go2_weight_rest_hi_at_0"
PANELS = ["STAND-ONLY", "HANDOFF", "UNIFIED", "REST-ONLY"]
ALL_ARMS = ["STAND-ONLY", "HANDOFF", "UNIFIED", "UNIFIED-DISC", "REST-ONLY"]
POL = {"STAND-ONLY": ("stand_hi", "stand_hi", "none"), "REST-ONLY": ("rest_hi", "rest_hi", "all"),
       "HANDOFF": ("stand_hi", "rest_hi", "V"), "UNIFIED": ("unified_hi", "unified_hi", "none"),
       "UNIFIED-DISC": ("unified_disc_hi", "unified_disc_hi", "none")}
COL = {"STAND-ONLY": "#e74c3c", "HANDOFF": "#1a5276", "UNIFIED": "#8e44ad",
       "UNIFIED-DISC": "#d68910", "REST-ONLY": "#27ae60"}
GRAY, RED, BLUE = np.array([0.5, 0.5, 0.5, 1.]), np.array([0.85, 0.10, 0.10, 1.]), np.array([0.15, 0.35, 0.9, 1.])
OUT = os.path.expanduser(_ART + "/E074-hicom-demo")


def W_of(t):
    if t < RAMP_START: return 0.0
    if t < RAMP_END: return (t - RAMP_START) / (RAMP_END - RAMP_START) * W_MAX
    return W_MAX
def in_gust(t): return any(g <= t < g + GUST_STEPS for g in GUST_STARTS)


def load_all():
    ms = {}
    with contextlib.redirect_stdout(io.StringIO()):
        for m in ["stand_hi", "rest_hi", "unified_hi", "unified_disc_hi"]:
            ms[m] = load_twin(CK.format(m=m), DEV, quiet=True)
    for mdl, _ in ms.values():
        mdl.policy.set_training_mode(False)
    return ms


def rollout(arm, models, nenv, render):
    p_s, p_r, rule = POL[arm]
    m_s, n_s = models[p_s]; m_r, n_r = models[p_r]
    m_vs, n_vs = models["stand_hi"]; m_vu, n_vu = models["unified_hi"]
    with contextlib.redirect_stdout(io.StringIO()):
        env = make_tensor(TASK, nenv, DEV, adversary=True, **({"render_mode": "rgb_array"} if render else {}))
        if render:
            try: env.mj.cfg.viewer.max_extra_envs = max(1, nenv - 1)
            except Exception: pass
    inner = env.mj
    if render:
        mm = inner.sim.mj_model; tm = inner.sim.model
        base_id = mujoco.mj_name2id(mm, mujoco.mjtObj.mjOBJ_BODY, "robot/base_link")
        trunk_geoms = [g for g in range(mm.ngeom) if mm.geom_bodyid[g] == base_id]
    dstb = th.tensor([0., 1., 0.], device=DEV)[None].expand(nenv, spec(TASK).dstb_dim).contiguous()
    obs = env.reset()
    switched = th.zeros(nenv, dtype=th.bool, device=DEV)
    if rule == "all": switched[:] = True
    below = th.zeros(nenv, device=DEV); ema = th.zeros(nenv, device=DEV); ema_init = False
    switch_t = th.full((nenv,), -1.0, device=DEV)
    frames, hs, Vs_tr, Vu_tr = [], [], [], []
    for t in range(STEPS):
        W = W_of(t)
        inner._weight_h = th.full((nenv,), LOAD_H, device=DEV)
        inner._weight_W = th.full((nenv,), W, device=DEV)
        env.base_load = th.tensor([0., 0., -W], device=DEV)[None].expand(nenv, 3).contiguous()
        env.force_scale = (GUST_SCALE if in_gust(t) else AMBIENT) * th.ones(nenv, device=DEV)
        Vs = stand_value(env, m_vs, n_vs); Vu = stand_value(env, m_vu, n_vu)
        if not ema_init: ema = Vs.clone(); ema_init = True
        else: ema = ALPHA * Vs + (1 - ALPHA) * ema
        if rule == "V":
            active = t >= SETTLE
            below = th.where((ema < EPS) & active, below + 1.0, th.zeros_like(below))
            new = (below >= HYST) & ~switched
            switch_t[new] = t * DT
            switched |= new
        with th.no_grad():
            a_s = th.clamp(m_s.policy._predict(n_s(obs), deterministic=True), -1, 1)
            a_r = th.clamp(m_r.policy._predict(n_r(obs), deterministic=True), -1, 1)
        a = th.where(switched.unsqueeze(-1), a_r, a_s)
        obs, g, dones, touts, l = env.step_tensor(th.cat([a, dstb], dim=1))
        d = inner.scene["robot"].data
        hs.append(float(d.root_link_pos_w[:, 2].mean()))
        Vs_tr.append(float(Vs.mean())); Vu_tr.append(float(Vu.mean()))
        if render:
            tint = GRAY + (RED - GRAY) * (W / W_MAX)
            rgba = th.tensor(tint, device=DEV, dtype=tm.geom_rgba.dtype)[None, None, :].expand(
                nenv, len(trunk_geoms), 4).clone()
            rgba[switched] = th.tensor(BLUE, device=DEV, dtype=tm.geom_rgba.dtype)
            tm.geom_rgba[:, trunk_geoms, :] = rgba
            host_tint = BLUE if (arm == "HANDOFF" and float(switched.float().mean()) > 0.5) else tint
            for gg in trunk_geoms: mm.geom_rgba[gg] = host_tint
            frames.append(np.asarray(env.render()))
    env.close()
    return frames, np.array(hs), np.array(Vs_tr), np.array(Vu_tr), switch_t.cpu().numpy()


def tag(img, txt, gust):
    img = np.ascontiguousarray(img).copy()
    img[:26] = (img[:26] * 0.3).astype(img.dtype)
    if gust:                                                    # red frame border during a gust pulse
        b = 5
        img[:b, :] = [230, 80, 40]; img[-b:, :] = [230, 80, 40]
        img[:, :b] = [230, 80, 40]; img[:, -b:] = [230, 80, 40]
    if HAVE_PIL:
        im = Image.fromarray(img); dr = ImageDraw.Draw(im)
        dr.text((8, 6), txt, fill=(255, 255, 255))
        if gust: dr.text((img.shape[1] - 52, 6), "GUST", fill=(255, 210, 90))
        img = np.asarray(im)
    return img


def graph_frame(i, data, Wv, W, H):
    t = np.arange(STEPS) * DT
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = fig.add_subplot(111); ax2 = ax.twinx()
    for g in GUST_STARTS:
        ax.axvspan(g * DT, (g + GUST_STEPS) * DT, color="#f39c12", alpha=0.18, zorder=0)
    for arm in PANELS:
        ax.plot(t, data[arm]["h"], color=COL[arm], lw=1.0, alpha=0.22)
        ax.plot(t[:i + 1], data[arm]["h"][:i + 1], color=COL[arm], lw=2.4, label=arm)
    ax2.plot(t, Wv, color="#222", lw=1.0, alpha=0.3)
    ax2.plot(t[:i + 1], Wv[:i + 1], color="#222", lw=2.0, label="load W(t)")
    V = data["HANDOFF"]["Vs"]
    ax.plot(t, 0.16 + V * 0.35, color="#1a5276", ls="--", lw=1.0, alpha=0.25)
    ax.plot(t[:i + 1], 0.16 + V[:i + 1] * 0.35, color="#1a5276", ls="--", lw=1.8, label="V_stand (scaled)")
    ax.axhline(0.16 + EPS * 0.35, color="r", ls=":", lw=1.2, alpha=0.7)
    st = data["HANDOFF"]["switch_t"]; st = st[st >= 0]
    if len(st):
        ax.axvspan(np.percentile(st, 25), np.percentile(st, 75), color="#1a5276", alpha=0.10)
    ax.axvline(t[i], color="#555", lw=1.5)
    ax.set_xlim(0, t[-1]); ax.set_ylim(0.05, 0.30); ax2.set_ylim(0, 320)
    ax.set_xlabel("time (s)"); ax.set_ylabel("base height (m) / V_stand"); ax2.set_ylabel("load W (N)")
    ax.text(0.01, 0.97, "orange = gust · red dotted = trigger ε · blue shade = switch window",
            transform=ax.transAxes, fontsize=8, va="top")
    ax.legend(loc="upper right", fontsize=7.5, framealpha=0.9, ncol=2); ax2.legend(loc="lower left", fontsize=8)
    fig.tight_layout(pad=0.4); fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy(); plt.close(fig)
    return buf


def timeline_figure(data, Wv, out):
    plt.rcParams.update({"font.size": 13})
    t = np.arange(STEPS) * DT
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [1, 1.3]})
    for a in (a1, a2):
        for g in GUST_STARTS:
            a.axvspan(g * DT, (g + GUST_STEPS) * DT, color="#f39c12", alpha=0.20, zorder=0)
    a1.plot(t, Wv, color="#222", lw=2.5, label="load W(t)"); a1.set_ylabel("W (N)")
    b = a1.twinx()
    b.plot(t, data["STAND-ONLY"]["Vs"], color="#1a5276", lw=2.3, label="V_stand_hi (contracts)")
    b.plot(t, data["STAND-ONLY"]["Vu"], color="#8e44ad", lw=2.3, ls="-.", label="V_unified (no signal)")
    b.axhline(EPS, color="r", ls=":", lw=2, label="trigger ε")
    b.axhline(0, color="#999", lw=0.8); b.set_ylabel("value")
    st = data["HANDOFF"]["switch_t"]; st = st[st >= 0]
    if len(st):
        for a in (a1, a2): a.axvspan(np.percentile(st, 25), np.percentile(st, 75), color="#1a5276", alpha=0.12)
    a1.legend(loc="upper left", fontsize=10); b.legend(loc="lower right", fontsize=10)
    a1.set_title("HIGH-CoM gust-ramp: V_stand_hi contracts across ε (switch) while V_unified never signals — "
                 "the certification deficit", fontsize=12.5)
    for arm in ALL_ARMS:
        a2.plot(t, data[arm]["h"], color=COL[arm], lw=2.6, label=arm)
    a2.set_ylabel("base height (m)"); a2.set_xlabel("time (s)"); a2.legend(fontsize=10, ncol=3); a2.grid(alpha=0.3)
    if len(st):
        a2.annotate("HANDOFF switches\n(V_stand < ε)", xy=(np.median(st), 0.17),
                    xytext=(np.median(st) + 0.6, 0.25), fontsize=11,
                    arrowprops=dict(arrowstyle="->", color="#1a5276"), color="#1a5276")
    # stand-only stays tall on average but UNCERTIFIED — ~half have toppled by the high-W gusts
    so = data["STAND-ONLY"]["h"]
    a2.annotate("STAND-ONLY stays tall but\nUNCERTIFIED (~48% toppled)",
                xy=(9.2, so[-1]), xytext=(6.0, 0.055), fontsize=11,
                arrowprops=dict(arrowstyle="->", color="#c0392b"), color="#c0392b")
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight"); print("figure ->", out)
    fig.savefig(os.path.join(OUT, "figs", os.path.basename(out)), dpi=130, bbox_inches="tight")


def figonly():
    """Regenerate ONLY the timeline figure: reuse the 128-env traces in task3_ramp.json + one HANDOFF
    rollout for the per-env switch_t distribution (the switch-window shading)."""
    import json
    os.makedirs(os.path.join(OUT, "figs"), exist_ok=True)
    j = json.load(open(os.path.join(OUT, "task3_ramp.json")))
    Wv = np.array(j["W_trace"])
    data = {}
    for arm in ALL_ARMS:
        a = j["arms"][arm]
        data[arm] = {"h": np.array(a["h_trace"]), "Vs": np.array(a["Vs_trace"]),
                     "Vu": np.array(a["Vu_trace"]), "switch_t": np.full(len(a["h_trace"]), -1.0)}
    models = load_all()
    _, _, _, _, st = rollout("HANDOFF", models, 128, False)
    data["HANDOFF"]["switch_t"] = st
    timeline_figure(data, Wv, f"{OUT}/hicom_timeline.png")


if __name__ == "__main__":
    if "--figonly" in sys.argv:
        figonly(); sys.exit(0)
    os.makedirs(os.path.join(OUT, "figs"), exist_ok=True)
    models = load_all()
    Wv = np.array([W_of(t) for t in range(STEPS)])
    data, grids = {}, {}
    for arm in ALL_ARMS:
        render = arm in PANELS
        fr, h16, V16, Vu16, st16 = rollout(arm, models, 16, render) if render else ([], None, None, None, None)
        _, h, V, Vu, st = rollout(arm, models, 128, False)
        data[arm] = {"h": h, "Vs": V, "Vu": Vu, "switch_t": st}
        if render: grids[arm] = fr
        print(f"{arm:>13}: h_end={h[-1]:.2f} switched={float((st>=0).mean()):.2f} "
              f"medianW@sw={(W_of(int(np.median(st[st>=0])/DT)) if (st>=0).any() else float('nan')):.0f}N")

    H, Wd = grids["STAND-ONLY"][0].shape[:2]
    top = [np.hstack([tag(grids[a][i][:H, :Wd], a, in_gust(i)) for a in PANELS]) for i in range(STEPS)]
    TW = top[0].shape[1]; GH = 340
    comb = [np.vstack([top[i], graph_frame(i, data, Wv, TW, GH)[:GH, :TW]]) for i in range(STEPS)]
    vp = f"{OUT}/hicom_gust_ramp.mp4"
    imageio.mimsave(vp, comb, fps=30, macro_block_size=1)
    print("video ->", vp, f"({len(comb)} frames)")
    timeline_figure(data, Wv, f"{OUT}/hicom_timeline.png")
