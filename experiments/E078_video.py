"""E078 (T004, DEMO 2) Task 4 — VIDEO + TIMELINE of the COMPOUND leg-death-while-loaded handoff.

3 panels: STAND-ONLY | HANDOFF | REST-ONLY, under the compound ODD (constant load W=80@h=0.25 + FR-leg death
θ ramp 1.0→0.1 over t=2..6s, hold). 10N ambient + 25N gusts at t={1.5s, 6.5s, 8s}. Visual tricks:
  * FR leg (bodies 6,7,8) tints gray→RED as θ drops (E061/E076 trick) — the dying motor.
  * HANDOFF trunk (body 2, base_link) tints gray→BLUE per-env when that env fires the V-trigger handoff.
  * gust frames get a red border + "GUST" tag.
Synced graph: θ(t), V_compound_stand + ε (CONTRACTS — unlike the flat unloaded leg), base heights per arm,
switch window. Plus a static timeline figure (compound_timeline.png).
"""
import os, sys, io, contextlib
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
from robot_safety_sandbox.envs.go2_broken_leg.env_cfg import _ensure_fr_cache
from _value_util import stand_value

DEV, DT = "cuda:0", 0.02
STEPS = 500
T_HOLD, T_RAMP_END = 100, 300              # θ=1.0 to t=2s; ramp to 0.1 by t=6s; hold
THETA_HI, THETA_LO = 1.0, 0.1
W, LOAD_H, SLAM_CAP = 80.0, 0.25, 184.0
GUST_STARTS, GUST_STEPS = [75, 325, 400], 25   # t=1.5s (healthy), 6.5s, 8s (dying leg)
AMBIENT, GUST_SCALE = 0.20, 0.70
EPS, HYST, ALPHA, SETTLE = -0.14, 6, 0.15, 110  # Task-2 trigger
FR_BODIES = (6, 7, 8); TRUNK_BODIES = (2,)
GRAY = np.array([0.5, 0.5, 0.5, 1.]); RED = np.array([0.9, 0.08, 0.08, 1.]); BLUE = np.array([0.15, 0.4, 0.95, 1.])
STAND = "results/go2_compound_runs/go2_compound_stand_adv/checkpoints/model_49999872_steps.zip"
REST  = "results/go2_compound_runs/go2_compound_rest_adv/checkpoints/model_49999872_steps.zip"
PULL = th.tensor([0., 1., 0.])
TASK = "go2_compound_rest"
PANELS = ["STAND-ONLY", "HANDOFF", "REST-ONLY"]
RULE = {"STAND-ONLY": "none", "HANDOFF": "V", "REST-ONLY": "always"}
COL = {"STAND-ONLY": "#e74c3c", "HANDOFF": "#2980b9", "REST-ONLY": "#27ae60"}
OUT = os.path.expanduser("~/artifacts/odd-conditioned/E078-compound-demo")


def theta_of(t):
    if t < T_HOLD: return THETA_HI
    if t < T_RAMP_END:
        return THETA_HI + (THETA_LO - THETA_HI) * (t - T_HOLD) / (T_RAMP_END - T_HOLD)
    return THETA_LO
def in_gust(t): return any(g <= t < g + GUST_STEPS for g in GUST_STARTS)


def rollout(arm, nenv, render, vmodel):
    rule = RULE[arm]
    with contextlib.redirect_stdout(io.StringIO()):
        env = make_tensor(TASK, nenv, DEV, adversary=True, **({"render_mode": "rgb_array"} if render else {}))
        if render:
            try: env.mj.cfg.viewer.max_extra_envs = max(1, nenv - 1)
            except Exception: pass
        m_s, n_s = load_twin(STAND if arm != "REST-ONLY" else REST, DEV, quiet=True)
        m_r, n_r = load_twin(REST, DEV, quiet=True)
    m_vs, n_vs = vmodel
    inner = env.mj; _ensure_fr_cache(inner)
    ids, nominal = inner._fr_act_ids, inner._fr_nominal_forcerange
    if render:
        mm = inner.sim.mj_model; tm = inner.sim.model
        fr_geoms = [g for g in range(mm.ngeom) if mm.geom_bodyid[g] in FR_BODIES]
        tk_geoms = [g for g in range(mm.ngeom) if mm.geom_bodyid[g] in TRUNK_BODIES]
    dstb = (PULL / PULL.norm()).to(DEV)[None].expand(nenv, spec(TASK).dstb_dim).contiguous()

    def drive(th_):
        inner.sim.model.actuator_forcerange[:, ids, :] = nominal * float(th_)
        inner._fr_torque_frac[:] = float(th_)
        inner._weight_W = th.full((nenv,), W, device=DEV)
        inner._weight_h = th.full((nenv,), LOAD_H, device=DEV)
        env.base_load = th.tensor([0., 0., -W], device=DEV)[None].expand(nenv, 3).contiguous()

    obs = env.reset()
    switched = th.zeros(nenv, dtype=th.bool, device=DEV) if rule != "always" else th.ones(nenv, dtype=th.bool, device=DEV)
    below = th.zeros(nenv, device=DEV); ema = th.zeros(nenv, device=DEV); ema_init = False
    frames, hs, Vs_tr, sw_tr = [], [], [], []
    for t in range(STEPS):
        theta = theta_of(t); drive(theta)
        env.force_scale = (GUST_SCALE if in_gust(t) else AMBIENT) * th.ones(nenv, device=DEV)
        Vs = stand_value(env, m_vs, n_vs)
        if not ema_init: ema = Vs.clone(); ema_init = True
        else: ema = ALPHA * Vs + (1 - ALPHA) * ema
        if rule == "V":
            active = t >= SETTLE
            below = th.where((ema < EPS) & active, below + 1.0, th.zeros_like(below))
            switched = switched | ((below >= HYST) & ~switched)
        with th.no_grad():
            a_s = th.clamp(m_s.policy._predict(n_s(obs), deterministic=True), -1, 1)
            a_r = th.clamp(m_r.policy._predict(n_r(obs), deterministic=True), -1, 1)
        a = th.where(switched.unsqueeze(-1), a_r, a_s) if rule != "none" else a_s
        obs, g, dones, touts, l = env.step_tensor(th.cat([a, dstb], dim=1))
        d = inner.scene["robot"].data
        hs.append(float(d.root_link_pos_w[:, 2].mean())); Vs_tr.append(float(Vs.mean()))
        sw_tr.append(float(switched.float().mean()))
        if render:
            frac = (THETA_HI - theta) / (THETA_HI - THETA_LO)
            tint = GRAY + (RED - GRAY) * frac
            tm.geom_rgba[:, fr_geoms, :] = th.tensor(tint, device=DEV, dtype=tm.geom_rgba.dtype)
            for gg in fr_geoms: mm.geom_rgba[gg] = tint
            # trunk → BLUE per-world when that env has handed off (only the HANDOFF arm dynamically switches)
            if arm == "HANDOFF":
                sw = switched.float().reshape(-1, 1, 1)                      # [nenv,1,1]
                tk = (th.tensor(GRAY, device=DEV, dtype=tm.geom_rgba.dtype)[None, None, :]
                      + (th.tensor(BLUE, device=DEV, dtype=tm.geom_rgba.dtype) - th.tensor(GRAY, device=DEV, dtype=tm.geom_rgba.dtype))[None, None, :] * sw)
                tm.geom_rgba[:, tk_geoms, :] = tk
                mm.geom_rgba[tk_geoms[0]] = (GRAY + (BLUE - GRAY) * float(switched.float().mean() > 0.5))
            frames.append(np.asarray(env.render()))
    env.close()
    return frames, np.array(hs), np.array(Vs_tr), np.array(sw_tr)


def tag(img, txt, gust, sub=None):
    img = np.ascontiguousarray(img).copy(); img[:26] = (img[:26] * 0.3).astype(img.dtype)
    if gust:
        b = 5
        img[:b, :] = [230, 80, 40]; img[-b:, :] = [230, 80, 40]
        img[:, :b] = [230, 80, 40]; img[:, -b:] = [230, 80, 40]
    if HAVE_PIL:
        im = Image.fromarray(img); dr = ImageDraw.Draw(im); dr.text((8, 6), txt, fill=(255, 255, 255))
        if sub: dr.text((8, 16), sub, fill=(150, 200, 255))
        if gust: dr.text((img.shape[1] - 52, 6), "GUST", fill=(255, 210, 90))
        img = np.asarray(im)
    return img


def graph_frame(i, data, thv, W_, H):
    t = np.arange(STEPS) * DT
    fig = plt.figure(figsize=(W_ / 100, H / 100), dpi=100)
    ax = fig.add_subplot(111); ax2 = ax.twinx()
    for g in GUST_STARTS:
        ax.axvspan(g * DT, (g + GUST_STEPS) * DT, color="#f39c12", alpha=0.18, zorder=0)
    ax.plot(t, thv, color="#333", lw=1.0, alpha=0.3)
    ax.plot(t[:i + 1], thv[:i + 1], color="#333", lw=2.2, label="θ (FR torque)")
    for arm in PANELS:
        ax2.plot(t, data[arm]["h"], color=COL[arm], lw=1.0, alpha=0.25)
        ax2.plot(t[:i + 1], data[arm]["h"][:i + 1], color=COL[arm], lw=2.2, label=f"{arm} h")
    V = data["HANDOFF"]["Vs"]
    ax2.plot(t, 0.17 + V * 0.10, color="#1a5276", ls="--", lw=1.0, alpha=0.25)
    ax2.plot(t[:i + 1], 0.17 + V[:i + 1] * 0.10, color="#1a5276", ls="--", lw=1.8, label="V_stand (contracts)")
    ax2.axhline(0.17 + EPS * 0.10, color="r", ls=":", lw=1.1, alpha=0.7)
    # switch window marker (median handoff step)
    sw = data["HANDOFF"]["sw"]
    ax.axvline(t[i], color="#555", lw=1.5)
    ax.set_xlim(0, t[-1]); ax.set_ylim(0, 1.1); ax2.set_ylim(0.05, 0.30)
    ax.set_xlabel("time (s)"); ax.set_ylabel("θ (FR torque frac)", color="#333")
    ax2.set_ylabel("base height (m) / V_stand")
    ax.text(0.01, 0.045, "V_compound_stand CONTRACTS as the loaded leg dies → crosses ε → certified handoff (trunk→BLUE)",
            transform=ax.transAxes, fontsize=8, color="#1a5276")
    ax.legend(loc="upper right", fontsize=8); ax2.legend(loc="lower left", fontsize=7, ncol=2)
    fig.tight_layout(pad=0.4); fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy(); plt.close(fig)
    return buf


def timeline_figure(data, thv, out):
    plt.rcParams.update({"font.size": 13})
    t = np.arange(STEPS) * DT
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 8.2), sharex=True, gridspec_kw={"height_ratios": [1, 1.3]})
    for a in (a1, a2):
        for g in GUST_STARTS:
            a.axvspan(g * DT, (g + GUST_STEPS) * DT, color="#f39c12", alpha=0.20, zorder=0)
    a1.plot(t, thv, color="#333", lw=2.5, label="θ (FR torque frac)"); a1.set_ylabel("θ"); a1.set_ylim(0, 1.1)
    b = a1.twinx()
    b.plot(t, data["HANDOFF"]["Vs"], color="#1a5276", lw=2.3, label="V_compound_stand (CONTRACTS)")
    b.axhline(EPS, color="r", ls=":", lw=2, label=f"ε={EPS:+.2f} (crossed → handoff)")
    b.axhline(0, color="#999", lw=0.8); b.set_ylabel("value"); b.set_ylim(-0.45, 0.15)
    # switch window
    sw = data["HANDOFF"]["sw"]; fired = np.where(sw > 0.5)[0]
    if len(fired):
        b.axvspan(t[max(fired[0]-6,0)], t[fired[0]+2 if fired[0]+2 < STEPS else -1], color="#2980b9", alpha=0.12)
        a1.text(t[fired[0]], 0.9, "handoff\nwindow", color="#2980b9", fontsize=10, ha="center")
    a1.legend(loc="upper right", fontsize=10); b.legend(loc="lower left", fontsize=10)
    a1.set_title("Compound leg-death-while-loaded: V_compound_stand CONTRACTS as θ→0.1 (discrim 1.54) → crosses ε "
                 "→ certified handoff\n(contrast the FLAT unloaded-leg certificate — the same axis is absorbable "
                 "without the load)", fontsize=11)
    for arm in PANELS:
        a2.plot(t, data[arm]["h"], color=COL[arm], lw=2.6, label=arm)
    a2.set_ylabel("base height (m)"); a2.set_xlabel("time (s)"); a2.legend(fontsize=11); a2.grid(alpha=0.3)
    a2.annotate("STAND-ONLY fights on the dying leg\n→ tips at the late gusts",
                xy=(8.0, data["STAND-ONLY"]["h"][400]), xytext=(2.4, 0.15), fontsize=10.5,
                arrowprops=dict(arrowstyle="->", color="#c0392b"), color="#c0392b")
    a2.annotate("HANDOFF descends to certified rest",
                xy=(7.0, data["HANDOFF"]["h"][350]), xytext=(0.6, 0.115), fontsize=10.5,
                arrowprops=dict(arrowstyle="->", color="#2471a3"), color="#2471a3")
    fig.tight_layout()
    os.makedirs(os.path.join(OUT, "figs"), exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "figs", os.path.basename(out)), dpi=130, bbox_inches="tight")
    print("figure ->", out)


if __name__ == "__main__":
    with contextlib.redirect_stdout(io.StringIO()):
        vmodel = load_twin(STAND, DEV, quiet=True)          # V_compound_stand readout twin
    thv = np.array([theta_of(t) for t in range(STEPS)])
    data, grids = {}, {}
    for arm in PANELS:
        fr, _, _, _ = rollout(arm, 12, True, vmodel)
        _, h, V, sw = rollout(arm, 128, False, vmodel)
        data[arm] = {"h": h, "Vs": V, "sw": sw}; grids[arm] = fr
        print(f"{arm:>11}: h_end={h[-1]:.2f}  V_end={V[-1]:+.3f}  switched={sw[-1]:.2f}")
    if "--figonly" not in sys.argv:
        H, Wd = grids["STAND-ONLY"][0].shape[:2]
        subs = {a: (f"handoff {data[a]['sw'][i]*100:.0f}%" if a == "HANDOFF" else None) for a in PANELS for i in [0]}
        top = [np.hstack([tag(grids[a][i][:H, :Wd], a, in_gust(i),
                              f"handoff {data[a]['sw'][i]*100:.0f}%" if a == "HANDOFF" else None) for a in PANELS])
               for i in range(STEPS)]
        TW = top[0].shape[1]; GH = 340
        comb = [np.vstack([top[i], graph_frame(i, data, thv, TW, GH)[:GH, :TW]]) for i in range(STEPS)]
        vp = f"{OUT}/compound_death_demo.mp4"
        imageio.mimsave(vp, comb, fps=30, macro_block_size=1)
        print("video ->", vp, f"({len(comb)} frames)")
    timeline_figure(data, thv, f"{OUT}/compound_timeline.png")
