"""E076 (T004) Part B.4 — VIDEO + TIMELINE of the leg-death demo (BEHAVIORAL, no handoff).

The B.2 gate found V_leg_stand FLAT vs θ (discrimination 0.18 — the reactive stance policy absorbs the
single-axis leg ODD; the E064/E068 lesson), so per directive the HANDOFF is SKIPPED. This video therefore
shows the two MODES behaviorally — STAND-ONLY | REST-ONLY — under a leg-death ramp, plus the flat V_leg_stand
trace that explains why no certified trigger exists.

Leg-death ramp: θ=1.0 until t=2s, ramp to 0.1 by t=6s, hold. 10N ambient pull + one 25N gust (→35N, 0.5s) at
t=7s. FR leg (bodies 6,7,8) tints gray->RED as θ drops (E061 trick). Synced graph below: θ(t), V_leg_stand +
ε, base heights per arm, gust band.
"""
import os, sys, io, contextlib, mujoco
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
STEPS = 400
T_HOLD, T_RAMP_END = 100, 300              # θ=1.0 to t=2s; ramp to 0.1 by t=6s; hold
THETA_HI, THETA_LO = 1.0, 0.1
GUST_START, GUST_STEPS = 350, 25           # 25N gust (→35N) at t=7s
AMBIENT, GUST_SCALE = 0.20, 0.70
EPS = -0.005                               # B.2 midpoint (flat certificate — shown for context, not used)
FR_BODIES = (6, 7, 8)
GRAY, RED = np.array([0.5, 0.5, 0.5, 1.]), np.array([0.9, 0.08, 0.08, 1.])
LEG_STAND = "results/go2_leg_family_runs/go2_leg_stand_adv/checkpoints/model_49999872_steps.zip"
LEG_REST = "results/go2_leg_family_runs/go2_leg_rest_adv/checkpoints/model_49999872_steps.zip"
PULL = th.tensor([0., 1., 0.])
TASK = "go2_leg_rest"
PANELS = ["STAND-ONLY", "REST-ONLY"]
POL = {"STAND-ONLY": LEG_STAND, "REST-ONLY": LEG_REST}
COL = {"STAND-ONLY": "#e74c3c", "REST-ONLY": "#27ae60"}
OUT = os.path.expanduser("~/artifacts/odd-conditioned/E076-leg-demo")


def theta_of(t):
    if t < T_HOLD: return THETA_HI
    if t < T_RAMP_END:
        return THETA_HI + (THETA_LO - THETA_HI) * (t - T_HOLD) / (T_RAMP_END - T_HOLD)
    return THETA_LO
def in_gust(t): return GUST_START <= t < GUST_START + GUST_STEPS


def rollout(arm, nenv, render, vmodel):
    ck = POL[arm]
    with contextlib.redirect_stdout(io.StringIO()):
        env = make_tensor(TASK, nenv, DEV, adversary=True, **({"render_mode": "rgb_array"} if render else {}))
        if render:
            try: env.mj.cfg.viewer.max_extra_envs = max(1, nenv - 1)
            except Exception: pass
        model, norm = load_twin(ck, DEV, quiet=True)
    m_vs, n_vs = vmodel
    inner = env.mj; _ensure_fr_cache(inner)
    ids, nominal = inner._fr_act_ids, inner._fr_nominal_forcerange
    if render:
        mm = inner.sim.mj_model; tm = inner.sim.model
        fr_geoms = [g for g in range(mm.ngeom) if mm.geom_bodyid[g] in FR_BODIES]
    dstb = (PULL / PULL.norm()).to(DEV)[None].expand(nenv, spec(TASK).dstb_dim).contiguous()
    obs = env.reset()
    frames, hs, Vs_tr = [], [], []
    for t in range(STEPS):
        theta = theta_of(t)
        inner.sim.model.actuator_forcerange[:, ids, :] = nominal * float(theta)
        inner._fr_torque_frac[:] = float(theta)
        env.force_scale = (GUST_SCALE if in_gust(t) else AMBIENT) * th.ones(nenv, device=DEV)
        Vs = stand_value(env, m_vs, n_vs)
        with th.no_grad():
            a = th.clamp(model.policy._predict(norm(obs), deterministic=True), -1, 1)
        obs, g, dones, touts, l = env.step_tensor(th.cat([a, dstb], dim=1))
        d = inner.scene["robot"].data
        hs.append(float(d.root_link_pos_w[:, 2].mean())); Vs_tr.append(float(Vs.mean()))
        if render:
            frac = (THETA_HI - theta) / (THETA_HI - THETA_LO)                # 0 at healthy, 1 at θ=0.1
            tint = GRAY + (RED - GRAY) * frac
            tm.geom_rgba[:, fr_geoms, :] = th.tensor(tint, device=DEV, dtype=tm.geom_rgba.dtype)
            for gg in fr_geoms:
                mm.geom_rgba[gg] = tint
            frames.append(np.asarray(env.render()))
    env.close()
    return frames, np.array(hs), np.array(Vs_tr)


def tag(img, txt, gust):
    img = np.ascontiguousarray(img).copy(); img[:26] = (img[:26] * 0.3).astype(img.dtype)
    if gust:
        b = 5
        img[:b, :] = [230, 80, 40]; img[-b:, :] = [230, 80, 40]
        img[:, :b] = [230, 80, 40]; img[:, -b:] = [230, 80, 40]
    if HAVE_PIL:
        im = Image.fromarray(img); dr = ImageDraw.Draw(im); dr.text((8, 6), txt, fill=(255, 255, 255))
        if gust: dr.text((img.shape[1] - 52, 6), "GUST", fill=(255, 210, 90))
        img = np.asarray(im)
    return img


def graph_frame(i, data, thv, W, H):
    t = np.arange(STEPS) * DT
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = fig.add_subplot(111); ax2 = ax.twinx()
    if GUST_START < STEPS:
        ax.axvspan(GUST_START * DT, (GUST_START + GUST_STEPS) * DT, color="#f39c12", alpha=0.18, zorder=0)
    ax.plot(t, thv, color="#333", lw=1.0, alpha=0.3)
    ax.plot(t[:i + 1], thv[:i + 1], color="#333", lw=2.2, label="θ (FR torque)")
    for arm in PANELS:
        ax2.plot(t, data[arm]["h"], color=COL[arm], lw=1.0, alpha=0.25)
        ax2.plot(t[:i + 1], data[arm]["h"][:i + 1], color=COL[arm], lw=2.4, label=f"{arm} height")
    V = data["STAND-ONLY"]["Vs"]
    ax2.plot(t, 0.16 + V * 0.30, color="#1a5276", ls="--", lw=1.0, alpha=0.25)
    ax2.plot(t[:i + 1], 0.16 + V[:i + 1] * 0.30, color="#1a5276", ls="--", lw=1.8, label="V_leg_stand (flat)")
    ax2.axhline(0.16 + EPS * 0.30, color="r", ls=":", lw=1.1, alpha=0.6)
    ax.axvline(t[i], color="#555", lw=1.5)
    ax.set_xlim(0, t[-1]); ax.set_ylim(0, 1.1); ax2.set_ylim(0.05, 0.30)
    ax.set_xlabel("time (s)"); ax.set_ylabel("θ (FR torque frac)", color="#333")
    ax2.set_ylabel("base height (m) / V_leg_stand")
    ax.text(0.01, 0.05, "V_leg_stand stays flat (~0) as the leg dies → NO certified handoff trigger (absorbed ODD)",
            transform=ax.transAxes, fontsize=8, color="#c0392b")
    ax.legend(loc="upper right", fontsize=8); ax2.legend(loc="lower left", fontsize=8, ncol=2)
    fig.tight_layout(pad=0.4); fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy(); plt.close(fig)
    return buf


def timeline_figure(data, thv, out):
    plt.rcParams.update({"font.size": 13})
    t = np.arange(STEPS) * DT
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [1, 1.3]})
    for a in (a1, a2):
        a.axvspan(GUST_START * DT, (GUST_START + GUST_STEPS) * DT, color="#f39c12", alpha=0.20, zorder=0)
    a1.plot(t, thv, color="#333", lw=2.5, label="θ (FR torque frac)"); a1.set_ylabel("θ"); a1.set_ylim(0, 1.1)
    b = a1.twinx()
    b.plot(t, data["STAND-ONLY"]["Vs"], color="#1a5276", lw=2.3, label="V_leg_stand (FLAT — absorbed)")
    b.axhline(EPS, color="r", ls=":", lw=2, label="ε (unused: no crossing)")
    b.axhline(0, color="#999", lw=0.8); b.set_ylabel("value"); b.set_ylim(-0.3, 0.3)
    a1.legend(loc="upper right", fontsize=10); b.legend(loc="lower left", fontsize=10)
    a1.set_title("Leg-death ramp: V_leg_stand stays FLAT as θ→0.1 — the single-axis leg ODD is reactively "
                 "absorbed, so the certificate emits no handoff signal (contrast the weight ladder)", fontsize=11.5)
    for arm in PANELS:
        a2.plot(t, data[arm]["h"], color=COL[arm], lw=2.6, label=arm)
    a2.set_ylabel("base height (m)"); a2.set_xlabel("time (s)"); a2.legend(fontsize=11); a2.grid(alpha=0.3)
    a2.annotate("STAND-ONLY keeps standing on the weak leg\n(reactively absorbs the degradation)",
                xy=(5.0, data["STAND-ONLY"]["h"][250]), xytext=(1.0, 0.15), fontsize=10.5,
                arrowprops=dict(arrowstyle="->", color="#c0392b"), color="#c0392b")
    a2.annotate("REST-ONLY sits low & safe throughout",
                xy=(5.0, data["REST-ONLY"]["h"][250]), xytext=(1.0, 0.115), fontsize=10.5,
                arrowprops=dict(arrowstyle="->", color="#1e8449"), color="#1e8449")
    fig.tight_layout()
    os.makedirs(os.path.join(OUT, "figs"), exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "figs", os.path.basename(out)), dpi=130, bbox_inches="tight")
    print("figure ->", out)


if __name__ == "__main__":
    with contextlib.redirect_stdout(io.StringIO()):
        vmodel = load_twin(LEG_STAND, DEV, quiet=True)          # V_leg_stand readout twin
    thv = np.array([theta_of(t) for t in range(STEPS)])
    data, grids = {}, {}
    for arm in PANELS:
        fr, _, _ = rollout(arm, 16, True, vmodel)
        _, h, V = rollout(arm, 128, False, vmodel)
        data[arm] = {"h": h, "Vs": V}; grids[arm] = fr
        print(f"{arm:>12}: h_end={h[-1]:.2f}  V_end={V[-1]:+.3f}")
    if "--figonly" not in sys.argv:
        H, Wd = grids["STAND-ONLY"][0].shape[:2]
        top = [np.hstack([tag(grids[a][i][:H, :Wd], a, in_gust(i)) for a in PANELS]) for i in range(STEPS)]
        TW = top[0].shape[1]; GH = 340
        comb = [np.vstack([top[i], graph_frame(i, data, thv, TW, GH)[:GH, :TW]]) for i in range(STEPS)]
        vp = f"{OUT}/leg_death_demo.mp4"
        imageio.mimsave(vp, comb, fps=30, macro_block_size=1)
        print("video ->", vp, f"({len(comb)} frames)")
    timeline_figure(data, thv, f"{OUT}/leg_timeline.png")
