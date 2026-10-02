"""E061 — VIDEO of the E060 headline: FR leg dies mid-episode (θ 1.0→0.2 @ t=2s) under a pull, soft-rest objective.
Side-by-side robot grids for BLIND | HISTORY | CONDITIONED (converged 50M), with a synced graph below plotting
θ(t) and the cumulative fall fraction for each arm. Story: after the leg dies, BLIND (must infer) climbs in falls
while CONDITIONED (told θ) stays up — the value-of-information ordering, made visible.
"""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, torch as th, imageio.v2 as imageio
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    from PIL import Image, ImageDraw
    HAVE_PIL = True
except Exception:
    HAVE_PIL = False
sys.path.insert(0, "external/robot-safety-sandbox")
from robot_safety_sandbox import make_tensor, spec
from robot_safety_sandbox.eval.policies import load_twin
from robot_safety_sandbox.envs.go2_broken_leg.env_cfg import _ensure_fr_cache

DEV = "cuda:0"; PULL = th.tensor([0., 1., 0.])
STEPS, T_SWITCH, DT = 200, 100, 0.02
THETA_HI = 1.0
FR_BODIES = (6, 7, 8)                       # FR_hip / FR_thigh / FR_calf — colored RED once the leg is degraded
RED = [0.9, 0.08, 0.08, 1.0]
CK = "results/go2_weak_leg_runs/{run}/checkpoints/model_49999872_steps.zip"
ARMS = {"BLIND": ("go2_weak_leg_blind_soft", "go2_weak_leg_blind_soft_adv"),
        "HISTORY": ("go2_weak_leg_history_soft", "go2_weak_leg_history_soft_adv"),
        "CONDITIONED": ("go2_weak_leg_conditioned_soft", "go2_weak_leg_conditioned_soft_adv")}
COL = {"BLIND": "#e67e22", "HISTORY": "#8e44ad", "CONDITIONED": "#1a5276"}


def rollout(task, run, nenv, render, theta_lo, fr):
    env = make_tensor(task, nenv, DEV, adversary=True, **({"render_mode": "rgb_array"} if render else {}))
    if render:
        try: env.mj.cfg.viewer.max_extra_envs = max(1, nenv - 1)
        except Exception: pass
    model, norm = load_twin(CK.format(run=run), DEV, quiet=True)
    inner = env.mj; _ensure_fr_cache(inner)
    ids, nominal = inner._fr_act_ids, inner._fr_nominal_forcerange
    m = inner.sim.mj_model
    fr_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] in FR_BODIES]  # FR-leg geoms (mesh+collision)
    orig_rgba = inner.sim.model.geom_rgba[:, fr_geoms, :].clone()
    env.force_scale = fr * th.ones(nenv, device=DEV)
    dstb = (PULL / PULL.norm()).to(DEV)[None].expand(nenv, spec(task).dstb_dim).contiguous()
    obs = env.reset(); frames = []; ever = th.zeros(nenv, dtype=th.bool, device=DEV); frac = []
    reddened = False
    for t in range(STEPS):
        theta = THETA_HI if t < T_SWITCH else theta_lo
        inner.sim.model.actuator_forcerange[:, ids, :] = nominal * float(theta)
        inner._fr_torque_frac[:] = float(theta)
        if render and theta < 1.0 and not reddened:                        # paint the FR leg RED when it degrades
            inner.sim.model.geom_rgba[:, fr_geoms, :] = th.tensor(RED, device=DEV)
            for g in fr_geoms: m.geom_rgba[g] = RED
            reddened = True
        with th.no_grad():
            a = th.clamp(model.policy._predict(norm(obs), deterministic=True), -1, 1)
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


def graph_frame(fracs, i, W, H, theta_lo):
    t = np.arange(STEPS) * DT
    theta = np.array([THETA_HI if s < T_SWITCH else theta_lo for s in range(STEPS)])
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = fig.add_subplot(111); ax2 = ax.twinx()
    ax.plot(t, theta, color="#333", lw=1.0, alpha=0.3); ax.plot(t[:i + 1], theta[:i + 1], color="#333", lw=2.2)
    for name, f in fracs.items():
        ax2.plot(t, f, color=COL[name], lw=1.0, alpha=0.25)
        ax2.plot(t[:i + 1], f[:i + 1], color=COL[name], lw=2.4, label=f"{name} fell")
    ax.axvline(T_SWITCH * DT, color="r", ls="--", lw=1.2, alpha=0.6)
    ax.axvline(t[i], color="#555", lw=1.6)
    ax.set_xlim(0, t[-1]); ax.set_ylim(0, 1.1); ax2.set_ylim(-0.02, 1.02)
    ax.set_xlabel("time (s)"); ax.set_ylabel(f"FR leg torque θ (1.0 → {theta_lo})", color="#333")
    ax2.set_ylabel("cumulative fall fraction")
    ax.text(T_SWITCH * DT, 1.06, " LEG DIES (red)", color="r", fontsize=9, va="top")
    ax2.legend(loc="center left", fontsize=8, framealpha=0.9)
    fig.tight_layout(pad=0.5); fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy(); plt.close(fig)
    return buf


def make_video(theta_lo, fr, label):
    NV = 16
    grids, fracs = {}, {}
    for arm, (task, run) in ARMS.items():
        grids[arm], _ = rollout(task, run, NV, True, theta_lo, fr)
        _, fracs[arm] = rollout(task, run, 128, False, theta_lo, fr)
        print(f"  {label} {arm}: fell {fracs[arm][-1]:.2f}")
    H, W = grids["BLIND"][0].shape[:2]
    order = ["BLIND", "HISTORY", "CONDITIONED"]
    top = [np.hstack([tag(grids[a][i][:H, :W], f"{a}  ({fracs[a][-1]*100:.0f}% fell)") for a in order])
           for i in range(STEPS)]
    TW = top[0].shape[1]; GH = 320
    comb = [np.vstack([top[i], graph_frame(fracs, i, TW, GH, theta_lo)[:GH, :TW]]) for i in range(STEPS)]
    os.makedirs(os.path.expanduser("~/artifacts/odd-conditioned/E060-soft-ramp"), exist_ok=True)
    out = os.path.expanduser(f"~/artifacts/odd-conditioned/E060-soft-ramp/{label}.mp4")
    imageio.mimsave(out, comb, fps=30, macro_block_size=1)
    print(f"  wrote {out} ({len(comb)} frames, {comb[0].shape[1]}x{comb[0].shape[0]})")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="run a single config by label")
    args = ap.parse_args()
    # (theta_lo, force_scale, label). force_scale x 50N = pull.
    CONFIGS = [
        (0.2, 0.3, "legdeath_20pct_15N_red"),
        (0.2, 0.5, "legdeath_20pct_25N_red"),
        (0.0, 0.0, "legdeath_DEAD_0N_red"),
        (0.0, 0.2, "legdeath_DEAD_10N_red"),
        (0.1, 0.0, "legdeath_10pct_0N_red"),
        (0.1, 0.2, "legdeath_10pct_10N_red"),
    ]
    for theta_lo, fr, label in CONFIGS:
        if args.only and args.only not in label:                 # substring match (e.g. --only 10pct)
            continue
        print(f"=== {label}: θ 1.0->{theta_lo}, pull {int(fr*50)}N ===")
        make_video(theta_lo, fr, label)
