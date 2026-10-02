"""E060 — MID-TRAINING soft-rest dynamic ramp: when the FR leg dies mid-episode, does the robot GRACEFULLY SIT
(low, level, settled, gentle contact) or FIGHT the pull and SLAM/tip? Quick read on the latest checkpoints to
decide whether to let E056-E059 run to convergence.

Soft-rest scoring: SAFE = no slam (non-foot contact < 80 N); TARGET = level + settled. Failure = slam
(illegal_contact >80N) OR tip (fell_over >70°). We degrade θ (FR torque) 1.0→0.2 at t=2s (per-step forcerange
write, live θ for the conditioned obs) under a constant pull, and report per policy:
  * survival(end)     — fraction still safe (didn't slam or tip) after the leg died
  * Δbase-height      — did it LOWER itself (sit) after the switch? (sit = graceful; stay-high = fighting)
  * fail mode         — of the failures, slam vs tip
"""
import os, sys, io, contextlib
import torch as th
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "external/robot-safety-sandbox")
from robot_safety_sandbox import make_tensor, spec
from robot_safety_sandbox.eval.policies import load_twin
from robot_safety_sandbox.envs.go2_broken_leg.env_cfg import _ensure_fr_cache

DEV, N, DT = "cuda:0", 256, 0.02
STEPS, T_SWITCH, THETA_HI, THETA_LO = 250, 100, 1.0, 0.2
PULL = th.tensor([0., 1., 0.])
# policy -> (soft eval task = its obs surface + soft margins + FR cache, run dir, latest ckpt steps)
POLICIES = {
    "20-spec":     ("go2_weak_leg_20_soft",          "go2_weak_leg_20_soft_adv",          49999872),
    "blind":       ("go2_weak_leg_blind_soft",       "go2_weak_leg_blind_soft_adv",       24999936),
    "conditioned": ("go2_weak_leg_conditioned_soft", "go2_weak_leg_conditioned_soft_adv", 24999936),
    "history":     ("go2_weak_leg_history_soft",     "go2_weak_leg_history_soft_adv",     24999936),
}


def run(task, ck, fr):
    with contextlib.redirect_stdout(io.StringIO()):
        env = make_tensor(task, N, DEV, adversary=True)
        model, norm = load_twin(ck, DEV, quiet=True)
    inner = env.mj
    _ensure_fr_cache(inner)
    ids, nominal = inner._fr_act_ids, inner._fr_nominal_forcerange
    dd = spec(task).dstb_dim
    env.force_scale = fr * th.ones(N, device=DEV)
    dstb = (PULL / PULL.norm()).to(DEV)[None].expand(N, dd).contiguous()
    obs = env.reset()

    def set_theta(theta):
        inner.sim.model.actuator_forcerange[:, ids, :] = nominal * float(theta)
        inner._fr_torque_frac[:] = float(theta)
    set_theta(THETA_HI)
    alive = th.ones(N, dtype=th.bool, device=DEV)
    slam = th.zeros(N, dtype=th.bool, device=DEV)
    tip = th.zeros(N, dtype=th.bool, device=DEV)
    h_pre = None
    for t in range(STEPS):
        set_theta(THETA_HI if t < T_SWITCH else THETA_LO)
        with th.no_grad():
            a = th.clamp(model.policy._predict(norm(obs), deterministic=True), -1, 1)
        obs, g, dones, touts, l = env.step_tensor(th.cat([a, dstb], dim=1))
        tm = inner.termination_manager
        fo = tm._term_dones.get("fell_over"); ic = tm._term_dones.get("illegal_contact")
        newfail = dones & ~touts & alive
        if fo is not None: tip |= newfail & fo
        if ic is not None: slam |= newfail & ic
        alive &= ~(dones & ~touts)
        if t == T_SWITCH - 1:
            h_pre = inner.scene["robot"].data.root_link_pos_w[:, 2].clone()
    d = inner.scene["robot"].data
    h_end = d.root_link_pos_w[:, 2]
    surv = alive
    dh = (h_end[surv] - h_pre[surv]).mean().item() if bool(surv.any()) else float("nan")
    h_now = h_end[surv].mean().item() if bool(surv.any()) else float("nan")
    env.close()
    return {"surv": float(surv.float().mean()), "dh": dh, "h": h_now,
            "slam": float(slam.float().mean()), "tip": float(tip.float().mean())}


if __name__ == "__main__":
    for FR in (0.3, 0.5):
        print(f"\n===== pull {int(FR*50)} N | θ {THETA_HI}->{THETA_LO} @ t={T_SWITCH*DT:.0f}s =====")
        print(f"{'policy':13s} {'survival':>9} {'Δheight(m)':>11} {'endH(m)':>8} {'slam':>6} {'tip':>6}")
        for name, (task, rundir, steps) in POLICIES.items():
            import glob
            cks = sorted(glob.glob(f"results/go2_weak_leg_runs/{rundir}/checkpoints/model_*_steps.zip"),
                         key=lambda x: int(x.split("model_")[1].split("_steps")[0]))
            ck = cks[-1]
            print(f"  [{name}: {ck.split('/')[-1]}]")
            r = run(task, ck, FR)
            print(f"{name:13s} {r['surv']:9.2f} {r['dh']:+11.3f} {r['h']:8.3f} {r['slam']:6.2f} {r['tip']:6.2f}")
    print("\nΔheight < 0 = LOWERED itself (sat) after leg death = graceful; ≈0 = kept fighting to stand.")
