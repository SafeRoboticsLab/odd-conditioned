"""E062 — does the TYPE of ODD-change dynamics matter? So far we only latched (step 1.0→0.2). Here we sweep
θ(t) SHAPES mid-episode and compare blind / history / conditioned (converged 50M), soft-rest scoring, 15 N pull.

Schedules (θ = FR torque, kept in the trained range [0.2,1.0]; change starts at t=2s):
  step     : 1.0 → 0.2 instantly (latch — the E060 baseline)
  ramp     : 1.0 → 0.2 linearly over 2 s (slow degradation — gives a reactive policy time to adapt)
  sine     : oscillate 1.0 ↔ 0.2, period 2 s (a FLICKERING / intermittent-power leg)
  recover  : 1.0 → 0.2 for 1.5 s → back to 1.0 (leg dies then comes back — can it stand up again?)

Question: is conditioning's edge robust across change shapes (it is told live θ), while blind/history depend on
how gradual/inferable the change is? Reports survival(t) traces + final survival & slam per (schedule, arm).
"""
import os, sys, io, contextlib, math
import torch as th
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "external/robot-safety-sandbox")
from robot_safety_sandbox import make_tensor, spec
from robot_safety_sandbox.eval.policies import load_twin
from robot_safety_sandbox.envs.go2_broken_leg.env_cfg import _ensure_fr_cache

DEV, N, DT = "cuda:0", 256, 0.02
STEPS, T0 = 250, 100        # change starts at t=2s
HI, LO = 1.0, 0.2
FR = 0.3                    # 15 N pull
PULL = th.tensor([0., 1., 0.])
CK = "results/go2_weak_leg_runs/{run}/checkpoints/model_49999872_steps.zip"
ARMS = [("blind", "go2_weak_leg_blind_soft", "go2_weak_leg_blind_soft_adv"),
        ("history", "go2_weak_leg_history_soft", "go2_weak_leg_history_soft_adv"),
        ("conditioned", "go2_weak_leg_conditioned_soft", "go2_weak_leg_conditioned_soft_adv")]
COL = {"blind": "#e67e22", "history": "#8e44ad", "conditioned": "#1a5276"}


def theta_of(sched, t):
    if t < T0:
        return HI
    s = t - T0
    if sched == "step":
        return LO
    if sched == "ramp":                                   # 1.0->0.2 over 2s (100 steps)
        return HI + (LO - HI) * min(1.0, s / 100.0)
    if sched == "sine":                                   # oscillate HI<->LO, period 2s
        return (HI + LO) / 2 + (HI - LO) / 2 * math.cos(2 * math.pi * s / 100.0)
    if sched == "recover":                                # LO for 1.5s then back to HI
        return LO if s < 75 else HI
    return LO


def run(task, run_dir, sched):
    with contextlib.redirect_stdout(io.StringIO()):
        env = make_tensor(task, N, DEV, adversary=True)
        model, norm = load_twin(CK.format(run=run_dir), DEV, quiet=True)
    inner = env.mj; _ensure_fr_cache(inner)
    ids, nominal = inner._fr_act_ids, inner._fr_nominal_forcerange
    env.force_scale = FR * th.ones(N, device=DEV)
    dstb = (PULL / PULL.norm()).to(DEV)[None].expand(N, spec(task).dstb_dim).contiguous()
    obs = env.reset(); alive = th.ones(N, dtype=th.bool, device=DEV); slam = th.zeros(N, dtype=th.bool, device=DEV)
    surv = []
    for t in range(STEPS):
        theta = theta_of(sched, t)
        inner.sim.model.actuator_forcerange[:, ids, :] = nominal * float(theta)
        inner._fr_torque_frac[:] = float(theta)
        with th.no_grad():
            a = th.clamp(model.policy._predict(norm(obs), deterministic=True), -1, 1)
        obs, g, dones, touts, l = env.step_tensor(th.cat([a, dstb], dim=1))
        ic = inner.termination_manager._term_dones.get("illegal_contact")
        nf = dones & ~touts & alive
        if ic is not None: slam |= nf & ic
        alive &= ~(dones & ~touts); surv.append(float(alive.float().mean()))
    env.close()
    return np.array(surv), float(alive.float().mean()), float(slam.float().mean())


if __name__ == "__main__":
    SCHED = ["step", "ramp", "sine", "recover"]
    res = {}
    print(f"{'schedule':9s} " + " ".join(f"{n:>22}" for n, _, _ in ARMS))
    for sched in SCHED:
        row = []
        for name, task, run_dir in ARMS:
            tr, fs, sl = run(task, run_dir, sched)
            res[(sched, name)] = tr
            row.append(f"surv={fs:.2f} slam={sl:.2f}")
        print(f"{sched:9s} " + " ".join(f"{c:>22}" for c in row))

    out = os.path.expanduser("~/artifacts/odd-conditioned/E062-odd-dynamics")
    os.makedirs(out, exist_ok=True)
    data = {"steps": STEPS, "t0": T0, "dt": DT, "sched": np.array(SCHED)}
    for s in SCHED:
        data[f"theta|{s}"] = np.array([theta_of(s, t) for t in range(STEPS)])
        for n, _, _ in ARMS:
            data[f"surv|{s}|{n}"] = res[(s, n)]
    np.savez(f"{out}/traces.npz", **data)
    print("saved", f"{out}/traces.npz  — run experiments/E062_plot.py to (re)draw the figure")
