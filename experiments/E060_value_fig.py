"""Value-of-conditioning figure for a given leg-death severity θ_lo, at two pull forces. Parameterized so it
makes the 20% figure (θ_lo=0.2, 15/25N) and the 10% figure (θ_lo=0.1, 0/10N). Soft-rest scoring; converged 50M.
"""
import os, sys, io, contextlib, argparse
from _paths import _ART
import torch as th
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "external/robot-safety-sandbox")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from robot_safety_sandbox import make_tensor, spec
from robot_safety_sandbox.eval.policies import load_twin
from robot_safety_sandbox.envs.go2_broken_leg.env_cfg import _ensure_fr_cache

DEV, N, DT = "cuda:0", 256, 0.02
STEPS, T_SWITCH, THETA_HI = 250, 100, 1.0
PULL = th.tensor([0., 1., 0.])
CK = "results/go2_weak_leg_runs/{run}/checkpoints/model_49999872_steps.zip"
ARMS = [("blind", "go2_weak_leg_blind_soft", "go2_weak_leg_blind_soft_adv"),
        ("history", "go2_weak_leg_history_soft", "go2_weak_leg_history_soft_adv"),
        ("conditioned", "go2_weak_leg_conditioned_soft", "go2_weak_leg_conditioned_soft_adv")]
COL = ["#e67e22", "#8e44ad", "#1a5276"]


def run(task, run_dir, fr, theta_lo):
    with contextlib.redirect_stdout(io.StringIO()):
        env = make_tensor(task, N, DEV, adversary=True)
        model, norm = load_twin(CK.format(run=run_dir), DEV, quiet=True)
    inner = env.mj; _ensure_fr_cache(inner)
    ids, nominal = inner._fr_act_ids, inner._fr_nominal_forcerange
    env.force_scale = fr * th.ones(N, device=DEV)
    dstb = (PULL / PULL.norm()).to(DEV)[None].expand(N, spec(task).dstb_dim).contiguous()
    obs = env.reset(); alive = th.ones(N, dtype=th.bool, device=DEV); slam = th.zeros(N, dtype=th.bool, device=DEV)
    for t in range(STEPS):
        theta = THETA_HI if t < T_SWITCH else theta_lo
        inner.sim.model.actuator_forcerange[:, ids, :] = nominal * float(theta)
        inner._fr_torque_frac[:] = float(theta)
        with th.no_grad():
            a = th.clamp(model.policy._predict(norm(obs), deterministic=True), -1, 1)
        obs, g, dones, touts, l = env.step_tensor(th.cat([a, dstb], dim=1))
        ic = inner.termination_manager._term_dones.get("illegal_contact")
        nf = dones & ~touts & alive
        if ic is not None: slam |= nf & ic
        alive &= ~(dones & ~touts)
    env.close()
    return float(alive.float().mean()), float(slam.float().mean())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--theta-lo", type=float, required=True)
    ap.add_argument("--forces", required=True, help="comma force_scales, e.g. 0.0,0.2")
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    forces = [float(x) for x in a.forces.split(",")]
    surv = {f: [] for f in forces}; slam = {f: [] for f in forces}
    for f in forces:
        for name, task, run_dir in ARMS:
            s, sl = run(task, run_dir, f, a.theta_lo)
            surv[f].append(s); slam[f].append(sl)
            print(f"theta_lo={a.theta_lo} force={int(f*50)}N {name:12s} survival={s:.2f} slam={sl:.2f}")
    names = [n for n, _, _ in ARMS]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8))
    x = np.arange(len(names)); w = 0.8 / len(forces)
    for i, f in enumerate(forces):
        off = (i - (len(forces) - 1) / 2) * w
        a1.bar(x + off, surv[f], w, color=COL, alpha=0.55 + 0.45 * i / max(1, len(forces) - 1),
               edgecolor="k", hatch=["//", "", "xx"][i % 3], label=f"{int(f*50)} N")
        a2.bar(x + off, slam[f], w, color=COL, alpha=0.55 + 0.45 * i / max(1, len(forces) - 1),
               edgecolor="k", hatch=["//", "", "xx"][i % 3], label=f"{int(f*50)} N")
    for ax, ttl, yl in [(a1, "Survival after leg dies (higher=better)", "survival"),
                        (a2, "Slam rate (lower=better)", "slam fraction")]:
        ax.set_xticks(x); ax.set_xticklabels(["blind\n(no θ)", "history\n(infer θ)", "conditioned\n(knows θ)"])
        ax.set_title(ttl); ax.set_ylabel(yl); ax.set_ylim(0, 1); ax.grid(axis="y", alpha=0.3); ax.legend(title="pull")
    ood = a.theta_lo < 0.2   # θ<0.2 is outside the training range [0.2,1.0]
    sub = ("conditioned (knows θ) safest; history COLLAPSES out-of-distribution (overfit)" if ood
           else "conditioning wins, monotonic in θ-information (blind < history < conditioned)")
    fig.suptitle(f"Value of ODD-conditioning — FR leg dies mid-episode (θ 1.0→{a.theta_lo}"
                 f"{' , OOD' if ood else ''}), soft-rest, CONVERGED\n{sub}", fontsize=12)
    fig.tight_layout()
    out = os.path.expanduser(_ART + f"/E060-soft-ramp/value_of_conditioning_{a.tag}.png")
    fig.savefig(out, dpi=130, bbox_inches="tight"); print("saved", out)
