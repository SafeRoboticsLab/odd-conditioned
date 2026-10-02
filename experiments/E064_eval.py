"""E064 eval — the fair fight, aggregated over seeds. Dynamic-trained blind/conditioned/history (+ optionally the
static-trained arms, + the RMA-student baseline written by E064_rma.py) evaluated on the mid-episode leg-death
ramp (soft-rest), at 15N & 25N. Reports survival + slam + HEDGING COST (pre-fault standing quality), mean±std
over seeds. Answers threats (a) does dynamic-blind close the gap, (c) seed spread.

Usage: python experiments/E064_eval.py            # dynamic-trained arms, all seeds present
"""
import os, sys, io, contextlib, glob, json
from _paths import _ART
import torch as th
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "external/robot-safety-sandbox")
import numpy as np
from robot_safety_sandbox import make_tensor, spec
from robot_safety_sandbox.eval.policies import load_twin
from robot_safety_sandbox.envs.go2_broken_leg.env_cfg import _ensure_fr_cache

DEV, N, DT = "cuda:0", 256, 0.02
STEPS, T_SWITCH, HI, LO = 250, 100, 1.0, 0.2
PULL = th.tensor([0., 1., 0.])
SEEDS = [0, 1, 2]
FORCES = [0.3, 0.5]                                  # 15N, 25N
# arm -> (eval task giving its obs surface, run-dir template with {s}=seed)
ARMS = {
    "blind":       ("go2_weak_leg_blind_soft",       "E064_s{s}/go2_weak_leg_blind_soft_dyn_adv"),
    "conditioned": ("go2_weak_leg_conditioned_soft", "E064_s{s}/go2_weak_leg_conditioned_soft_dyn_adv"),
    "history":     ("go2_weak_leg_history_soft",     "E064_s{s}/go2_weak_leg_history_soft_dyn_adv"),
}
CKROOT = "results/go2_weak_leg_runs"


def ckpt(rundir_tmpl, s):
    p = f"{CKROOT}/{rundir_tmpl.format(s=s)}/checkpoints/model_49999872_steps.zip"
    return p if os.path.exists(p) else None


def rollout(task, ck, fr):
    with contextlib.redirect_stdout(io.StringIO()):
        env = make_tensor(task, N, DEV, adversary=True)
        model, norm = load_twin(ck, DEV, quiet=True)
    inner = env.mj; _ensure_fr_cache(inner)
    ids, nominal = inner._fr_act_ids, inner._fr_nominal_forcerange
    env.force_scale = fr * th.ones(N, device=DEV)
    dstb = (PULL / PULL.norm()).to(DEV)[None].expand(N, spec(task).dstb_dim).contiguous()
    obs = env.reset(); alive = th.ones(N, dtype=th.bool, device=DEV); slam = th.zeros(N, dtype=th.bool, device=DEV)
    pre_h, pre_ontarget = [], []
    for t in range(STEPS):
        theta = HI if t < T_SWITCH else LO
        inner.sim.model.actuator_forcerange[:, ids, :] = nominal * float(theta)
        inner._fr_torque_frac[:] = float(theta)
        with th.no_grad():
            a = th.clamp(model.policy._predict(norm(obs), deterministic=True), -1, 1)
        obs, g, dones, touts, l = env.step_tensor(th.cat([a, dstb], dim=1))
        ic = inner.termination_manager._term_dones.get("illegal_contact")
        nf = dones & ~touts & alive
        if ic is not None: slam |= nf & ic
        alive &= ~(dones & ~touts)
        if t < T_SWITCH:                                    # HEDGING COST: quality while the leg is still healthy
            d = inner.scene["robot"].data
            pre_h.append(d.root_link_pos_w[alive, 2].mean().item() if bool(alive.any()) else float('nan'))
            pre_ontarget.append(float((l[alive] > 0).float().mean()) if bool(alive.any()) else float('nan'))
    env.close()
    return {"surv": float(alive.float().mean()), "slam": float(slam.float().mean()),
            "pre_h": float(np.nanmean(pre_h)), "pre_ontarget": float(np.nanmean(pre_ontarget))}


if __name__ == "__main__":
    results = {}   # (arm, force) -> list over seeds of dicts
    for arm, (task, tmpl) in ARMS.items():
        for fr in FORCES:
            vals = []
            for s in SEEDS:
                ck = ckpt(tmpl, s)
                if ck is None:
                    continue
                vals.append(rollout(task, ck, fr))
            results[(arm, fr)] = vals
            if vals:
                surv = np.array([v["surv"] for v in vals]); slam = np.array([v["slam"] for v in vals])
                heh = np.array([v["pre_h"] for v in vals])
                print(f"{arm:12s} {int(fr*50)}N  n={len(vals)}  surv={surv.mean():.2f}±{surv.std():.2f}  "
                      f"slam={slam.mean():.2f}  pre_height={heh.mean():.3f}m")
    out = os.path.expanduser(_ART + "/E064-fair-fight")
    os.makedirs(out, exist_ok=True)
    json.dump({f"{a}|{f}": v for (a, f), v in results.items()}, open(f"{out}/results.json", "w"), indent=2)
    print(f"\nsaved -> {out}/results.json  (HEDGING: lower pre_height = more pre-emptive crouch = hidden cost)")
