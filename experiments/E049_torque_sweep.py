"""E049 — SURVIVAL SWEEP / SWAP-MATRIX for the leg-degradation specialists (does the strategy BIFURCATE?).

Three fixed-θ specialists were trained under the two-player game (ReachAvoidPPO2P), θ = FR-leg allowable
torque fraction:
    normal  = go2_stabilize        (torque_frac 1.0)   results/go2_weak_leg_runs/go2_stabilize_adv
    weak50  = go2_weak_leg          (torque_frac 0.5)   results/go2_weak_leg_runs/go2_weak_leg_adv
    weak20  = go2_weak_leg_20       (torque_frac 0.2)   results/go2_weak_leg_runs/go2_weak_leg_20_adv

This evaluates EACH policy across a GRID of FR-torque levels (the swap matrix: policy × test-θ) under a
FIXED SCRIPTED lateral pull (the payload-phase eval convention: unit pull [0,1,0] at force_scale × force_max,
NOT the learned adversary). Metric per cell, reset-independent (E037 convention):
    * falls/env-sec         — survival (lower = safer)
    * frac-fell             — did it break at all
    * end y-drift (m)       — STRATEGY readout: how far the base slid along the pull (dodge/give vs brace/hold)

BIFURCATION read: if the strategy is genuinely θ-specialized, each policy survives best near its OWN training
θ and degrades off it — e.g. the NORMAL policy (relies on the FR leg) should FALL at low test-θ where that leg
is weak, while weak20 holds; and the drift pattern should differ (a policy that unloads the FR corner slides
differently than one that pushes off it). A FLAT matrix + identical drift = one strategy covers all θ = NO
bifurcation (the payload story), i.e. sound PPO absorbed the ODD.
"""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
import torch as th
sys.path.insert(0, "external/robot-safety-sandbox")
from robot_safety_sandbox import make_tensor, spec
from robot_safety_sandbox.eval.policies import load_twin

DEV = "cuda:0"; PULL = th.tensor([0., 1., 0.]); DT = 0.02; STEPS = 250; NENV = 128
CK = "results/go2_weak_leg_runs/{run}/checkpoints/model_49999872_steps.zip"
POLICIES = {                          # display name -> (run dir, its training θ)
    "normal(1.0)": ("go2_stabilize_adv", 1.0),
    "weak50(0.5)": ("go2_weak_leg_adv", 0.5),
    "weak20(0.2)": ("go2_weak_leg_20_adv", 0.2),
}
TEST_PCT = [100, 70, 50, 30, 20, 10]                  # FR-torque test grid (columns)
FORCE_SCALES = [0.3, 0.5, 0.7]                        # x force_max(50N) -> 15 / 25 / 35 N pull


def rollout(model, norm, task, fr):
    env = make_tensor(task, NENV, DEV, adversary=True)
    env.force_scale = fr * th.ones(NENV, device=DEV)
    dd = spec(task).dstb_dim
    dstb = (PULL / PULL.norm()).to(DEV)[None].expand(NENV, dd).contiguous()
    obs = env.reset()
    d = env.mj.scene["robot"].data
    y0 = d.root_link_pos_w[:, 1].clone()
    nfall = 0; t_first = th.full((NENV,), -1.0, device=DEV)
    for t in range(STEPS):
        with th.no_grad():
            a = th.clamp(model.policy._predict(norm(obs), deterministic=True), -1, 1)
        obs, g, dones, touts, l = env.step_tensor(th.cat([a, dstb], dim=1))
        fail = dones & (~touts)
        nfall += int(fail.sum())
        fresh = fail & (t_first < 0)
        t_first = th.where(fresh, th.full_like(t_first, t * DT), t_first)
    dy = env.mj.scene["robot"].data.root_link_pos_w[:, 1] - y0
    fell = t_first >= 0
    surv = ~fell
    drift = float(dy[surv].mean()) if bool(surv.any()) else float("nan")   # survivors only
    env.close()
    return nfall / (NENV * STEPS * DT), float(fell.float().mean()), drift


if __name__ == "__main__":
    # load the three twins once
    twins = {}
    for name, (run, _th) in POLICIES.items():
        twins[name] = load_twin(CK.format(run=run), DEV, quiet=True)
        print(f"loaded {name:12s} <- {run} ({type(twins[name][0]).__name__})")
    results = {}   # (force, policy, pct) -> (fps, frac, drift)
    for fr in FORCE_SCALES:
        print(f"\n===== scripted pull {fr*50:.0f} N (force_scale {fr}) =====")
        hdr = f"{'test-θ':>7} | " + " | ".join(f"{n:^24}" for n in POLICIES)
        print(hdr); print("-" * len(hdr))
        print(f"{'':>7} | " + " | ".join(f"{'falls/s  frac  drift(m)':^24}" for _ in POLICIES))
        for pct in TEST_PCT:
            task = "go2_stabilize" if pct == 100 else f"go2_weak_leg_sweep_{pct}"
            cells = []
            for name in POLICIES:
                model, norm = twins[name]
                try:
                    fps, frac, drift = rollout(model, norm, task, fr)
                    results[(fr, name, pct)] = (fps, frac, drift)
                    cells.append(f"{fps:6.2f}  {frac:4.2f}  {drift:+6.3f}")
                except Exception as e:
                    cells.append(f"ERR {type(e).__name__}")
            marker = ""  # mark each policy's own training θ
            print(f"{pct:>6}% | " + " | ".join(f"{c:^24}" for c in cells))
    # save raw results for plotting
    import json
    out = os.path.expanduser("~/artifacts/odd-conditioned/E049-torque-sweep")
    os.makedirs(out, exist_ok=True)
    with open(f"{out}/results.json", "w") as f:
        json.dump({f"{k[0]}|{k[1]}|{k[2]}": v for k, v in results.items()}, f, indent=2)
    print(f"\nsaved -> {out}/results.json")
    print("\nBIFURCATION? Read DOWN each column: does the policy trained at that θ survive best there and worse")
    print("off-θ (diagonal structure)? And does end-drift DIFFER across policies at the same test-θ (distinct")
    print("strategy)? Flat survival + identical drift = one strategy absorbs all θ = NO bifurcation.")
