"""E010 — does VANILLA ReachAvoidSAC already solve dynamic ODD implicitly? (bicycle, quick test)

Buzi's question: before building any conditioning/estimation machinery, just train the stock
algorithm on the target env with a VARYING ODD and measure whether it already handles it.

Setup: `BicycleGoalODD`, ODD = control-authority multiplier c ∈ [0.4, 1.0].
  - `blind`      : vanilla ReachAvoidSAC, c varies per-episode, NOT in the obs. THE TEST.
  - `blind_dyn`  : same, but c also re-rolls mid-episode (resample_prob>0) — true dynamic ODD.
  - `spec_hi`/`spec_lo` : specialists pinned at c=1.0 / c=0.4 — the per-ODD CEILING.
  - (optional) `oracle` : c appended to the obs — the conditioned upper bound.

Evaluate every policy across a c-sweep (fixed c per eval episode): REACH rate and COLLISION rate.
The question answers itself in the table:
  - blind reaches ~specialist at every c with ~0 collisions  => vanilla already solves it; the
    whole conditioning thesis is in trouble for this task. (Report honestly if so.)
  - blind is conservative (low reach at easy c) OR collides at low c (over-drives authority it
    lacks) => vanilla does NOT solve dynamic ODD; conditioning/estimation is motivated. Predicted.

Torch threads capped (E008c lesson: 24-thread default thrashes tiny nets).
"""
import argparse
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", os.environ.get("TORCH_THREADS", "4"))
import torch  # noqa: E402
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "4")))

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from odd_conditioned.envs.bicycle_odd import BicycleGoalODD  # noqa: E402

WANDB_PROJECT = "odd-conditioned"


def make_env(fixed_c, resample_prob, expose_odd, seed):
    from stable_baselines3.common.monitor import Monitor
    return Monitor(BicycleGoalODD(fixed_c=fixed_c, resample_prob=resample_prob,
                                  expose_odd=expose_odd, randomize=True, seed=seed))


def train(tag, fixed_c, resample_prob, expose_odd, steps, seed, out, device, use_wandb, git_hash):
    from stable_baselines3.common.vec_env import DummyVecEnv
    from safety_sb3 import ReachAvoidSAC

    run = None
    if use_wandb:
        import wandb
        from datetime import datetime
        run = wandb.init(project=WANDB_PROJECT, group=f"E010-{tag}",
                         name=f"{datetime.now():%Y%m%d-%H%M}_{tag}_s{seed}",
                         config=dict(tag=tag, fixed_c=fixed_c, resample_prob=resample_prob,
                                     expose_odd=expose_odd, steps=steps, seed=seed, git=git_hash,
                                     algo="ReachAvoidSAC (single-agent, v0.2.1)"),
                         sync_tensorboard=True, reinit=True, save_code=False)
    env = DummyVecEnv([lambda: make_env(fixed_c, resample_prob, expose_odd, seed)])
    model = ReachAvoidSAC("MlpPolicy", env, learning_rate=5e-4,
                          policy_kwargs=dict(net_arch=[128, 128, 128]), gamma=0.99,
                          verbose=0, seed=seed, device=device,
                          tensorboard_log=os.path.join(out, "tb") if use_wandb else None)
    model.learn(total_timesteps=steps, progress_bar=False)
    p = os.path.join(out, f"{tag}.zip")
    model.save(p)
    if run is not None:
        run.finish()
    return p


def evaluate(zip_path, expose_odd, c_grid, n_ep=60, seed=1000):
    """STRATEGY/EFFECTIVENESS metrics per fixed c — not just safe rate.

    Buzi's point: safe rate is not the differentiator (a blind policy learns one conservative mode
    that is safe across all c, like DR). The question is whether the learned STRATEGY *adapts* to c
    (exploits agility when c is high, backs off when low) or is c-INVARIANT. So we measure the
    policy's EFFECTIVENESS as a function of c, and read the SLOPE:
      * a c-invariant one-mode policy  -> metrics ~FLAT in c (esp. blind, which cannot see c);
      * an ODD-adaptive policy         -> metrics track c (faster / tighter / more exploitative
                                          when c is high).
    Metrics (over SUCCESSFUL reaches, plus the safe rates for context):
      reach_time   steps to goal            (adaptive: DOWN as c up — exploits agility)
      mean_speed   avg v on the path        (adaptive: UP as c up)
      clearance    min g_x along traj       (adaptive: cuts closer at high c, keeps margin at low c)
      cmd_accel/omega  mean |commanded u|   (the policy's chosen aggressiveness, BEFORE c scaling —
                                             a c-invariant policy commands the same u regardless)
    """
    from safety_sb3 import ReachAvoidSAC
    model = ReachAvoidSAC.load(zip_path, device="cpu")
    out = {}
    for c in c_grid:
        reached = collided = 0
        rt, spd, clr, ca, co = [], [], [], [], []
        for e in range(n_ep):
            env = BicycleGoalODD(fixed_c=c, resample_prob=0.0, expose_odd=expose_odd,
                                 randomize=True, seed=seed + e)
            obs, _ = env.reset(seed=seed + e)
            done = got = hit = False
            t = 0; vs = []; gmin = np.inf; accs = []; oms = []
            while not done:
                a, _ = model.predict(obs, deterministic=True)
                a = np.asarray(a).reshape(-1)
                accs.append(abs(float(a[0]))); oms.append(abs(float(a[1])))
                obs, g, term, trunc, info = env.step(a)
                vs.append(float(env.s[2]))            # speed state
                gmin = min(gmin, g)
                if info.get("collided"): hit = True
                if info.get("reached"): got = True
                t += 1; done = term or trunc
            collided += hit
            if got and not hit:                       # strategy metrics on CLEAN reaches only
                reached += 1
                rt.append(t); spd.append(np.mean(vs)); clr.append(gmin)
                ca.append(np.mean(accs)); co.append(np.mean(oms))
        m = lambda a: float(np.mean(a)) if a else float("nan")
        out[f"{c:.2f}"] = dict(reach=reached / n_ep, collide=collided / n_ep,
                               reach_time=m(rt), mean_speed=m(spd), clearance=m(clr),
                               cmd_accel=m(ca), cmd_omega=m(co))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=150_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/E010")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--arms", nargs="+",
                    default=["blind", "blind_dyn", "spec_hi", "spec_lo", "oracle"])
    ap.add_argument("--arm", help="train exactly ONE arm and exit (parallel fan-out)")
    ap.add_argument("--score-only", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import subprocess
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
            git_hash += "-DIRTY"
    except Exception:
        git_hash = "?"
    print(f"E010  code {git_hash}  steps {args.steps}  threads {torch.get_num_threads()}")

    # (tag, fixed_c, resample_prob, expose_odd)
    specs = {
        "blind":     (None, 0.0,   False),   # varying c per-episode, not observed  <- THE TEST
        "blind_dyn": (None, 0.004, False),   # + mid-episode ODD change (dynamic)
        "spec_hi":   (1.0,  0.0,   False),   # ceiling at easy c
        "spec_lo":   (0.4,  0.0,   False),   # ceiling at hard c
        "oracle":    (None, 0.0,   True),    # c in the obs (conditioned upper bound)
    }
    c_grid = [0.5, 0.75, 1.0, 1.25, 1.5]

    # single-arm mode: one training, then exit (the driver fans 5 of these out in parallel)
    if args.arm:
        fc, rp, eo = specs[args.arm]
        print(f"  train {args.arm} (fixed_c={fc}, resample={rp}, oracle={eo}) ...", flush=True)
        train(args.arm, fc, rp, eo, args.steps, args.seed, args.out, args.device,
              not args.no_wandb, git_hash)
        return

    zips = {}
    for tag in args.arms:
        fc, rp, eo = specs[tag]
        p = os.path.join(args.out, f"{tag}.zip")
        if not (args.score_only and os.path.exists(p)):
            print(f"  train {tag} (fixed_c={fc}, resample={rp}, oracle={eo}) ...", flush=True)
            p = train(tag, fc, rp, eo, args.steps, args.seed, args.out, args.device,
                      not args.no_wandb, git_hash)
        zips[tag] = (p, eo)

    results = {}
    for tag, (p, eo) in zips.items():
        results[tag] = evaluate(p, eo, c_grid)
    json.dump(results, open(os.path.join(args.out, "eval.json"), "w"), indent=2)

    # --- the STRATEGY question: does effectiveness track c, or is it flat (one conservative mode)?
    def slope(arm, key):
        """least-squares slope of metric vs c over the grid — the adaptation signal."""
        r = results.get(arm, {})
        xs = [c for c in c_grid if f"{c:.2f}" in r and not np.isnan(r[f"{c:.2f}"][key])]
        ys = [r[f"{c:.2f}"][key] for c in xs]
        if len(xs) < 2:
            return float("nan")
        return float(np.polyfit(xs, ys, 1)[0])

    metrics = ["reach", "collide", "reach_time", "mean_speed", "clearance", "cmd_accel", "cmd_omega"]
    for key in metrics:
        print(f"\n{key} @ c = " + " ".join(f"{c:.2f}" for c in c_grid) + "   [slope vs c]")
        for tag in results:
            row = results[tag]
            cells = " ".join(f"{row[f'{c:.2f}'][key]:5.2f}" for c in c_grid)
            print(f"  {tag:>10} | {cells}   [{slope(tag, key):+.2f}]")

    print("\n" + "=" * 74)
    print("STRATEGY VERDICT — the question is the SLOPE of effectiveness vs c, not the safe rate.")
    print("A c-INVARIANT (one conservative mode) policy has ~flat metrics; an ODD-ADAPTIVE policy")
    print("tracks c (faster / higher-speed / tighter clearance as c rises).")
    for tag in ("blind", "oracle", "spec_hi"):
        if tag in results:
            print(f"  {tag:>10}: speed slope {slope(tag,'mean_speed'):+.2f}, "
                  f"reach_time slope {slope(tag,'reach_time'):+.2f}, "
                  f"clearance slope {slope(tag,'clearance'):+.2f}, "
                  f"cmd_accel slope {slope(tag,'cmd_accel'):+.2f}")
    print("  Read: blind flat + oracle sloped => conditioning ENABLES strategy adaptation (the point).")
    print("        blind also sloped         => the realized behaviour adapts via DYNAMICS alone,")
    print("                                     even though the POLICY map is c-invariant — check")
    print("                                     cmd_accel/omega slopes (the policy's OWN choice).")
    print("=" * 74)


if __name__ == "__main__":
    main()
