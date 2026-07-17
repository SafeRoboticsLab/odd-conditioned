"""E008c — THE GATE, attempt 2. Fixes the six confounds E008b diagnosed.

The question (professor, gate criterion): can ONE conditioned critic recover E003's family?
  FIRE (pivot to pathology paper): specialists recover their slices while conditioned stays
    statistically indistinguishable from blind across seeds.
  PASS (capability paper alive): conditioned tracks the specialists.
  BROKEN RECIPE (debug, don't pivot): specialists ALSO fail — contradicts ISAACS's 5-D validation.

Fixes vs E008 (each closes a confound that biased E008 toward optimism / conditioned≈blind):
  (C) resample_prob=0 — STATIC per-episode ODD, so the static E003 slice IS the correct fixed
      point. E008 resampled mid-episode ⇒ its true fixed point was a jump-ODD value; scoring it
      against static slices was the V-vs-frozen-slices error in reverse. (Mid-episode resampling
      belongs to E010, scored against E009's V*.)
  #1  NORMALIZED ODD feature (env.normalize_mass: [2,8]→[-1,1]) — E008 fed raw m~5 alongside
      sin/cos∈[-1,1]; a near-constant large input a net learns to ignore.
  A   WIDE spawns (spawn_omega_frac 0.5→0.98) — E008 supervised only half the scoring grid's ω
      range, so the high-|ω| boundary was pure extrapolation.
  γ   ANNEAL γ→0.999 — the discounted value recovers the true (maximal) set only as γ→1
      (Fisac 2019). Prerequisite for the maximality CLAIM. Direction is →1, not →0.
  ctrl SPECIALIST arm — the control that separates 'conditioning is costly' from 'the problem is
      hard for this RL setup at all'. Without it E008 could not attribute its own failure.
  scoring min(g,V̂) COMPOSITION + region decomposition (E008b) — score the deployed filter, not
      raw Q, and only count in-support-boundary optimism.
"""
import argparse
import json
import os
import sys

# CRITICAL: cap torch threads BEFORE importing torch. The env is a 2-D ODE (free); 100% of the
# cost is SAC gradient steps on tiny [128,128,128] nets. On a 24-core box torch's default
# all-cores threading THRASHES: measured 3 steps/s at 24 threads vs 102 steps/s at 4 (34x).
# 4 threads/job × 5 jobs = 20 ≤ 24 cores, so all arms run in parallel in ~50 min, not ~5 days.
os.environ.setdefault("OMP_NUM_THREADS", os.environ.get("TORCH_THREADS", "4"))
import torch  # noqa: E402
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "4")))

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from odd_conditioned.envs.pendulum_odd import PendulumODD, THETA_MAX, OMEGA_LIM   # noqa: E402
from experiments.E008b_rescore import raw_value, g_grid, regions, decompose        # noqa: E402

WANDB_PROJECT = "odd-conditioned"


class GammaAnneal(BaseCallback):
    """Linear γ: γ0 → γ_end over the first `frac` of training, then hold. SB3 reads self.gamma
    at each train() call, so mutating it anneals the backup. Logs it."""
    def __init__(self, g0, g_end, total, frac=0.8):
        super().__init__()
        self.g0, self.g_end, self.total, self.frac = g0, g_end, total, frac

    def _on_step(self):
        p = min(self.num_timesteps / (self.frac * self.total), 1.0)
        self.model.gamma = self.g0 + p * (self.g_end - self.g0)
        if self.num_timesteps % 10_000 == 0:
            self.logger.record("odd/gamma", self.model.gamma)
        return True


def make_env(fixed_mass, obs_mode, spawn_omega_frac, seed):
    from stable_baselines3.common.monitor import Monitor
    kw = dict(odd_mode="static", obs_mode=obs_mode, resample_prob=0.0,   # STATIC — the fix
              spawn_omega_frac=spawn_omega_frac, seed=seed)
    if fixed_mass is not None:
        kw["mass_range"] = (fixed_mass, fixed_mass)      # specialist: pin the ODD
    return Monitor(PendulumODD(**kw))


def train_one(arm, fixed_mass, steps, seed, g0, g_end, spawn_omega_frac, out, device, use_wandb):
    from stable_baselines3.common.vec_env import DummyVecEnv
    from safety_sb3 import IsaacsSAC
    from odd_conditioned.callbacks import ODDSafeSetEval

    obs_mode = "blind" if arm == "blind" else "oracle"
    tag = arm if fixed_mass is None else f"{arm}_m{fixed_mass:g}"
    tag += f"_s{seed}"

    run = None
    if use_wandb:
        import wandb
        from datetime import datetime
        run = wandb.init(project=WANDB_PROJECT, group=f"E008c-{arm}",
                         name=f"{datetime.now():%Y%m%d-%H%M}_{tag}",
                         config=dict(arm=arm, fixed_mass=fixed_mass, steps=steps, seed=seed,
                                     g0=g0, g_end=g_end, spawn_omega_frac=spawn_omega_frac,
                                     static=True, normalized_feature=True, algo="IsaacsSAC(v0.2.0 avoid)"),
                         sync_tensorboard=True, reinit=True, save_code=False)

    env = DummyVecEnv([lambda: make_env(fixed_mass, obs_mode, spawn_omega_frac, seed)])
    model = IsaacsSAC("MlpPolicy", env, ctrl_action_dim=1, learning_rate=5e-4,
                      policy_kwargs=dict(net_arch=[128, 128, 128]), gamma=g0,
                      verbose=0, seed=seed, device=device,
                      tensorboard_log=os.path.join(out, "tb") if use_wandb else None)
    cbs = [GammaAnneal(g0, g_end, steps)]
    # specialists eval at their own rung; conditioned/blind eval across the ladder
    rungs = [fixed_mass] if fixed_mass is not None else [2.0, 3.5, 5.0, 6.5, 8.0]
    cbs.append(ODDSafeSetEval("results/E002_mass/sweep.npz", rungs, obs_mode,
                              eval_freq=max(steps // 6, 5000), use_wandb=use_wandb,
                              fig_dir=os.path.join(out, f"overlays_{tag}")))
    model.learn(total_timesteps=steps, progress_bar=False, callback=cbs)
    p = os.path.join(out, f"{tag}.zip")
    model.save(p)
    if run is not None:
        run.finish()
    return p


def score(zips, rungs, truth_npz, out):
    """Composed + region-decomposed scoring (E008b). Returns per-(arm,rung) in-support boundary
    optimism and composed IoU — the quantities the fire criterion compares."""
    from safety_sb3 import IsaacsSAC
    t = np.load(truth_npz)
    th, om, sweep, Vt = t["theta"], t["omega"], t["sweep"], t["V"]
    cell = (th[1] - th[0]) * (om[1] - om[0])
    g = g_grid(th, om); fail, outsup, insup = regions(th, om)
    # eval features must be normalized IDENTICALLY to training
    probe = PendulumODD(odd_mode="static", obs_mode="oracle", mass_range=(2.0, 8.0))
    res = {}
    for tag, (arm, fixed, seed, p) in zips.items():
        if not os.path.exists(p):
            continue
        model = IsaacsSAC.load(p, device="cpu")
        obs_mode = "blind" if arm == "blind" else "oracle"
        eval_rungs = [fixed] if fixed is not None else rungs
        for m in eval_rungs:
            feat = [probe.normalize_mass(m)]
            V = raw_value(model, th, om, feat, obs_mode)
            Vtrue = Vt[int(np.argmin(np.abs(sweep - m)))]
            res[f"{tag}__m{m:g}"] = decompose(V, Vtrue, g, cell, fail, outsup, insup)
    json.dump(res, open(os.path.join(out, "scores.json"), "w"), indent=2)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300_000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--rungs", type=float, nargs="+", default=[2.0, 3.5, 5.0, 6.5, 8.0])
    ap.add_argument("--specialist-rungs", type=float, nargs="+", default=[2.0, 5.0, 8.0])
    ap.add_argument("--g0", type=float, default=0.99)
    ap.add_argument("--g-end", type=float, default=0.999)
    ap.add_argument("--spawn-omega-frac", type=float, default=0.98)
    ap.add_argument("--out", default="results/E008c")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--score-only", action="store_true", help="skip training, re-score existing zips")
    ap.add_argument("--arm", help="train exactly ONE arm and exit (for parallel fan-out): "
                                  "conditioned | blind | specialist")
    ap.add_argument("--mass", type=float, help="fixed mass for --arm specialist")
    ap.add_argument("--seed", type=int, help="single seed for --arm mode")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # --- single-arm mode: one training, then exit. The driver fans 5 of these out in parallel.
    if args.arm:
        fixed = args.mass if args.arm == "specialist" else None
        train_one(args.arm, fixed, args.steps, args.seed if args.seed is not None else args.seeds[0],
                  args.g0, args.g_end, args.spawn_omega_frac, args.out, args.device, not args.no_wandb)
        return

    # the training plan: conditioned + blind + specialists, per seed
    plan = []  # (arm, fixed_mass, seed)
    for s in args.seeds:
        plan += [("conditioned", None, s), ("blind", None, s)]
        plan += [("specialist", m, s) for m in args.specialist_rungs]
    print(f"E008c: {len(plan)} trainings ({args.steps:,} steps each, seeds {args.seeds}), "
          f"γ {args.g0}→{args.g_end}, spawn_ω_frac {args.spawn_omega_frac}, STATIC ODD, "
          f"{torch.get_num_threads()} torch threads")

    zips = {}
    for arm, fixed, s in plan:
        tag = (arm if fixed is None else f"{arm}_m{fixed:g}") + f"_s{s}"
        p = os.path.join(args.out, f"{tag}.zip")
        if not (args.score_only and os.path.exists(p)):
            print(f"  train {tag} ...", flush=True)
            p = train_one(arm, fixed, args.steps, s, args.g0, args.g_end,
                          args.spawn_omega_frac, args.out, args.device, not args.no_wandb)
        zips[tag] = (arm, fixed, s, p)

    res = score(zips, args.rungs, "results/E002_mass/sweep.npz", args.out)

    # --- the fire criterion, per rung: specialist vs conditioned vs blind (in-support boundary optimism)
    print("\n" + "=" * 78)
    print("FIRE CRITERION — in-support boundary optimism (lower=better; 0=perfect boundary)")
    print(f"{'rung':>5} | {'specialist':>11} | {'conditioned':>12} | {'blind':>8}  (mean over seeds)")
    print("-" * 60)
    def avg(pred):
        vals = [v["optimism_in_support_boundary"] for k, v in res.items() if pred(k)]
        return float(np.mean(vals)) if vals else float("nan")
    for m in args.rungs:
        sp = avg(lambda k, m=m: k.startswith("specialist_m") and k.endswith(f"m{m:g}") and f"m{m:g}_s" in k)
        co = avg(lambda k, m=m: k.startswith("conditioned_s") and k.endswith(f"__m{m:g}"))
        bl = avg(lambda k, m=m: k.startswith("blind_s") and k.endswith(f"__m{m:g}"))
        print(f"{m:>5} | {sp:>11.3f} | {co:>12.3f} | {bl:>8.3f}")
    print("\nRead: specialists LOW at every rung + conditioned≈blind ⇒ FIRE (pivot to pathology).")
    print("      conditioned tracks specialists ⇒ PASS (capability paper alive).")
    print("      specialists ALSO high ⇒ BROKEN RECIPE (debug, don't pivot).")
    print("=" * 78)


if __name__ == "__main__":
    main()
