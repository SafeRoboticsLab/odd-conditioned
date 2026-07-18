"""E013 (Rung 1) — the privileged CONSTANT-μ conditioned reach-avoid critic on FrictionBicycleEnv.

The friction-toy analog of E008c (which first validated the thesis at 2-D on a mass ODD): does ONE
conditioned critic V(x;μ) recover the true safe-set FAMILY across μ∈[0.1,1.0], while a BLIND critic is
frozen at one worst-case-averaged set? Arms (E008c protocol):
  * conditioned : obs=mu_local, μ~U[0.1,1.0] CONSTANT per episode  <- the privileged critic
  * blind       : obs=blind,    μ~U[0.1,1.0] constant per episode but NOT observed  <- worst-case collapse
  * spec_hi/lo  : obs=blind,    μ fixed 1.0 / 0.1  <- per-ODD specialist ceilings

CONSTANT-μ per episode is load-bearing (env docstring / 2026-07-18 decision): conditioning on the
instantaneous μ under a time-varying field makes the Bellman target learn the DR-AVERAGED value
(E008b blind_dyn pathology). Slices are trained static; time-variation enters only at Rung 4 (V*).

Algorithm: safety_sb3 ReachAvoidSAC (single-player reach-avoid, no adversary yet — Buzi). Torch threads
capped to 4 (E008c lesson: 24-thread default thrashes tiny nets). Scoring (IoU-vs-μ against the E011
grid slices, re-solved with the env's GRADED l for value-parity) is a separate --score-only pass.
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
from odd_conditioned.envs.friction_bicycle import (  # noqa: E402
    FrictionBicycleEnv, ConstantMu, MU_RANGE)

WANDB_PROJECT = "odd-conditioned"
WANDB_ENTITY = "buzinguyen"

# (tag, obs_mode, fixed_mu)
ARMS = {
    "conditioned": ("mu_local", None),   # μ observed, varies per episode  <- THE critic
    "blind":       ("blind",    None),   # μ varies, NOT observed          <- worst-case collapse
    "spec_hi":     ("blind",    1.0),    # dry specialist ceiling
    "spec_lo":     ("blind",    0.1),    # ice specialist ceiling
}


def make_env(obs_mode, fixed_mu, seed):
    from stable_baselines3.common.monitor import Monitor
    prov = ConstantMu(mu_range=MU_RANGE, fixed=fixed_mu)
    return Monitor(FrictionBicycleEnv(mu_provider=prov, obs_mode=obs_mode, randomize=True, seed=seed))


def train(tag, steps, seed, out, device, use_wandb, git_hash):
    from stable_baselines3.common.vec_env import DummyVecEnv
    from safety_sb3 import ReachAvoidSAC
    obs_mode, fixed_mu = ARMS[tag]

    run = None
    if use_wandb:
        import wandb
        from datetime import datetime
        run = wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, group=f"E013-{tag}",
                         name=f"{datetime.now():%Y%m%d-%H%M}_{tag}_s{seed}",
                         config=dict(tag=tag, obs_mode=obs_mode, fixed_mu=fixed_mu, steps=steps,
                                     seed=seed, git=git_hash, mu_range=MU_RANGE,
                                     algo="ReachAvoidSAC (single-player, friction toy Rung 1)"),
                         sync_tensorboard=True, reinit=True, save_code=False)
    env = DummyVecEnv([lambda: make_env(obs_mode, fixed_mu, seed)])
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/E013")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--arm", help="train exactly ONE arm and exit (parallel fan-out)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import subprocess
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
            git_hash += "-DIRTY"
    except Exception:
        git_hash = "?"
    print(f"E013 Rung1  code {git_hash}  steps {args.steps}  threads {torch.get_num_threads()}")

    if args.arm:
        print(f"  train {args.arm} {ARMS[args.arm]} ...", flush=True)
        train(args.arm, args.steps, args.seed, args.out, args.device, not args.no_wandb, git_hash)
        return
    for tag in ARMS:
        print(f"  train {tag} ...", flush=True)
        train(tag, args.steps, args.seed, args.out, args.device, not args.no_wandb, git_hash)


if __name__ == "__main__":
    main()
