"""E006 — Can ONE conditioned critic represent the whole ODD family?

This is the question the whole project reduces to. E001 established the maximal family is what
we want (the classical CLF/CBF region is 1.6-3.8x smaller, structurally -- an ellipse cannot
fill a parallelogram). E003/E004/E005 built the family as exact ground truth. The theory is
inherited (Gandhi & Mhaskar 2008 state the handoff condition; LTS99 gives its maximal form)
and uncomputable past ~5 states. What is UNestablished is computational:

    Can a single neural critic, conditioned on the ODD, recover the family that a grid can
    only produce one slice at a time -- without collapsing to the worst case?

THE THREE ARMS (this is the experiment; the rest is plumbing)
-------------------------------------------------------------
  specialist  -- one critic per ODD rung. The REFERENCE: what RL can do when it doesn't have
                 to share. Scored against grid ground truth to prove the RL itself is sound.
  conditioned -- ONE critic, ODD handed to it (oracle). The upper bound on conditioning:
                 tests REPRESENTATION only.
  blind       -- ONE critic, no ODD input. Cannot express the family, so its optimum under a
                 mixed-ODD distribution is a compromise. The MOTIVATIONAL BASELINE -- collapse
                 here is nearly tautological and is NOT a finding (proposal 4).

  conditioned vs specialist = the cost of sharing one network   <- the real question
  blind vs conditioned      = why you must condition at all     <- the tautology

REPORT DIFFERENTIALS, NEVER ABSOLUTE VIOLATION RATES. The value-based filter leaks by
construction -- Provably Optimal RL's own experiments show the rollout filter hits zero
violations while the value filter does not. A violation under any arm may be generic critic
error rather than a conditioning effect. Specialists use the SAME filter, so the differential
isolates the effect.

WHAT WE ARE LOOKING FOR (proposal 4-revised)
--------------------------------------------
  1. COLLAPSE   -- conditioned safe-set volume -> the worst rung's. Premise confirmed.
     NB: theta must move the (|g|, l) MAGNITUDES, not just the dynamics, for the formally
     predicted mechanism (the risk dial p* = |g|/(|g|+l)) to be in play at all.
  2. OPTIMISM   -- conditioned claims sets LARGER than ground truth. The dangerous failure:
     an over-claiming filter is unsafe. Expected from adversary staleness -- with theta
     sampled across the family, each rung gets a fraction of the adversary's updates, and the
     best-response disturbance differs per rung, so the critic is optimistic exactly at the
     hard rungs.
  3. PARITY     -- conditioned ~= specialist ~= truth. Then conditioning is free, the
     architecture contribution is thin, and the job becomes finding where it breaks.

PRECONDITIONS (each cost the lab a failed multi-hour run -- vault: Reach-avoid RL gotchas)
------------------------------------------------------------------------------------------
  * STATIC LR under two-player. KL-adaptive LR is single-player; in an ISAACS ctrl<->dstb
    phase machine the adversary's ~0 KL balloons the SHARED lr and the next ctrl update takes
    a catastrophic step (observed: 0.88 -> 0.14 success).
  * clip_range_vf ~ 0.2. A fresh critic + hard domain randomization on a warm-started actor
    diverges (value_loss 1e9 -> 1e14, unrecoverable). ODD-CONDITIONING IS A FORM OF DR, so
    this is a PREDICTED failure of the very first conditioned run. Healthy value_loss ~ 1e-2.
  * Reward IS the margin -- never VecNormalize(norm_reward=True); it moves the zero level set,
    i.e. the safety boundary.
  * Pass hyperparameters explicitly; the sandbox's train.py defaults drifted from the recipe
    read back off the shipped Digit checkpoints.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from odd_conditioned.envs.pendulum_odd import PendulumODD, THETA_MAX, OMEGA_LIM  # noqa: E402


# --------------------------------------------------------------------------- scoring

def learned_safe_set(model, theta_grid, omega_grid, odd_feat, obs_mode, device="cpu"):
    """V_hat(x) on a (theta, omega) grid at a fixed ODD, from the critic.

    For a two-player critic the deployed value is
        V(x) = Q(x, pi_ctrl(x), pi_dstb(x, pi_ctrl(x)))
    i.e. the control's best response met by the adversary's best response -- the same object
    safe_adaptation_dev/script/plot_safe_set.py evaluates. Anything else (e.g. Q at a zero
    disturbance) silently scores an easier game than the one that was trained.
    """
    import torch

    TH, OM = np.meshgrid(theta_grid, omega_grid, indexing="ij")
    core = np.stack([np.sin(TH).ravel(), np.cos(TH).ravel(), OM.ravel()], axis=1)
    if obs_mode == "oracle":
        feat = np.tile(np.asarray(odd_feat, dtype=np.float32), (core.shape[0], 1))
        obs = np.concatenate([core, feat], axis=1).astype(np.float32)
    elif obs_mode == "blind":
        obs = core.astype(np.float32)
    else:
        raise NotImplementedError("history obs_mode needs a rollout to populate the trace")

    with torch.no_grad():
        t = torch.as_tensor(obs, device=device)
        # NOT model.predict(): IsaacsPolicy.predict returns the CONTROL action only (it is the
        # deployment path -- the safety controller with no adversary). The deployed VALUE needs
        # both players. Query the two actors and concatenate, in the [-1,1] space the critic was
        # trained on (SB3 stores scale_action'd actions in the replay buffer).
        a_ctrl = model.policy.actor(t, deterministic=True)
        a_dstb = model.policy.dstb_actor(t, deterministic=True)
        a = torch.cat([a_ctrl, a_dstb], dim=1)
        q = model.critic(t, a)
        # twin critics -> min, matching the conservative target used in training
        q = torch.min(*q) if isinstance(q, (list, tuple)) else q
        V = q.cpu().numpy().reshape(TH.shape)
    return V


def compare(V_learned, V_true, cell_area):
    """Volumes + the two signed errors that matter, kept separate.

    Borrowing PBF's h*-sign device: one signed scalar separating under- from over-conservative.
      optimism  -- learned says SAFE where truth says UNSAFE. The dangerous direction: a filter
                   that over-claims will let a task policy walk into a state it cannot save.
      conserv.  -- learned says UNSAFE where truth says SAFE. Costs liveness, not safety.
    Reporting only IoU would average these into one number and hide which way it fails.
    """
    lm, tm = V_learned >= 0, V_true >= 0
    inter = float((lm & tm).sum() * cell_area)
    union = float((lm | tm).sum() * cell_area)
    return dict(
        vol_learned=float(lm.sum() * cell_area),
        vol_true=float(tm.sum() * cell_area),
        vol_ratio=float(lm.sum() / max(tm.sum(), 1)),
        iou=inter / max(union, 1e-12),
        optimism=float((lm & ~tm).sum() * cell_area),      # UNSAFE direction
        conservatism=float((~lm & tm).sum() * cell_area),  # liveness cost
    )


# --------------------------------------------------------------------------- training

def make_env(odd_mode, obs_mode, fixed_mass=None, seed=0):
    from stable_baselines3.common.monitor import Monitor

    kw = dict(odd_mode=odd_mode, obs_mode=obs_mode, seed=seed)
    if fixed_mass is not None:
        # specialist: pin the ODD and stop resampling it mid-episode
        kw.update(mass_range=(fixed_mass, fixed_mass), resample_prob=0.0)
    return Monitor(PendulumODD(**kw))


WANDB_PROJECT = "odd-conditioned"      # canonical name from docs/log/experiments.md.
                                       # NEVER the framework default.


def _wandb_init(arm, tag, steps, fixed_mass, odd_mode, seed, git_hash):
    """project / group / run-name per the project-log skill:
         project = the canonical name (never the framework default)
         group   = the experiment name        e.g. E006-conditioned
         run     = YYYYMMDD-HHMM_<delta>      e.g. 20260716-2015_m8
    Weights stay LOCAL -- no wandb.save, no weight artifacts.
    """
    import wandb
    from datetime import datetime

    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    return wandb.init(
        project=WANDB_PROJECT, group=f"E006-{arm}", name=f"{stamp}_{tag}",
        config=dict(arm=arm, steps=steps, fixed_mass=fixed_mass, odd_mode=odd_mode,
                    seed=seed, git=git_hash, algo="IsaacsSAC", net_arch=[128, 128, 128],
                    lr=5e-4, lr_schedule="STATIC (KL-adaptive is single-player)",
                    ctrl_action_dim=1, margin_mode="avoid (l_neg -> matches E003 avoid truth)"),
        reinit=True, save_code=False,
    )


def train(arm, steps, fixed_mass, odd_mode, seed, out_dir, device,
          truth=None, rungs=(), use_wandb=True, git_hash="unknown"):
    from stable_baselines3.common.vec_env import DummyVecEnv
    from safety_sb3 import IsaacsSAC

    obs_mode = "blind" if arm == "blind" else "oracle"
    tag = arm if fixed_mass is None else f"{arm}_m{fixed_mass:g}"
    run = _wandb_init(arm, tag, steps, fixed_mass, odd_mode, seed, git_hash) if use_wandb else None

    env = DummyVecEnv([lambda: make_env(odd_mode, obs_mode, fixed_mass, seed)])

    model = IsaacsSAC(
        "MlpPolicy", env,
        ctrl_action_dim=1,                  # action = [u, F]: 1 leading ctrl dim, rest is dstb
        dstb_update_period=1,               # timescale separation knob. The adversary must
                                            # TRACK its best response or the critic goes
                                            # optimistic at the hard rungs -- that is the T2
                                            # staleness hypothesis, so this is a lever, not a
                                            # default to leave alone.
        learning_rate=5e-4,                 # STATIC. KL-adaptive lr is single-player and
                                            # catastrophic under two-player (see header).
        policy_kwargs=dict(net_arch=[128, 128, 128]),   # the verified recipe, explicit --
                                            # the sandbox's defaults drifted to [512,256,128]
        gamma=0.99, verbose=0, seed=seed, device=device,
        tensorboard_log=os.path.join(out_dir, "tb") if use_wandb else None,
    )

    cbs = []
    if truth is not None and rungs:
        from odd_conditioned.callbacks import ODDSafeSetEval
        # Return is meaningless here (reward IS the margin, so it only says "did not fall").
        # What we watch is the learned SET vs the grid's, per rung -- collapse and optimism
        # are only visible there.
        cbs.append(ODDSafeSetEval(truth, rungs, obs_mode, eval_freq=max(steps // 8, 5_000),
                                  use_wandb=use_wandb,
                                  fig_dir=os.path.join(out_dir, f"overlays_{tag}")))

    model.learn(total_timesteps=steps, progress_bar=False, callback=cbs or None)

    p = os.path.join(out_dir, f"{tag}.zip")
    model.save(p)        # LOCAL only -- never wandb.save / weight artifacts
    if run is not None:
        run.finish()
    return model, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/E006")
    ap.add_argument("--odd-mode", default="static", choices=["static", "sine"])
    ap.add_argument("--steps", type=int, default=300_000)
    ap.add_argument("--rungs", type=float, nargs="+", default=[2.0, 3.5, 5.0, 6.5, 8.0],
                    help="ODD rungs to score against ground truth (and to train specialists on)")
    ap.add_argument("--arms", nargs="+", default=["conditioned", "blind", "specialist"])
    ap.add_argument("--truth", default="results/E002_mass/sweep.npz",
                    help="E003's ground-truth family — scored against directly, no re-solving")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")   # 2-D obs, tiny nets: CPU beats GPU here
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import subprocess
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                           text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        if dirty:
            git_hash += "-DIRTY"      # a dirty launch is a reproducibility hole; record it
    except Exception:
        git_hash = "unknown"
    print(f"code: {git_hash}")

    truth = np.load(args.truth)
    th_g, om_g, sweep, Vtrue = truth["theta"], truth["omega"], truth["sweep"], truth["V"]
    cell = (th_g[1] - th_g[0]) * (om_g[1] - om_g[0])
    print(f"ground truth: {args.truth}  ({len(sweep)} rungs, {sweep[0]:g}..{sweep[-1]:g} kg)")

    def truth_at(m):
        return Vtrue[int(np.argmin(np.abs(sweep - m)))]

    kw = dict(truth=args.truth, rungs=args.rungs, use_wandb=not args.no_wandb, git_hash=git_hash)
    results, t0 = {}, time.time()
    for arm in args.arms:
        if arm == "specialist":
            for m in args.rungs:
                model, p = train(arm, args.steps, m, args.odd_mode, args.seed, args.out,
                                 args.device, **kw)
                V = learned_safe_set(model, th_g, om_g, [m], "oracle", args.device)
                results[f"specialist_m{m:g}"] = compare(V, truth_at(m), cell)
                print(f"  specialist m={m:g}: {results[f'specialist_m{m:g}']}")
        else:
            model, p = train(arm, args.steps, None, args.odd_mode, args.seed, args.out,
                             args.device, **kw)
            for m in args.rungs:
                V = learned_safe_set(model, th_g, om_g, [m],
                                     "blind" if arm == "blind" else "oracle", args.device)
                results[f"{arm}_m{m:g}"] = compare(V, truth_at(m), cell)
                print(f"  {arm} m={m:g}: {results[f'{arm}_m{m:g}']}")

    with open(os.path.join(args.out, "scores.json"), "w") as f:
        json.dump({"results": results, "rungs": args.rungs, "steps": args.steps,
                   "odd_mode": args.odd_mode, "truth": args.truth}, f, indent=2)
    print(f"\ntotal {time.time()-t0:.0f}s -> {args.out}/scores.json")


if __name__ == "__main__":
    main()
