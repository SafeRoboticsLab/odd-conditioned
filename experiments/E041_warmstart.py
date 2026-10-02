"""E041 (probe B — WARM-START PROBE). E039 showed retraining blind/history from scratch on the SOUND
0.4.0 reach-avoid critic (ReachAvoidSAC2P, entropy removed from the critic TARGET per §7) COLLAPSED the
policy. Professor diagnosis (2026-08-03 daily log): v0.3.0's buggy soft target was accidentally a
soft->hard homotopy/curriculum; without it a FROM-SCRATCH policy falls into an absorbing "all-doomed"
fixed point (V ~= -0.1 clamp everywhere the policy fails, dead gradients). This probe tests the mechanism.

  WARM-START the COMPETENT v0.3.0 blind policy into a fresh 0.4.0 ReachAvoidSAC2P and fine-tune ~5M steps
  under the pure-HJ (0.4.0) target at FIXED gamma=0.9999 (no re-anneal, no re-softening).

  * HOLDS: in-dist eval safe_rate stays >=~0.9 -> the hard target is fine NEAR a competent policy; the
    cold-start homotopy mechanism is CONFIRMED (P1 two-stage fix de-risked).
  * DEGRADES from the competent start -> equilibrium cycling dominates; freeze the ctrl entirely in the fix.

Mechanics (reuses train_off_policy.py's model-construction path; NO edits to safety_sb3/RSS):
  - Build a fresh ReachAvoidSAC2P for `go2_payload_blind` with gamma FIXED at 0.9999, gamma_anneal OFF,
    hypers = configs/go2_payload_blind.yaml. Fresh (empty) replay buffer; num_timesteps starts at 0.
  - GRAFT the v0.3.0 GameplaySAC blind actor + dstb_actor + critic + critic_target (policy.pth group
    prefixes stripped, strict=False -- 47-dim obs, 12 ctrl, 3 dstb, nets 256x3/128x3 all match; E040 proved
    the actor graft). Seed the obs-normalizer from the v0.3.0 tensornormalize.pt (the stats that actor
    expects) with its large count, so it stays ~frozen through the fine-tune.
  - Adversary force is held at FULL (force_max=50, no re-ramp -> no task-difficulty curriculum either;
    the only softened-then-hardened knob we are testing is the critic target, and it stays hard). Per-env
    survival force scaling kept (PerEnvForceScaleCallback), same as the trainer.
  - SANITY: eval in-dist safe_rate ONCE post-graft pre-train; must be HIGH (~0.9+, matching v0.3.0) or the
    graft/normalizer is wrong -- fix before the long run. Then SafeSuccessRateEvalCallback logs the curve
    every 1M, checkpoint every 1M.

Run on ws3:  cd ~/odd-v040; MUJOCO_GL=egl \
  PYTHONPATH=external/safety-stable-baselines:external/robot-safety-sandbox \
  ~/miniconda3/envs/mjlab/bin/python experiments/E041_warmstart.py [--sanity-only] [--no-wandb]
"""
from __future__ import annotations
import argparse
import io
import os
import zipfile

os.environ.setdefault("MUJOCO_GL", "egl")
import torch as th  # noqa: E402
from stable_baselines3.common.callbacks import (  # noqa: E402
    CallbackList, CheckpointCallback)

import safety_sb3  # noqa: E402
from safety_sb3 import SafeSuccessRateEvalCallback  # noqa: E402
from safety_sb3.tensor_env import TensorVecNormalize  # noqa: E402
from robot_safety_sandbox import algo_name, make_tensor, spec  # noqa: E402
from robot_safety_sandbox.callbacks import (  # noqa: E402
    PerEnvForceScaleCallback, TensorNormSaveCallback, VideoWandbCallback)

TASK = "go2_payload_blind"
DEV = "cuda:0"
NENV = 512
STEPS = 5_000_000
GAMMA = 0.9999                       # FIXED hard target throughout (no anneal)
SEED = 0
OUT = "results/go2_payload_runs/go2_payload_blind_warmstart"
V030_ZIP = "results/go2_payload_runs/go2_payload_blind_gameplaysac/final_model.zip"
V030_NORM = "results/go2_payload_runs/go2_payload_blind_gameplaysac/tensornormalize.pt"
EVAL_FREQ = 1_000_000
CKPT_FREQ = 1_000_000
EVAL_ROLLOUTS = 100
EVAL_ENVS = 128


def v030_groups():
    """v0.3.0 GameplaySAC policy.pth -> {actor, dstb_actor, critic, critic_target} sub-state-dicts."""
    sd = th.load(io.BytesIO(zipfile.ZipFile(V030_ZIP).read("policy.pth")),
                 map_location=DEV, weights_only=False)
    out = {}
    for g in ("actor", "dstb_actor", "critic", "critic_target"):
        out[g] = {k[len(g) + 1:]: v for k, v in sd.items() if k.startswith(g + ".")}
    return out


def graft(model):
    """Overwrite the 0.4.0 policy submodules with the competent v0.3.0 weights."""
    g = v030_groups()
    for name in ("actor", "dstb_actor", "critic", "critic_target"):
        sub = getattr(model.policy, name)
        miss, unexp = sub.load_state_dict(g[name], strict=False)
        # The only tolerable missing/unexpected keys are non-parameter buffers; the
        # weight-bearing latent/mu/qf keys must all be present.
        bad = [k for k in miss if any(t in k for t in ("latent", "mu.", "log_std", "qf"))]
        assert not bad, f"{name} graft MISSING weight keys: {bad}"
        assert not unexp, f"{name} graft UNEXPECTED keys: {unexp}"
        print(f"  [graft] {name}: {len(g[name])} tensors loaded "
              f"(miss={len(miss)} unexp={len(unexp)})")


def seed_normalizer(env):
    """Seed the model's running obs-normalizer from the v0.3.0 stats (large count -> ~frozen)."""
    st = th.load(V030_NORM, map_location=DEV, weights_only=False)
    assert isinstance(env, TensorVecNormalize), f"model.env is {type(env)}, expected TensorVecNormalize"
    env.obs_mean = st["obs_mean"].to(DEV)
    env.obs_var = st["obs_var"].to(DEV)
    env.count = st["count"].to(DEV) if th.is_tensor(st["count"]) else th.as_tensor(st["count"], device=DEV)
    print(f"  [norm] seeded obs_mean/var from v0.3.0 (count={float(env.count):.1f})")


class _SyncedSafeSuccessEval(SafeSuccessRateEvalCallback):
    """Push the training normalizer stats into the eval env right before each eval fires
    (the eval env is frozen during eval, so it won't self-update). Mirrors train_off_policy.py."""
    def _on_step(self):
        if self.eval_freq > 0 and self.num_timesteps >= self._next_eval:
            tenv, ev = self.model.env, self.eval_env
            if hasattr(tenv, "obs_mean") and hasattr(ev, "obs_mean"):
                ev.obs_mean = tenv.obs_mean.clone()
                ev.obs_var = tenv.obs_var.clone()
                ev.count = tenv.count.clone() if th.is_tensor(tenv.count) else tenv.count
        return super()._on_step()


def build_model(env):
    Algo = getattr(safety_sb3, algo_name(TASK, adversary=True, family="off_policy"))  # ReachAvoidSAC2P
    s = spec(TASK)
    akw = dict(
        normalize_obs=True,
        gamma=GAMMA, gamma_anneal=False,      # <-- pure hard target, no re-anneal
        min_alpha=1e-3, max_alpha=None,
        learning_rate=1e-4, tau=0.01, target_update_interval=2,
        ent_coef="auto_0.1",
        buffer_size=1_000_000, batch_size=4096, train_freq=1, gradient_steps=4,
        learning_starts=5 * NENV,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256, 256], qf=[128, 128, 128])),
        seed=SEED, verbose=1, device=DEV, tensorboard_log=OUT,
        terminal_type="all",
        ctrl_action_dim=s.ctrl_dim,
        use_leaderboard=False,                # probe reads safe_rate eval, not the league
    )
    model = Algo("MlpPolicy", env, **akw)
    return model


def make_eval_env():
    ev = TensorVecNormalize(make_tensor(TASK, EVAL_ENVS, DEV, adversary=True))
    return ev


def sanity_eval(model):
    """One post-graft, pre-train in-dist safe_rate eval (same protocol as the training callback)."""
    ev = make_eval_env()
    ev.obs_mean = model.env.obs_mean.clone()
    ev.obs_var = model.env.obs_var.clone()
    ev.count = model.env.count.clone() if th.is_tensor(model.env.count) else model.env.count
    cb = SafeSuccessRateEvalCallback(ev, n_rollouts=EVAL_ROLLOUTS, eval_freq=1,
                                     reach_avoid=True, verbose=0)
    cb.model = model
    safe, succ, eplen, n = cb._run_eval()
    ev.close()
    print(f"[SANITY graft] in-dist safe_rate={safe:.3f} success_rate={succ:.3f} "
          f"ep_len_mean={eplen:.1f} (n={n})   <- expect HIGH ~0.9+ if graft is correct")
    return safe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sanity-only", action="store_true",
                    help="graft + one eval, then exit (no training/launch)")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    print(f"[E041] warm-start probe: task={TASK} nenv={NENV} steps={STEPS} "
          f"gamma={GAMMA} (anneal OFF)  out={OUT}")

    env = make_tensor(TASK, NENV, DEV, adversary=True)     # force_max=50 default (full, no re-ramp)
    model = build_model(env)
    print("[E041] grafting v0.3.0 competent blind weights ...")
    graft(model)
    seed_normalizer(model.env)

    safe0 = sanity_eval(model)
    if args.sanity_only:
        print("[E041] --sanity-only: done.")
        return
    if safe0 < 0.7:
        raise SystemExit(f"[E041] ABORT: post-graft safe_rate {safe0:.3f} < 0.7 -- graft/normalizer "
                         "is wrong; fix before the 5M run (do NOT launch a broken warm start).")

    # --- callbacks: checkpoints + norm + full-force survival curriculum + safe_rate eval ---
    eval_env = make_eval_env()
    cbs = [
        CheckpointCallback(save_freq=max(1, CKPT_FREQ // NENV),
                           save_path=os.path.join(OUT, "checkpoints"), name_prefix="model"),
        TensorNormSaveCallback(os.path.join(OUT, "checkpoints"), save_freq_steps=CKPT_FREQ),
        PerEnvForceScaleCallback(lo=0.3, init=0.5),        # force_max stays 50 (no ForceRampCallback)
        _SyncedSafeSuccessEval(eval_env, n_rollouts=EVAL_ROLLOUTS, eval_freq=EVAL_FREQ,
                               reach_avoid=True, verbose=1),
    ]

    if not args.no_wandb:
        try:
            import wandb
            from wandb.integration.sb3 import WandbCallback
            wandb.init(project="odd-conditioned", name="go2_payload_blind_warmstart",
                       config=dict(task=TASK, steps=STEPS, gamma=GAMMA, gamma_anneal=False,
                                   warm_start="v0.3.0_blind_gameplaysac", probe="E041"),
                       sync_tensorboard=True, save_code=False, reinit=True)
            cbs.append(WandbCallback(verbose=0))
            _vtask = TASK + "_video"
            from robot_safety_sandbox import list_tasks
            _vtask = _vtask if _vtask in list_tasks() else TASK
            cbs.append(VideoWandbCallback(
                lambda: make_tensor(_vtask, 8, DEV, adversary=False, render_mode="rgb_array"),
                interval=2_000_000))
        except Exception as e:
            print(f"[E041] wandb disabled ({type(e).__name__}: {e}); metrics still go to TB + stdout")

    print(f"[E041] launching {STEPS} steps of pure-HJ fine-tune from the competent graft ...")
    model.learn(total_timesteps=STEPS, callback=CallbackList(cbs))
    model.save(os.path.join(OUT, "final_model.zip"))
    if hasattr(model.env, "save"):
        model.env.save(os.path.join(OUT, "tensornormalize.pt"))
    print(f"[E041] done -> {OUT}")


if __name__ == "__main__":
    main()
