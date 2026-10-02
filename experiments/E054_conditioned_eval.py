"""E054 — the PAYOFF eval for the leg ODD: does one ODD-conditioned policy match the BEST specialist at every
FR-torque, and beat the blind worst-case?

Overlays SIX policies on the E049 survival-vs-torque axis, under the same fixed scripted lateral pull:
  SPECIALISTS (fixed-θ, the achievable frontier):
    normal (θ1.0)  results/go2_weak_leg_runs/go2_stabilize_adv
    weak50 (θ0.5)  results/go2_weak_leg_runs/go2_weak_leg_adv
    weak20 (θ0.2)  results/go2_weak_leg_runs/go2_weak_leg_20_adv        (E050, retrained @25N)
  RANDOMIZED-θ ARMS (one policy over θ∈[0.2,1.0]):
    blind          results/go2_weak_leg_runs/go2_weak_leg_blind_adv     (θ hidden — worst-case baseline)
    conditioned    results/go2_weak_leg_runs/go2_weak_leg_conditioned_adv(θ on actor+critic — oracle)
    history        results/go2_weak_leg_runs/go2_weak_leg_history_adv    (K=16 stack — deployable)

Each policy is evaluated at a grid of FIXED test-θ. Obs surfaces differ, so each arm evals on its own task:
  specialists + blind : go2_weak_leg_sweep_{pct}          (47-dim)
  conditioned         : go2_weak_leg_conditioned_at_{pct} (48-dim, θ pinned = test θ)
  history             : go2_weak_leg_history_at_{pct}      (752-dim)
Metric = falls/env-second (reset-independent, E037). Scripted pull [0,1,0] at force_scale × 50N.

READ: conditioned/history should hug the LOWER ENVELOPE of the specialists (match the best fixed-θ policy at
each θ) while blind sits ABOVE (pays the worst-case tax). That is the value of ODD-conditioning.
"""
import os, sys, json
from _paths import _ART
os.environ.setdefault("MUJOCO_GL", "egl")
import torch as th
sys.path.insert(0, "external/robot-safety-sandbox")
from robot_safety_sandbox import make_tensor, spec
from robot_safety_sandbox.eval.policies import load_twin

DEV = "cuda:0"; PULL = th.tensor([0., 1., 0.]); DT = 0.02; STEPS = 250; NENV = 128
CK = "results/go2_weak_leg_runs/{run}/checkpoints/model_49999872_steps.zip"
PCTS = [100, 70, 50, 30, 20, 10]
FORCE_SCALES = [0.5, 0.7]                       # 25N (training) / 35N (harder)

# policy -> (run dir, eval-task template). {pct} filled per grid point.
POLICIES = {
    "spec normal(1.0)":  ("go2_stabilize_adv",           "sweep"),
    "spec weak50(0.5)":  ("go2_weak_leg_adv",            "sweep"),
    "spec weak20(0.2)":  ("go2_weak_leg_20_adv",         "sweep"),
    "blind":             ("go2_weak_leg_blind_adv",      "sweep"),
    "conditioned":       ("go2_weak_leg_conditioned_adv","cond"),
    "history":           ("go2_weak_leg_history_adv",    "hist"),
}
TASK = {"sweep": lambda p: "go2_stabilize" if p == 100 else f"go2_weak_leg_sweep_{p}",
        "cond":  lambda p: f"go2_weak_leg_conditioned_at_{p}",
        "hist":  lambda p: f"go2_weak_leg_history_at_{p}"}


def rollout(model, norm, task, fr):
    env = make_tensor(task, NENV, DEV, adversary=True)
    env.force_scale = fr * th.ones(NENV, device=DEV)
    dd = spec(task).dstb_dim
    dstb = (PULL / PULL.norm()).to(DEV)[None].expand(NENV, dd).contiguous()
    obs = env.reset()
    nfall = 0; fell = th.zeros(NENV, dtype=th.bool, device=DEV)
    for t in range(STEPS):
        with th.no_grad():
            a = th.clamp(model.policy._predict(norm(obs), deterministic=True), -1, 1)
        obs, g, dones, touts, l = env.step_tensor(th.cat([a, dstb], dim=1))
        fail = dones & (~touts)
        nfall += int(fail.sum()); fell |= fail
    env.close()
    return nfall / (NENV * STEPS * DT), float(fell.float().mean())


if __name__ == "__main__":
    twins = {}
    for name, (run, _kind) in POLICIES.items():
        twins[name] = load_twin(CK.format(run=run), DEV, quiet=True)
        print(f"loaded {name:20s} <- {run} ({type(twins[name][0]).__name__})")
    results = {}
    for fr in FORCE_SCALES:
        print(f"\n===== scripted pull {int(fr*50)} N =====")
        print(f"{'θ%':>4} | " + " | ".join(f"{n:^18}" for n in POLICIES))
        for p in PCTS:
            cells = []
            for name, (_run, kind) in POLICIES.items():
                model, norm = twins[name]
                try:
                    fps, frac = rollout(model, norm, TASK[kind](p), fr)
                    results[(fr, name, p)] = (fps, frac)
                    cells.append(f"{fps:5.2f} ({frac:.2f})")
                except Exception as e:
                    cells.append(f"ERR {type(e).__name__}")
            print(f"{p:>4} | " + " | ".join(f"{c:^18}" for c in cells))
    out = os.path.expanduser(_ART + "/E054-conditioned-eval")
    os.makedirs(out, exist_ok=True)
    json.dump({f"{k[0]}|{k[1]}|{k[2]}": v for k, v in results.items()}, open(f"{out}/results.json", "w"), indent=2)
    print(f"\nsaved -> {out}/results.json")
