"""E019 break-boundary sweep (Leg B, step 1) — find where the trained arms FALL.

The silent-failure premise (blind/history survive OOD but can't DETECT it) is currently an ASSERTION —
they survived every OOD we tried. This sweep EXHIBITS the break: push payload mass (sloshy, top-heavy)
past the training max (7.5 kg) and see where each arm's safe_rate collapses. If mass alone doesn't break
them, escalate to a model-class violation (shifting-CoM / asymmetric) — the professor's contingency.

Metric: safe_rate = frac of envs that NEVER fell, at force 0.0 (can it even stand?) and 1.0 (50 N pull).
The break boundary = the mass at which safe_rate drops below ~1.0 for each arm.
"""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
import torch as th
sys.path.insert(0, "external/robot-safety-sandbox")
import safety_sb3
from robot_safety_sandbox import make_tensor, spec, algo_name
from safety_sb3.tensor_env import TensorVecNormalize

RUNS = "results/go2_payload_runs"
PULL = th.tensor([0.0, 1.0, 0.0])
MASSES = [8, 12, 16, 20, 25]
ARMS = {"CONDITIONED": f"{RUNS}/go2_payload_conditioned_gameplaysac",
        "BLIND": f"{RUNS}/go2_payload_blind_gameplaysac",
        "HISTORY": f"{RUNS}/go2_payload_history_gameplaysac"}
OBS = {"CONDITIONED": "conditioned", "BLIND": "blind", "HISTORY": "history"}


def safe_rate(run_dir, task, dev, fr, steps=220, nenv=128):
    env = make_tensor(task, nenv, dev, adversary=True)
    Algo = getattr(safety_sb3, algo_name(task, adversary=True).replace("PPO", "SAC"))
    model = Algo.load(f"{run_dir}/final_model.zip", env=env, device=dev,
                      custom_objects={"_use_lb": False, "_lb_dir": "/tmp/lb", "tensorboard_log": None, "buffer_size": 1})
    norm = TensorVecNormalize.load(f"{run_dir}/tensornormalize.pt", env); norm.training = False
    n = env.num_envs; dd = spec(task).dstb_dim
    env.force_scale = fr * th.ones(n, device=dev)
    dstb = (PULL / PULL.norm()).to(dev)[None].expand(n, dd).contiguous()
    robot = env.mj.scene["robot"]; obs = env.reset()
    fell = th.zeros(n, dtype=th.bool, device=dev)
    for _ in range(steps):
        with th.no_grad():
            a = th.clamp(model.policy._predict(norm.normalize_obs(obs), deterministic=True), -1, 1)
        obs, g, dones, touts, l = env.step_tensor(th.cat([a, dstb], dim=1))
        fell |= (-robot.data.projected_gravity_b[:, 2] < 0.4)
    env.close()
    return 1.0 - float(fell.float().mean())


if __name__ == "__main__":
    dev = "cuda:0"
    arms = list(ARMS)
    print(f"\nE019 break-boundary — safe_rate (frac never fell). sloshy top-heavy payload; train max = 7.5 kg.")
    for fr in (0.0, 1.0):
        print(f"\n=== force {fr} ({fr*50:.0f} N pull) ===")
        print(f"{'mass (kg)':10} | " + " | ".join(f"{a:>12}" for a in arms))
        for m in MASSES:
            cells = [f"{safe_rate(ARMS[a], f'go2_payload_sweep_{OBS[a]}_{m}', dev, fr):.2f}" for a in arms]
            print(f"{m:>7}    | " + " | ".join(f"{c:>12}" for c in cells))
    print("\nbreak boundary = mass where safe_rate drops < 1.0. If none break by 25kg -> escalate to model-class violation.")
