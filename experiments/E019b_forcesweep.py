"""E019b force sweep — the make-or-break test for the project premise: is there ANY regime where BLIND
FALLS but a theta-INFORMED policy SURVIVES? The mass sweep (E019) showed conditioned <= blind everywhere
(knowing theta only HURTS via OOD fragility). The remaining hope is FORCE: at high enough pull, blind gets
dragged PAST RECOVERY (reaction insufficient), while a policy that knows it's heavy-sloshy braces hard and
survives. If blind never breaks below where conditioned also breaks, the premise fails on this task.

IN-DIST payloads (theta valid): light-rigid (1.2kg k=300), heavy-sloshy (7kg k=0). Force sweep = 50..250 N.
Metric: safe_rate = frac never fell. WANT: a cell where BLIND < CONDITIONED (theta-knowledge saves it).
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
FORCES = [1.0, 2.0, 3.0, 4.0, 5.0]   # x force_max 50 => 50..250 N
COND = {"light_rigid": ("go2_payload_conditioned_light_rigid", "go2_payload_light_rigid", "go2_payload_history_light_rigid"),
        "heavy_sloshy": ("go2_payload_conditioned_heavy_sloshy", "go2_payload_heavy_sloshy", "go2_payload_history_heavy_sloshy")}
ARM_RUNS = [("CONDITIONED", f"{RUNS}/go2_payload_conditioned_gameplaysac"),
            ("BLIND", f"{RUNS}/go2_payload_blind_gameplaysac"),
            ("HISTORY", f"{RUNS}/go2_payload_history_gameplaysac")]


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
    print("\nE019b force sweep — safe_rate (frac never fell). IN-DIST payloads (theta valid). WANT: BLIND < CONDITIONED.")
    for payload, tasks in COND.items():
        by_arm = dict(zip(["CONDITIONED", "BLIND", "HISTORY"], tasks))
        print(f"\n=== {payload} (in-dist) ===")
        print(f"{'force (N)':10} | " + " | ".join(f"{a:>12}" for a, _ in ARM_RUNS))
        for fr in FORCES:
            cells = [f"{safe_rate(rd, by_arm[a], dev, fr):.2f}" for a, rd in ARM_RUNS]
            print(f"{fr*50:>7.0f}    | " + " | ".join(f"{c:>12}" for c in cells))
    print("\nIf a cell has BLIND < CONDITIONED -> theta-knowledge SAVES it -> premise LIVES. If blind falls only")
    print("where conditioned also falls -> force just breaks everyone -> need model-class / anticipation task.")
