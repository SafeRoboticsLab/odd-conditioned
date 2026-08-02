"""E018 PRE-CHECK (professor's gate for the evidence-expires probe): does the ORACLE-informed first-0.3s
impulse response DIFFER by payload type? If the optimal first-~0.3s reaction to a 50 N step is the SAME
for light-rigid vs heavy-sloshy, then pre-knowledge of theta buys nothing in the impulse window and the
evidence-expires probe would tie 4-ways -> redesign the impulse before training anything.

Protocol: settle to a quiet stance (no force), then apply a 50 N (+y) step; record the base drift + tilt
trajectory over the first 0.5 s for the CONDITIONED policy at the matching fixed theta (= an oracle
specialist). Compare light-rigid vs heavy-sloshy at t = 0.1 / 0.2 / 0.3 s.
"""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
import torch as th
sys.path.insert(0, "external/robot-safety-sandbox")
import safety_sb3
from robot_safety_sandbox import make_tensor, spec, algo_name
from safety_sb3.tensor_env import TensorVecNormalize

RUN = "results/go2_payload_runs/go2_payload_conditioned_gameplaysac"
PULL = th.tensor([0.0, 1.0, 0.0])


def impulse(task, dev, settle=30, horizon=25, nenv=128):
    env = make_tensor(task, nenv, dev, adversary=True)
    Algo = getattr(safety_sb3, algo_name(task, adversary=True).replace("PPO", "SAC"))
    model = Algo.load(f"{RUN}/final_model.zip", env=env, device=dev,
                      custom_objects={"_use_lb": False, "_lb_dir": "/tmp/lb", "tensorboard_log": None, "buffer_size": 1})
    norm = TensorVecNormalize.load(f"{RUN}/tensornormalize.pt", env); norm.training = False
    n = env.num_envs; dd = spec(task).dstb_dim
    dstb = (PULL / PULL.norm()).to(dev)[None].expand(n, dd).contiguous()
    robot = env.mj.scene["robot"]; obs = env.reset()

    def act():
        with th.no_grad():
            return th.clamp(model.policy._predict(norm.normalize_obs(obs), deterministic=True), -1, 1)
    env.force_scale = th.zeros(n, device=dev)               # settle (no force)
    for _ in range(settle):
        obs, *_ = env.step_tensor(th.cat([act(), dstb], dim=1))
    p0 = robot.data.root_link_pos_w[:, :2].clone()
    env.force_scale = th.ones(n, device=dev)                # 50 N step
    drift, tilt = [], []
    for _ in range(horizon):
        obs, *_ = env.step_tensor(th.cat([act(), dstb], dim=1))
        drift.append(float((robot.data.root_link_pos_w[:, 1] - p0[:, 1]).mean()))
        tilt.append(float((1 - (-robot.data.projected_gravity_b[:, 2])).mean()))
    env.close()
    return drift, tilt


if __name__ == "__main__":
    dev = "cuda:0"
    print("\nE018 pre-check: oracle first-0.3s impulse response (50 N step from quiet stance).")
    print(f"{'payload':16} | {'drift@0.1/0.2/0.3s (m)':>26} | {'tilt@0.1/0.2/0.3s':>22}")
    res = {}
    for task, label in [("go2_payload_conditioned_light_rigid", "light-rigid"),
                        ("go2_payload_conditioned_heavy_sloshy", "heavy-sloshy")]:
        d, t = impulse(task, dev); res[label] = (d, t)
        idx = [4, 9, 14]  # 0.1/0.2/0.3 s @ 50 Hz
        print(f"{label:16} | {'/'.join(f'{d[i]:5.2f}' for i in idx):>26} | {'/'.join(f'{t[i]:5.3f}' for i in idx):>22}")
    dl, tl = res["light-rigid"]; dh, th_ = res["heavy-sloshy"]
    diff03 = abs(dl[14] - dh[14])
    print(f"\nfirst-0.3s DRIFT difference light vs heavy = {diff03:.2f} m")
    print(">0.3m: payload type changes the early response -> pre-knowledge helps -> probe VIABLE.")
    print("~0: impulse response type-insensitive -> redesign impulse (direction/force) before Leg A.")
