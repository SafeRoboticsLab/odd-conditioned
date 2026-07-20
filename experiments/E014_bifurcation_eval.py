"""E014 bifurcation eval — do the two specialists flip strategy (dodge vs brace) under a lateral pull?

Loads each specialist's final GameplaySAC model + its obs-normalizer, applies a FIXED lateral pull at
force ratio 0.35 (= 0.35 * force_max 50 = 17.5 N, same for both — a controlled, comparable probe), and
measures the CONTROL policy's response. Dodge = base drifts TOWARD the pull (turn+hop to cancel it);
brace = base stays PUT (plant and resist). Also a no-force baseline.
"""
import os, sys, argparse
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, torch as th
sys.path.insert(0, "external/robot-safety-sandbox")
import safety_sb3
from robot_safety_sandbox import make_tensor, spec, algo_name
from safety_sb3.tensor_env import TensorVecNormalize

RUNS = "results/go2_payload_runs"
PULL_DIR = th.tensor([0.0, 1.0, 0.0])   # +y lateral (robot's side)

def load(task, device):
    d = f"{RUNS}/{task}_gameplaysac"
    env = make_tensor(task, 64, device, adversary=True)
    sac_name = algo_name(task, adversary=True).replace("PPO", "SAC")   # GameplayPPO -> GameplaySAC (SAC ckpt)
    Algo = getattr(safety_sb3, sac_name)
    model = Algo.load(f"{d}/final_model.zip", env=env, device=device,
                      custom_objects={"_use_lb": False, "_lb_dir": "/tmp/lb", "tensorboard_log": None,
                                      "buffer_size": 1})   # eval only -> no big replay buffer
    norm = TensorVecNormalize.load(f"{d}/tensornormalize.pt", env); norm.training = False
    return env, model, norm

def rollout(env, model, norm, device, force_ratio, steps=250):
    n = env.num_envs; s = spec(env_task); dd = s.dstb_dim
    env.force_scale = force_ratio * th.ones(n, device=device)     # 0.35 -> 17.5 N
    dstb = (PULL_DIR / PULL_DIR.norm()).to(device)[None].expand(n, dd).contiguous()
    robot = env.mj.scene["robot"]
    obs = env.reset()
    p0 = robot.data.root_link_pos_w[:, :2].clone()
    disp_y = []; tilt = []; reached = 0; coll = 0
    for k in range(steps):
        with th.no_grad():
            a = th.clamp(model.policy._predict(norm.normalize_obs(obs), deterministic=True), -1, 1)
        obs, g, dones, touts, l = env.step_tensor(th.cat([a, dstb], dim=1))
        rel = robot.data.root_link_pos_w[:, :2] - p0
        up = -robot.data.projected_gravity_b[:, 2]              # 1=upright
        disp_y.append(rel[:, 1].mean().item()); tilt.append((1 - up).mean().item())
    rel = robot.data.root_link_pos_w[:, :2] - p0
    return dict(drift_pull=float(rel[:, 1].mean()), drift_lat=float(rel[:, 0].abs().mean()),
                mean_tilt=float(np.mean(tilt)), fell=float((-robot.data.projected_gravity_b[:,2] < 0.5).float().mean()))

if __name__ == "__main__":
    dev = "cuda:0"
    print(f"force ratio 0.35 = {0.35*50:.1f} N lateral (+y) pull\n")
    print(f"{'specialist':14} {'force':>6} | {'drift_TOWARD_pull(m)':>20} {'drift_lateral(m)':>16} {'mean_tilt':>9} {'fell_frac':>9}")
    for task in ("go2_payload_light_rigid", "go2_payload_heavy_sloshy"):
        global env_task; env_task = task
        env, model, norm = load(task, dev)
        for fr in (0.0, 0.35):
            r = rollout(env, model, norm, dev, fr)
            tag = task.replace("go2_payload_", "")
            print(f"{tag:14} {fr:6.2f} | {r['drift_pull']:20.3f} {r['drift_lat']:16.3f} {r['mean_tilt']:9.3f} {r['fell']:9.2f}")
        env.close()
    print("\nDODGE = large drift TOWARD pull; BRACE = drift ~0 (stays put).")
