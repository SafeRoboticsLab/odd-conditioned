"""E017 strategy read-out — 3-way (CONDITIONED raw-theta vs BLIND vs HISTORY), across in-dist AND OOD.

Metric: drift TOWARD pull (m) at force 0.35 = the strategy signal. DODGE = large drift (hop toward pull);
BRACE = ~0 (resist). Answers two questions at once:
  (1) IN-DIST (light_rigid, heavy_sloshy): does HISTORY show the dodge<->brace split like the oracle
      (=> "best of both worlds": OOD-robust like blind AND adaptive like conditioned)?
  (2) OOD (12kg sloshy/rigid): does HISTORY apply the RIGHT strategy (brace, low drift) or just survive
      with a generic/wrong one (silent failure)? Compared against what conditioned/blind do.
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
FR = float(sys.argv[1]) if len(sys.argv) > 1 else 0.35

ARMS = {
  "CONDITIONED": (f"{RUNS}/go2_payload_conditioned_gameplaysac",
    {"light_rigid": "go2_payload_conditioned_light_rigid", "heavy_sloshy": "go2_payload_conditioned_heavy_sloshy",
     "ood_sloshy": "go2_payload_conditioned_ood_sloshy", "ood_rigid": "go2_payload_conditioned_ood_rigid"}),
  "BLIND": (f"{RUNS}/go2_payload_blind_gameplaysac",
    {"light_rigid": "go2_payload_light_rigid", "heavy_sloshy": "go2_payload_heavy_sloshy",
     "ood_sloshy": "go2_payload_ood_sloshy", "ood_rigid": "go2_payload_ood_rigid"}),
  "HISTORY": (f"{RUNS}/go2_payload_history_gameplaysac",
    {"light_rigid": "go2_payload_history_light_rigid", "heavy_sloshy": "go2_payload_history_heavy_sloshy",
     "ood_sloshy": "go2_payload_history_ood_sloshy", "ood_rigid": "go2_payload_history_ood_rigid"}),
}
CONDS = [("light_rigid", "IN-DIST light-rigid (expect DODGE)"), ("heavy_sloshy", "IN-DIST heavy-sloshy (expect BRACE)"),
         ("ood_sloshy", "OOD 12kg sloshy (expect BRACE)"), ("ood_rigid", "OOD 12kg rigid (expect BRACE)")]


def drift(run_dir, task, dev, steps=220, nenv=128):
    env = make_tensor(task, nenv, dev, adversary=True)
    Algo = getattr(safety_sb3, algo_name(task, adversary=True).replace("PPO", "SAC"))
    model = Algo.load(f"{run_dir}/final_model.zip", env=env, device=dev,
                      custom_objects={"_use_lb": False, "_lb_dir": "/tmp/lb", "tensorboard_log": None, "buffer_size": 1})
    norm = TensorVecNormalize.load(f"{run_dir}/tensornormalize.pt", env); norm.training = False
    n = env.num_envs; dd = spec(task).dstb_dim
    env.force_scale = FR * th.ones(n, device=dev)
    dstb = (PULL / PULL.norm()).to(dev)[None].expand(n, dd).contiguous()
    robot = env.mj.scene["robot"]; obs = env.reset(); p0 = robot.data.root_link_pos_w[:, :2].clone()
    fell = th.zeros(n, dtype=th.bool, device=dev)
    for _ in range(steps):
        with th.no_grad():
            a = th.clamp(model.policy._predict(norm.normalize_obs(obs), deterministic=True), -1, 1)
        obs, g, dones, touts, l = env.step_tensor(th.cat([a, dstb], dim=1))
        fell |= (-robot.data.projected_gravity_b[:, 2] < 0.4)
    d = float((robot.data.root_link_pos_w[:, 1] - p0[:, 1]).mean())
    env.close()
    return d, float(fell.float().mean())


if __name__ == "__main__":
    import os.path as osp
    dev = "cuda:0"
    arms = [a for a in ARMS if osp.exists(f"{ARMS[a][0]}/final_model.zip")]
    print(f"\nE017 strategy — drift TOWARD pull (m) @ force {FR} ({FR*50:.0f}N). DODGE=large, BRACE=~0. (fell frac)")
    print(f"{'condition':36} | " + " | ".join(f"{a:>16}" for a in arms))
    print("-" * (38 + 19 * len(arms)))
    for key, label in CONDS:
        cells = []
        for a in arms:
            rd, tasks = ARMS[a]
            d, f = drift(rd, tasks[key], dev)
            cells.append(f"{d:5.2f}m (f{f:.2f})")
        print(f"{label:36} | " + " | ".join(f"{c:>16}" for c in cells))
    print("\nHISTORY 'best of both' = dodge in-dist-light, brace elsewhere; OOD brace (low drift) = correct, not silent-lucky.")
