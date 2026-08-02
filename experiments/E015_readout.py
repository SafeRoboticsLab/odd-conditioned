"""E015 read-out — the conditioned-vs-blind payoff (E008c analog on Go2).

ONE conditioned policy (sees theta) and ONE blind policy (does not), both trained on the ODD-randomized
env (rigidity x total-mass, per-EPISODE). Question: does the conditioned policy match BOTH specialists'
strategies — DODGE at light-rigid theta, BRACE at heavy-sloshy theta — while the blind policy is stuck on
a single worst-case compromise at both?

Each policy is evaluated at the two fixed-ODD extremes under the SAME lateral pull (force ratio 0.35):
  - conditioned (57-dim obs): on go2_payload_conditioned_{light_rigid,heavy_sloshy} (specialist physics
    + the theta obs term, so theta is read live and correct for that ODD).
  - blind (55-dim obs): on the plain specialists go2_payload_{light_rigid,heavy_sloshy}.
Metric: drift TOWARD the pull (m). DODGE = large drift (hops toward the pull to cancel it); BRACE = ~0.
A large conditioned gap + a small blind gap = conditioning recovers the whole per-ODD strategy family.
"""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, torch as th
sys.path.insert(0, "external/robot-safety-sandbox")
import safety_sb3
from robot_safety_sandbox import make_tensor, spec, algo_name
from safety_sb3.tensor_env import TensorVecNormalize

RUNS = "results/go2_payload_runs"
PULL = th.tensor([0.0, 1.0, 0.0])          # +y lateral
FR = float(sys.argv[1]) if len(sys.argv) > 1 else 0.35


def load_eval(run_dir, eval_task, dev, steps=250):
    env = make_tensor(eval_task, 64, dev, adversary=True)
    Algo = getattr(safety_sb3, algo_name(eval_task, adversary=True).replace("PPO", "SAC"))
    model = Algo.load(f"{run_dir}/final_model.zip", env=env, device=dev,
                      custom_objects={"_use_lb": False, "_lb_dir": "/tmp/lb",
                                      "tensorboard_log": None, "buffer_size": 1})
    norm = TensorVecNormalize.load(f"{run_dir}/tensornormalize.pt", env); norm.training = False
    n = env.num_envs; dd = spec(eval_task).dstb_dim
    env.force_scale = FR * th.ones(n, device=dev)
    dstb = (PULL / PULL.norm()).to(dev)[None].expand(n, dd).contiguous()
    robot = env.mj.scene["robot"]
    obs = env.reset(); p0 = robot.data.root_link_pos_w[:, :2].clone()
    for _ in range(steps):
        with th.no_grad():
            a = th.clamp(model.policy._predict(norm.normalize_obs(obs), deterministic=True), -1, 1)
        obs, g, dones, touts, l = env.step_tensor(th.cat([a, dstb], dim=1))
    rel = robot.data.root_link_pos_w[:, :2] - p0
    drift = float(rel[:, 1].mean())
    fell = float((-robot.data.projected_gravity_b[:, 2] < 0.5).float().mean())
    env.close()
    return drift, fell


if __name__ == "__main__":
    dev = "cuda:0"
    rows = [("CONDITIONED", f"{RUNS}/go2_payload_conditioned_gameplaysac",
             "go2_payload_conditioned_light_rigid", "go2_payload_conditioned_heavy_sloshy"),
            ("BLIND", f"{RUNS}/go2_payload_blind_gameplaysac",
             "go2_payload_light_rigid", "go2_payload_heavy_sloshy")]
    print(f"\nE015 read-out — force ratio {FR} = {FR*50:.1f} N +y pull\n"
          f"drift TOWARD pull (m): DODGE = large, BRACE = ~0\n")
    print(f"{'policy':12} | {'light_rigid theta':>18} | {'heavy_sloshy theta':>18} | {'|gap|':>6}")
    print("-" * 66)
    for name, rd, lr, hs in rows:
        dlr, flr = load_eval(rd, lr, dev)
        dhs, fhs = load_eval(rd, hs, dev)
        print(f"{name:12} | {dlr:8.2f}m (fell {flr:.2f}) | {dhs:8.2f}m (fell {fhs:.2f}) | {abs(dlr-dhs):5.2f}")
    print("\nEXPECT: CONDITIONED gap LARGE (dodge@light-rigid, brace@heavy-sloshy = adapts);\n"
          "        BLIND gap SMALL (one strategy at both = cannot adapt).")
