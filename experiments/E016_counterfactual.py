"""E016 counterfactual — does the conditioned policy CONDITION on theta, or just ride the physics?

The E015 read-out showed the conditioned policy dodges @ light-rigid and braces @ heavy-sloshy when it
sees the CORRECT theta. But that gap could be the physics alone (different payload => different drift)
rather than the policy using its theta-input. The clean causal test (professor's E016): hold the PHYSICS
fixed, feed the policy the WRONG theta, and see if the strategy flips.

theta is the last 2 obs dims (payload_odd appended last). We overwrite obs[:, -2:] with a chosen
(raw, [-1,1]) theta before normalization, so the policy's proprioception is real but its theta-belief is
whatever we inject. If drift changes with the injected theta (physics held fixed), the policy is CAUSALLY
conditioning on theta. If drift is ~flat across theta, it ignores the input (the E010 pathology).

  2x2: {light_rigid physics, heavy_sloshy physics} x {theta=light_rigid, theta=heavy_sloshy}
  diagonal = correct theta (reproduces E015: dodge 2.52 / brace 0.40); off-diagonal = wrong theta.
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
FR = float(sys.argv[1]) if len(sys.argv) > 1 else 0.35
# raw payload_odd theta values for each extreme: rigidity in [0,300]->[-1,1], mass in [1.2,7.5]->[-1,1]
THETA = {"light_rigid": th.tensor([1.0, -1.0]), "heavy_sloshy": th.tensor([-1.0, 0.841])}
PHYS = {"light_rigid": "go2_payload_conditioned_light_rigid",
        "heavy_sloshy": "go2_payload_conditioned_heavy_sloshy"}


def eval_cf(phys_task, theta_inject, dev, steps=250, nenv=256):
    env = make_tensor(phys_task, nenv, dev, adversary=True)
    Algo = getattr(safety_sb3, algo_name(phys_task, adversary=True).replace("PPO", "SAC"))
    model = Algo.load(f"{RUN}/final_model.zip", env=env, device=dev,
                      custom_objects={"_use_lb": False, "_lb_dir": "/tmp/lb",
                                      "tensorboard_log": None, "buffer_size": 1})
    norm = TensorVecNormalize.load(f"{RUN}/tensornormalize.pt", env); norm.training = False
    n = env.num_envs; dd = spec(phys_task).dstb_dim
    env.force_scale = FR * th.ones(n, device=dev)
    dstb = (PULL / PULL.norm()).to(dev)[None].expand(n, dd).contiguous()
    tinj = theta_inject.to(dev)[None].expand(n, 2).contiguous()
    robot = env.mj.scene["robot"]
    obs = env.reset(); p0 = robot.data.root_link_pos_w[:, :2].clone()
    for _ in range(steps):
        o = obs.clone(); o[:, -2:] = tinj                    # inject the (possibly wrong) theta belief
        with th.no_grad():
            a = th.clamp(model.policy._predict(norm.normalize_obs(o), deterministic=True), -1, 1)
        obs, g, dones, touts, l = env.step_tensor(th.cat([a, dstb], dim=1))
    rel = robot.data.root_link_pos_w[:, :2] - p0
    drift = float(rel[:, 1].mean()); sem = float(rel[:, 1].std() / (n ** 0.5))
    env.close()
    return drift, sem


if __name__ == "__main__":
    dev = "cuda:0"
    print(f"\nE016 counterfactual — force {FR} = {FR*50:.1f} N pull. drift TOWARD pull (m).")
    print(f"{'PHYSICS (rows) vs theta':26} | {'theta=light_rigid':>17} | {'theta=heavy_sloshy':>18}")
    print("-" * 70)
    res = {}
    for pk in ("light_rigid", "heavy_sloshy"):
        d_lr, s_lr = eval_cf(PHYS[pk], THETA["light_rigid"], dev)
        d_hs, s_hs = eval_cf(PHYS[pk], THETA["heavy_sloshy"], dev)
        res[pk] = (d_lr, s_lr, d_hs, s_hs)
        tag_lr = "correct" if pk == "light_rigid" else "WRONG"
        tag_hs = "correct" if pk == "heavy_sloshy" else "WRONG"
        print(f"{pk+' physics':26} | {d_lr:6.2f}+-{s_lr:.2f}m {tag_lr:7} | {d_hs:6.2f}+-{s_hs:.2f}m {tag_hs:7}")
    print("-" * 70)
    # causal signal: within each fixed physics, does swapping theta move drift beyond the error bars?
    for pk in ("light_rigid", "heavy_sloshy"):
        d_lr, s_lr, d_hs, s_hs = res[pk]
        eff = d_lr - d_hs; noise = (s_lr ** 2 + s_hs ** 2) ** 0.5
        verdict = "theta CAUSAL" if abs(eff) > 2 * noise else "within noise -> theta effect weak/none"
        print(f"{pk} physics: theta-swap effect (lr - hs) = {eff:+.2f} m  (noise ~{noise:.2f}) -> {verdict}")
