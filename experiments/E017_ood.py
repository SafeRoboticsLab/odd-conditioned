"""E017-OOD — 3-way out-of-distribution generalization: CONDITIONED (raw theta) vs BLIND vs HISTORY.

Hypothesis (Buzi): a raw EXACT theta-input is OOD-fragile — an extrapolated theta (mass 12kg->+2.4,
stiffness 400->+1.7) induces the wrong strategy and the conditioned policy FALLS (safe 0.18-0.32). A
HISTORY-conditioned policy (frame-stacked proprio+action, no theta) infers the payload from the dynamics
it FEELS, so it should degrade GRACEFULLY OOD (no extrapolated input; strategy-relevant implicit latent).
The BLIND (no theta, no history) was already OOD-robust (safe 1.00) because it just reacts to felt physics.

OOD physics (OUTSIDE training mass<=7.5kg, stiffness<=300): heavy-SLOSHY 12kg k=0; heavy-RIGID 12kg k=400.
Metric: safe_rate = frac of envs that NEVER fell; at force 0.0 (stand under 12kg?) and 1.0 (50N pull).
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

# arm -> (run_dir, {ood_key: eval_task})
ARMS = {
    "CONDITIONED": (f"{RUNS}/go2_payload_conditioned_gameplaysac",
                    {"sloshy": "go2_payload_conditioned_ood_sloshy", "rigid": "go2_payload_conditioned_ood_rigid"}),
    "BLIND":       (f"{RUNS}/go2_payload_blind_gameplaysac",
                    {"sloshy": "go2_payload_ood_sloshy", "rigid": "go2_payload_ood_rigid"}),
    "HISTORY":     (f"{RUNS}/go2_payload_history_gameplaysac",
                    {"sloshy": "go2_payload_history_ood_sloshy", "rigid": "go2_payload_history_ood_rigid"}),
}
OOD_LABELS = {"sloshy": "heavy-SLOSHY 12kg k=0", "rigid": "heavy-RIGID 12kg k=400"}


def safe_rate(run_dir, task, dev, fr, steps=220, nenv=128):
    env = make_tensor(task, nenv, dev, adversary=True)
    Algo = getattr(safety_sb3, algo_name(task, adversary=True).replace("PPO", "SAC"))
    model = Algo.load(f"{run_dir}/final_model.zip", env=env, device=dev,
                      custom_objects={"_use_lb": False, "_lb_dir": "/tmp/lb",
                                      "tensorboard_log": None, "buffer_size": 1})
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
    import os.path as osp
    dev = "cuda:0"
    arms = [a for a in ARMS if osp.exists(f"{ARMS[a][0]}/final_model.zip")]
    print(f"E017-OOD safe_rate (frac never fell). arms present: {arms}")
    for fr in (0.0, 1.0):
        print(f"\n=== force {fr} ({fr*50:.0f} N pull) ===")
        print(f"{'OOD payload':26} | " + " | ".join(f"{a:>12}" for a in arms))
        for k, label in OOD_LABELS.items():
            row = []
            for a in arms:
                rd, tasks = ARMS[a]
                row.append(f"{safe_rate(rd, tasks[k], dev, fr):.2f}")
            print(f"{label:26} | " + " | ".join(f"{v:>12}" for v in row))
    print("\n>1 arm safe under OOD = generalizes. HISTORY expected to degrade gracefully where CONDITIONED (raw theta) fell.")
