"""E036 — CHECKPOINT DIAGNOSTIC. The HISTORY arm's dynamic-ODD number (~50%) was measured on final_model.zip,
which the training log shows is a 0.42 in-distribution TROUGH of an unstable run (36M hit 1.00; 42-48M held
0.92-0.97; 50M final collapsed to 0.42). The good policies survive only as leaderboard actor snapshots
(ctrl_42M=0.969, ctrl_44M=0.961, ctrl_48M=0.945). Here we graft those actor weights into the loaded model and
re-run the E021 dynamic-ODD eval (rigidity JUMP rigid->sloshy, reset-safe falls/env-sec) to separate two
hypotheses: (H1) bad-checkpoint artifact -> a good ckpt jumps to low fall rate; (H2) train/test distribution
mismatch (trained on static per-episode theta, tested within-episode) -> even the good ckpt stays high.
Normalizer: the 42-48M steps did not snapshot tensornorm, so we reuse the final tensornormalize.pt (obs stats
are converged by 48M; negligible for a diagnostic). BLIND-final is the reference (its final ckpt is healthy).
"""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco, torch as th
sys.path.insert(0, "external/robot-safety-sandbox")
import safety_sb3
from robot_safety_sandbox import make_tensor, spec, algo_name
from safety_sb3.tensor_env import TensorVecNormalize

R = "results/go2_payload_runs"; DEV = "cuda:0"; PULL = th.tensor([0., 1., 0.]); FR = 1.0
STEPS = 300; T_JUMP = 150; TRANSW = 50
HRUN = f"{R}/go2_payload_history_gameplaysac"; HTASK = "go2_payload_history_heavy_sloshy"
BRUN = f"{R}/go2_payload_blind_gameplaysac"; BTASK = "go2_payload_heavy_sloshy"


def jidx(env):
    m = env.mj.sim.mj_model
    js = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"robot/payload_j{i}") for i in range(1, 5)]
    return th.tensor([j for j in js if j >= 0], device=DEV, dtype=th.long)


def run(run_dir, task, graft=None, dynamic=True):
    env = make_tensor(task, 128, DEV, adversary=True)
    Algo = getattr(safety_sb3, algo_name(task, adversary=True).replace("PPO", "SAC"))
    m = Algo.load(f"{run_dir}/final_model.zip", env=env, device=DEV,
                  custom_objects={"_use_lb": False, "_lb_dir": "/tmp/lb", "tensorboard_log": None, "buffer_size": 1})
    if graft is not None:
        sd = th.load(graft, map_location=DEV, weights_only=False)
        miss, unexp = m.policy.actor.load_state_dict(sd, strict=False)
        assert not [k for k in miss if "latent_pi" in k or k.startswith(("mu.", "log_std."))], f"graft miss {miss}"
    nm = TensorVecNormalize.load(f"{run_dir}/tensornormalize.pt", env); nm.training = False
    n = 128; dd = spec(task).dstb_dim; env.force_scale = FR * th.ones(n, device=DEV)
    dstb = (PULL / PULL.norm()).to(DEV)[None].expand(n, dd).contiguous()
    ji = jidx(env); obs = env.reset(); fails = [0, 0, 0]
    for t in range(STEPS):
        k = 0.0 if (not dynamic or t >= T_JUMP) else 300.0
        env.mj.sim.model.jnt_stiffness[:, ji] = k
        with th.no_grad():
            a = th.clamp(m.policy._predict(nm.normalize_obs(obs), deterministic=True), -1, 1)
        obs, g, dones, touts, l = env.step_tensor(th.cat([a, dstb], dim=1))
        nf = int((dones & (~touts)).sum())
        if t < T_JUMP: fails[0] += nf
        elif t < T_JUMP + TRANSW: fails[1] += nf
        else: fails[2] += nf
    env.close()
    return (fails[0] / (n * T_JUMP * 0.02), fails[1] / (n * TRANSW * 0.02),
            fails[2] / (n * (STEPS - T_JUMP - TRANSW) * 0.02))


if __name__ == "__main__":
    LB = f"{HRUN}/leaderboard"
    ARMS = [("BLIND-final", BRUN, BTASK, None),
            ("HIST-final(0.42)", HRUN, HTASK, None),
            ("HIST-ctrl48M(0.94)", HRUN, HTASK, f"{LB}/ctrl_48000000.pt"),
            ("HIST-ctrl42M(0.97)", HRUN, HTASK, f"{LB}/ctrl_42000384.pt")]
    print(f"\nE036 CKPT DIAG — dynamic ODD (rigidity JUMP @ {T_JUMP*0.02:.1f}s, {FR*50:.0f}N pull). falls/env-sec.")
    print(f"{'arm':20} | {'RIGID':>7} | {'TRANSIENT':>10} | {'SLOSHY':>7}")
    for name, rd, task, graft in ARMS:
        rr, tr, sr = run(rd, task, graft=graft, dynamic=True)
        print(f"{name:20} | {rr:7.2f} | {tr:10.2f} | {sr:7.2f}")
    print("\nH1 (bad ckpt): ctrl48/42M TRANSIENT << HIST-final. H2 (dist mismatch): ctrl48/42M stays high.")
