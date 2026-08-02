"""E021 — DYNAMIC ODD (the true test): the payload changes WITHIN the episode. A static ODD lets a reactive
policy converge (blind ~ everyone); a mid-episode JUMP is a commitment window — does an arm that TRACKS the
change (history) or is TOLD it (conditioned, live theta obs) beat the blind reactive policy on the transient?

Scenario: RIGIDITY jump. Payload starts RIGID (jnt_stiffness=300), then at t=T_JUMP snaps to SLOSHY (0) —
a 7kg top-heavy load that suddenly starts sloshing (a shifting/liquefying cargo). jnt_stiffness overridden
live every step (mass built correctly at 7kg -> no inertia issue). Fall rate (g<0, reset-independent) is
reported in the RIGID phase, the TRANSIENT window right after the jump, and the SLOSHY phase.
Also a static-sloshy control (k=0 throughout) to isolate the transient cost of the jump.
"""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco, torch as th
sys.path.insert(0, "external/robot-safety-sandbox")
import safety_sb3
from robot_safety_sandbox import make_tensor, spec, algo_name
from safety_sb3.tensor_env import TensorVecNormalize

R = "results/go2_payload_runs"; DEV = "cuda:0"; PULL = th.tensor([0., 1., 0.]); FR = 1.0
STEPS = 300; T_JUMP = 150; TRANSW = 50   # jump at 3.0s; transient window = 1.0s after
ARMS = {"CONDITIONED": (f"{R}/go2_payload_conditioned_gameplaysac", "go2_payload_conditioned_heavy_sloshy"),
        "BLIND": (f"{R}/go2_payload_blind_gameplaysac", "go2_payload_heavy_sloshy"),
        "HISTORY": (f"{R}/go2_payload_history_gameplaysac", "go2_payload_history_heavy_sloshy")}


def jidx(env):
    m = env.mj.sim.mj_model
    js = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"robot/payload_j{i}") for i in range(1, 5)]
    return th.tensor([j for j in js if j >= 0], device=DEV, dtype=th.long)


def run(run_dir, task, dynamic=True):
    env = make_tensor(task, 128, DEV, adversary=True)
    Algo = getattr(safety_sb3, algo_name(task, adversary=True).replace("PPO", "SAC"))
    m = Algo.load(f"{run_dir}/final_model.zip", env=env, device=DEV,
                  custom_objects={"_use_lb": False, "_lb_dir": "/tmp/lb", "tensorboard_log": None, "buffer_size": 1})
    nm = TensorVecNormalize.load(f"{run_dir}/tensornormalize.pt", env); nm.training = False
    n = 128; dd = spec(task).dstb_dim; env.force_scale = FR * th.ones(n, device=DEV)
    dstb = (PULL / PULL.norm()).to(DEV)[None].expand(n, dd).contiguous()
    ji = jidx(env); obs = env.reset()
    fails = [0, 0, 0]  # rigid-phase, transient, sloshy-phase
    for t in range(STEPS):
        k = 0.0 if (not dynamic or t >= T_JUMP) else 300.0   # dynamic: rigid then sloshy; control: always sloshy
        env.mj.sim.model.jnt_stiffness[:, ji] = k
        with th.no_grad():
            a = th.clamp(m.policy._predict(nm.normalize_obs(obs), deterministic=True), -1, 1)
        obs, g, dones, touts, l = env.step_tensor(th.cat([a, dstb], dim=1))
        nf = int((dones & (~touts)).sum())
        if t < T_JUMP: fails[0] += nf
        elif t < T_JUMP + TRANSW: fails[1] += nf
        else: fails[2] += nf
    env.close()
    rr = fails[0] / (n * T_JUMP * 0.02)
    tr = fails[1] / (n * TRANSW * 0.02)
    sr = fails[2] / (n * (STEPS - T_JUMP - TRANSW) * 0.02)
    return rr, tr, sr


if __name__ == "__main__":
    print(f"\nE021 DYNAMIC ODD — rigidity JUMP rigid->sloshy @ t={T_JUMP*0.02:.1f}s, {FR*50:.0f}N pull. falls/env-sec.")
    print(f"{'arm':12} | {'RIGID phase':>12} | {'TRANSIENT(jump)':>16} | {'SLOSHY phase':>13} | {'ctrl sloshy@transient':>22}")
    for a, (rd, task) in ARMS.items():
        rr, tr, sr = run(rd, task, dynamic=True)
        _, tc, _ = run(rd, task, dynamic=False)   # static-sloshy control: transient-window fall rate
        print(f"{a:12} | {rr:12.2f} | {tr:16.2f} | {sr:13.2f} | {tc:22.2f}")
    print("\nTRANSIENT >> ctrl-sloshy => the JUMP itself (dynamic ODD) causes falls a reactive policy can't pre-empt.")
    print("If HISTORY/COND transient < BLIND transient => tracking/knowing the change helps on dynamic ODD.")
