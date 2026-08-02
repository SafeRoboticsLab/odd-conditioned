"""E023 — sweep the *type* of mid-episode ODD change to find which is hardest for a reactive policy.
Classical-control hypotheses being tested on the payload rigidity k(t) (300=rigid .. 0=sloshy):
  - STEP (instantaneous jump): shocks a reactive controller (bandwidth-exceeding change).
  - RAMP at varying rate: quasi-static (slow) should be survivable; a *critical* rate may be worst
    (rate-induced tipping / R-tipping — thrown off the stable branch even though every frozen k is stable).
  - SINUSOIDAL parametric forcing at varying period: a resonant period may destabilize (parametric resonance
    / Mathieu). Slow osc ~ quasi-static; fast osc ~ averaged; a mid band should bite hardest.
Controls: static_sloshy (sustained k=0, no dynamics) and static_rigid (k=300). Metric = falls/env-sec over
the active window [T0, end] (reset-independent g<0 terminations). Constant 50 N pull. Lower = safer.
Read: for each shape, BLIND fall rate (absolute severity) and BLIND-minus-HISTORY (does tracking help there).
"""
import os, sys, math
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco, torch as th
sys.path.insert(0, "external/robot-safety-sandbox")
import safety_sb3
from robot_safety_sandbox import make_tensor, spec, algo_name
from safety_sb3.tensor_env import TensorVecNormalize

R = "results/go2_payload_runs"; DEV = "cuda:0"; PULL = th.tensor([0., 1., 0.]); FR = 1.0
STEPS = 360; T0 = 60; NENV = 128; DT = 0.02
ARMS = {"CONDITIONED": (f"{R}/go2_payload_conditioned_gameplaysac", "go2_payload_conditioned_heavy_sloshy"),
        "BLIND": (f"{R}/go2_payload_blind_gameplaysac", "go2_payload_heavy_sloshy"),
        "HISTORY": (f"{R}/go2_payload_history_gameplaysac", "go2_payload_history_heavy_sloshy")}
# (name, kind, param)   param = ramp duration (steps) or osc period (steps)
SCHEDULES = [("static_rigid", "static", 300.0), ("static_sloshy", "static", 0.0),
             ("step", "step", 0), ("ramp_0.2s", "ramp", 10), ("ramp_0.8s", "ramp", 40),
             ("ramp_1.6s", "ramp", 80), ("ramp_3.0s", "ramp", 150),
             ("osc_0.4s", "osc", 20), ("osc_0.8s", "osc", 40), ("osc_1.6s", "osc", 80), ("osc_3.0s", "osc", 150)]


def k_of(kind, p, t):
    if kind == "static": return p
    if t < T0: return 300.0
    s = t - T0
    if kind == "step": return 0.0
    if kind == "ramp": return max(0.0, 300.0 * (1.0 - s / p))
    if kind == "osc": return 150.0 + 150.0 * math.cos(2 * math.pi * s / p)  # starts rigid, oscillates rigid<->sloshy
    return 0.0


def jidx(env):
    m = env.mj.sim.mj_model
    js = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"robot/payload_j{i}") for i in range(1, 5)]
    return th.tensor([j for j in js if j >= 0], device=DEV, dtype=th.long)


def fall_rate(run_dir, task, kind, p):
    env = make_tensor(task, NENV, DEV, adversary=True)
    Algo = getattr(safety_sb3, algo_name(task, adversary=True).replace("PPO", "SAC"))
    m = Algo.load(f"{run_dir}/final_model.zip", env=env, device=DEV,
                  custom_objects={"_use_lb": False, "_lb_dir": "/tmp/lb", "tensorboard_log": None, "buffer_size": 1})
    nm = TensorVecNormalize.load(f"{run_dir}/tensornormalize.pt", env); nm.training = False
    dd = spec(task).dstb_dim; env.force_scale = FR * th.ones(NENV, device=DEV)
    dstb = (PULL / PULL.norm()).to(DEV)[None].expand(NENV, dd).contiguous()
    ji = jidx(env); obs = env.reset(); nfail = 0
    for t in range(STEPS):
        env.mj.sim.model.jnt_stiffness[:, ji] = k_of(kind, p, t)
        with th.no_grad():
            a = th.clamp(m.policy._predict(nm.normalize_obs(obs), deterministic=True), -1, 1)
        obs, g, dones, touts, l = env.step_tensor(th.cat([a, dstb], dim=1))
        if t >= T0: nfail += int((dones & (~touts)).sum())
    env.close()
    return nfail / (NENV * (STEPS - T0) * DT)


if __name__ == "__main__":
    arms = list(ARMS)
    print(f"\nE023 dynamic-ODD SHAPE sweep — falls/env-sec over active window [{T0*DT:.1f}s,{STEPS*DT:.1f}s], {FR*50:.0f}N pull.")
    print(f"{'change type':14} | " + " | ".join(f"{a:>12}" for a in arms) + " | BLIND-HIST")
    rows = []
    for name, kind, p in SCHEDULES:
        cells = {a: fall_rate(ARMS[a][0], ARMS[a][1], kind, p) for a in arms}
        gap = cells["BLIND"] - cells["HISTORY"]
        rows.append((name, cells, gap))
        print(f"{name:14} | " + " | ".join(f"{cells[a]:12.2f}" for a in arms) + f" | {gap:+.2f}")
    worst_abs = max(rows, key=lambda r: r[1]["BLIND"])
    worst_gap = max(rows, key=lambda r: r[2])
    print(f"\nHardest for BLIND (absolute): {worst_abs[0]} @ {worst_abs[1]['BLIND']:.2f}")
    print(f"Largest BLIND-HISTORY gap (tracking helps most): {worst_gap[0]} @ {worst_gap[2]:+.2f}")
    print("Compare ramp rates: if a MID rate > step -> rate-induced tipping. If an osc period spikes -> parametric resonance.")
