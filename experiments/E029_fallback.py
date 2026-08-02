"""E029 — which FALLBACK is safest when B_hat=empty (theta unknown/OOD)? The old "fall back to robust blind"
rested on the BROKEN E019 metric (blind 0.99-1.00 to 25kg). Under the corrected falls/env-sec metric we re-ask
it, and add the standard practice option: a HARDCODED stance-hold (command the nominal standing pose, action=0,
PD holds it -- no policy, no theta). Compare STANCE vs BLIND vs HISTORY across static in-dist, OOD, and dynamic
ODD. A good fallback must be theta-AGNOSTIC and as-safe-as-possible without theta; this tells us which one.
"""
import os, sys, math
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco, torch as th
sys.path.insert(0, "external/robot-safety-sandbox")
import safety_sb3
from robot_safety_sandbox import make_tensor, spec, algo_name
from safety_sb3.tensor_env import TensorVecNormalize

DEV = "cuda:0"; PULL = th.tensor([0., 1., 0.]); DT = 0.02; NENV = 128; STEPS = 300
RUNS = {"BLIND": "results/go2_payload_runs/go2_payload_blind_gameplaysac",
        "HISTORY": "results/go2_payload_runs/go2_payload_history_gameplaysac"}
T_JUMP = 150


def worst_k(t):   # 0.75 Hz full rigid<->sloshy swing, after a rigid warmup (E025/E026 worst)
    return 300.0 if t < 60 else 150.0 + 150.0 * math.cos(2 * math.pi * 0.75 * (t - 60) * DT)


def jump_k(t):
    return 0.0 if t >= T_JUMP else 300.0


def jidx(env):
    m = env.mj.sim.mj_model
    js = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"robot/payload_j{i}") for i in range(1, 5)]
    return th.tensor([j for j in js if j >= 0], device=DEV, dtype=th.long)


def falls(arm, task, fr, kfn=None):
    env = make_tensor(task, NENV, DEV, adversary=True)
    dd = spec(task).dstb_dim; env.force_scale = fr * th.ones(NENV, device=DEV)
    dstb = (PULL / PULL.norm()).to(DEV)[None].expand(NENV, dd).contiguous()
    ji = jidx(env) if kfn else None
    m = nm = None
    if arm != "STANCE":
        Algo = getattr(safety_sb3, algo_name(task, adversary=True).replace("PPO", "SAC"))
        m = Algo.load(f"{RUNS[arm]}/final_model.zip", env=env, device=DEV,
                      custom_objects={"_use_lb": False, "_lb_dir": "/tmp/lb", "tensorboard_log": None, "buffer_size": 1})
        nm = TensorVecNormalize.load(f"{RUNS[arm]}/tensornormalize.pt", env); nm.training = False
    obs = env.reset(); nfail = 0
    for t in range(STEPS):
        if kfn is not None: env.mj.sim.model.jnt_stiffness[:, ji] = kfn(t)
        if arm == "STANCE":
            a = th.zeros(NENV, 12, device=DEV)            # nominal stance: PD holds default joint pose
        else:
            with th.no_grad():
                a = th.clamp(m.policy._predict(nm.normalize_obs(obs), deterministic=True), -1, 1)
        obs, g, dones, touts, l = env.step_tensor(th.cat([a, dstb], dim=1))
        nfail += int((dones & (~touts)).sum())
    env.close()
    return nfail / (NENV * STEPS * DT)


# (label, base-payload, force, kfn)  -- STANCE/BLIND use go2_payload_<base>, HISTORY uses go2_payload_history_<base>
SCEN = [("light_rigid  0N (sanity)", "light_rigid", 0.0, None),
        ("light_rigid  50N", "light_rigid", 1.0, None),
        ("heavy_sloshy 50N", "heavy_sloshy", 1.0, None),
        ("OOD 12kg sloshy 50N", "ood_sloshy", 1.0, None),
        ("OOD 12kg rigid 50N", "ood_rigid", 1.0, None),
        ("DYN jump r->s 50N", "heavy_sloshy", 1.0, jump_k),
        ("DYN worst 0.75Hz 50N", "heavy_sloshy", 1.0, worst_k)]
ARMS = ["STANCE", "BLIND", "HISTORY"]


def task_for(arm, base):
    return f"go2_payload_history_{base}" if arm == "HISTORY" else f"go2_payload_{base}"


if __name__ == "__main__":
    print(f"\nE029 FALLBACK comparison — falls/env-sec (lower=safer), corrected metric, {NENV} envs x {STEPS*DT:.0f}s.")
    print(f"{'scenario':26} | " + " | ".join(f"{a:>9}" for a in ARMS) + " | best")
    for label, base, fr, kfn in SCEN:
        cells = {a: falls(a, task_for(a, base), fr, kfn) for a in ARMS}
        best = min(cells, key=cells.get)
        print(f"{label:26} | " + " | ".join(f"{cells[a]:9.2f}" for a in ARMS) + f" | {best}")
    print("\nA good empty-set fallback is the theta-agnostic arm safest in the OOD/DYN rows. Stance-hold = the")
    print("hardcoded baseline; if a learned arm doesn't beat it there, the hardcoded stance IS the fallback.")
