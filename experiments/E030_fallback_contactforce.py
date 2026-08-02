"""E030 — re-evaluate the fallback under a DIFFERENT safe set (the fallback's own ODD). Idea (Buzi): a
'lie-down-softly' fallback violates the STANDING margin (trunk-corner height >0.10m) by construction, but is
safe under a CONTACT-FORCE margin: ground contact on non-foot bodies is fine as long as it's gentle (<=10N),
so belly+legs resting softly = safe, a hard slam = unsafe. So the fallback's g_x measures non-foot ground
contact FORCE, not height.

Implementation: build the eval env with end_criterion='timeout' -> ALL failure terminations stripped, NO
auto-reset. The robot may sink / settle / collapse freely; we read the existing `nonfoot_ground_touch`
ContactSensor (all non-foot collision geoms vs terrain) each step and call it UNSAFE if force > 10N. We report
the unsafe fraction AND the actual force magnitudes (p50/p90/max), so we can see whether a settle is soft.
Compare STANCE-HOLD (a=0) vs BLIND vs HISTORY across static + dynamic. Contrast with the standing-margin verdict.
"""
import os, sys, math
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco, torch as th
sys.path.insert(0, "external/robot-safety-sandbox")
import safety_sb3
from robot_safety_sandbox import make_tensor, spec, algo_name
from safety_sb3.tensor_env import TensorVecNormalize

DEV = "cuda:0"; PULL = th.tensor([0., 1., 0.]); DT = 0.02; NENV = 128; STEPS = 300; THR = 10.0
RUNS = {"BLIND": "results/go2_payload_runs/go2_payload_blind_gameplaysac",
        "HISTORY": "results/go2_payload_runs/go2_payload_history_gameplaysac"}


def worst_k(t): return 300.0 if t < 60 else 150.0 + 150.0 * math.cos(2 * math.pi * 0.75 * (t - 60) * DT)
def jidx(env):
    m = env.mj.sim.mj_model
    js = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"robot/payload_j{i}") for i in range(1, 5)]
    return th.tensor([j for j in js if j >= 0], device=DEV, dtype=th.long)


def nonfoot_force(env):
    s = env.mj.scene["nonfoot_ground_touch"]; d = s.data
    fh = d.force_history if d.force_history is not None else d.force      # [B,N,H,3] or [B,N,3]
    return th.norm(fh, dim=-1).flatten(1).amax(dim=1)                    # [B] per-env max non-foot ground force


def evalf(arm, task, fr, kfn=None):
    env = make_tensor(task, NENV, DEV, adversary=True, end_criterion="timeout")   # no failure resets
    dd = spec(task).dstb_dim; env.force_scale = fr * th.ones(NENV, device=DEV)
    dstb = (PULL / PULL.norm()).to(DEV)[None].expand(NENV, dd).contiguous()
    ji = jidx(env) if kfn else None
    m = nm = None
    if arm != "STANCE":
        Algo = getattr(safety_sb3, algo_name(task, adversary=True).replace("PPO", "SAC"))
        m = Algo.load(f"{RUNS[arm]}/final_model.zip", env=env, device=DEV,
                      custom_objects={"_use_lb": False, "_lb_dir": "/tmp/lb", "tensorboard_log": None, "buffer_size": 1})
        nm = TensorVecNormalize.load(f"{RUNS[arm]}/tensornormalize.pt", env); nm.training = False
    obs = env.reset(); forces = []; unsafe_steps = 0; ever = th.zeros(NENV, dtype=th.bool, device=DEV)
    for t in range(STEPS):
        if kfn is not None: env.mj.sim.model.jnt_stiffness[:, ji] = kfn(t)
        if arm == "STANCE":
            a = th.zeros(NENV, 12, device=DEV)
        else:
            with th.no_grad():
                a = th.clamp(m.policy._predict(nm.normalize_obs(obs), deterministic=True), -1, 1)
        obs, g, dones, touts, l = env.step_tensor(th.cat([a, dstb], dim=1))
        f = nonfoot_force(env); forces.append(f)
        bad = f > THR; unsafe_steps += int(bad.sum()); ever |= bad
    env.close()
    F = th.cat(forces)
    return (unsafe_steps / (NENV * STEPS), float(ever.float().mean()),
            float(F.median()), float(th.quantile(F, 0.90)), float(F.max()))


SCEN = [("light_rigid  0N", "light_rigid", 0.0, None),
        ("light_rigid  50N", "light_rigid", 1.0, None),
        ("heavy_sloshy 50N", "heavy_sloshy", 1.0, None),
        ("DYN worst 0.75Hz 50N", "heavy_sloshy", 1.0, worst_k)]
ARMS = ["STANCE", "BLIND", "HISTORY"]


def task_for(arm, base): return f"go2_payload_history_{base}" if arm == "HISTORY" else f"go2_payload_{base}"


if __name__ == "__main__":
    print(f"\nE030 CONTACT-FORCE margin (fallback's safe set): unsafe = non-foot ground force > {THR:.0f}N, no resets.")
    print(f"cells = unsafe-step-frac / ever-unsafe / force[p50,p90,max]N. {NENV} envs x {STEPS*DT:.0f}s.")
    for label, base, fr, kfn in SCEN:
        print(f"\n--- {label} ---")
        for a in ARMS:
            uf, ev, p50, p90, mx = evalf(a, task_for(a, base), fr, kfn)
            print(f"  {a:8} | unsafe {uf:5.2f} step-frac | ever {ev:4.2f} | force p50 {p50:6.1f}  p90 {p90:6.1f}  max {mx:7.1f} N")
    print(f"\nIf STANCE has LOW unsafe here (soft settle <={THR:.0f}N) but was unsafe under the standing/height margin,")
    print("that CONFIRMS: the fallback needs its OWN g_x, and a soft lie-down/settle IS safe under a force margin.")
