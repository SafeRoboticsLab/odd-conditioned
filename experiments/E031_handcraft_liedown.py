"""E031 (v3) — HANDCRAFTED soft-descent / lie-down fallback, GUARDED DESCENT. A from-scratch open-loop fold
tips over: the Go2 cannot hold itself without active balance (a=0 already tips). So the fallback reuses the
robot's EXISTING balance (the trained standing policy, which reads IMU+proprio) and FADES it into a scripted
symmetric squat: action = (1-tau)*balance + tau*fold. Early -> policy holds it upright; late -> the fold
dominates and it settles belly-down. No new training. Renders a video; reports base-height descent, uprightness
(stays up during descent?), and peak non-foot contact force (soft?). end_criterion='timeout' (no resets).

BAL_POL is the standing policy used purely for balance during the descent.
"""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, torch as th, imageio.v2 as imageio, mujoco
sys.path.insert(0, "external/robot-safety-sandbox")
import safety_sb3
from robot_safety_sandbox import make_tensor, spec, algo_name
from safety_sb3.tensor_env import TensorVecNormalize
try:
    from PIL import Image, ImageDraw; HAVE_PIL = True
except Exception: HAVE_PIL = False

DEV = "cuda:0"; PULL = th.tensor([0., 1., 0.]); DT = 0.02; NENV = 2; STEPS = 360
HOLD, DESC = 40, 240          # 0.8s balance, then fade to fold over 4.8s, then rest
TASK = "go2_payload_history_light_rigid"
BAL = "results/go2_payload_runs/go2_payload_history_gameplaysac"
FOLD_THIGH, FOLD_CALF, FOLD_HIP = 0.55, -1.15, 0.0  # moderate knee-dominant squat, feet planted (no splay)
F_TARGET = 25.0                                     # stop lowering if ANY non-foot contact appears (stay on feet)


def joint_groups(env):
    m = env.mj.sim.mj_model; thigh, calf, hip, side = [], [], [], []
    for i in range(12):
        jid = int(m.actuator_trnid[i, 0]); nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid) or ""
        if "thigh" in nm: thigh.append(i)
        elif "calf" in nm: calf.append(i)
        elif "hip" in nm: hip.append(i)
        side.append(1.0 if nm[1:2] == "R" else -1.0)          # splay: R hips +, L hips -
    return (th.tensor(thigh, device=DEV), th.tensor(calf, device=DEV),
            th.tensor(hip, device=DEV), th.tensor(side, device=DEV))


def run(fr):
    env = make_tensor(TASK, NENV, DEV, adversary=True, render_mode="rgb_array", end_criterion="timeout")
    try: env.mj.cfg.viewer.max_extra_envs = 1
    except Exception: pass
    Algo = getattr(safety_sb3, algo_name(TASK, adversary=True).replace("PPO", "SAC"))
    m = Algo.load(f"{BAL}/final_model.zip", env=env, device=DEV,
                  custom_objects={"_use_lb": False, "_lb_dir": "/tmp/lb", "tensorboard_log": None, "buffer_size": 1})
    nm = TensorVecNormalize.load(f"{BAL}/tensornormalize.pt", env); nm.training = False
    dd = spec(TASK).dstb_dim; env.force_scale = fr * th.ones(NENV, device=DEV)
    dstb = (PULL / PULL.norm()).to(DEV)[None].expand(NENV, dd).contiguous()
    robot = env.mj.scene["robot"]; sens = env.mj.scene["nonfoot_ground_touch"]
    thigh, calf, hip, side = joint_groups(env); obs = env.reset()
    fold = th.zeros(NENV, 12, device=DEV)
    fold[:, thigh] = FOLD_THIGH / 0.75; fold[:, calf] = FOLD_CALF / 0.75
    fold[:, hip] = (FOLD_HIP * side[hip]) / 0.75              # splay legs outward for a wide belly-down base
    tau = th.zeros(NENV, device=DEV)                          # PER-ENV descent progress (force-gated)
    frames, zs, ups, ff = [], [], [], []
    for t in range(STEPS):
        fh = sens.data.force_history if sens.data.force_history is not None else sens.data.force
        force = th.norm(fh, dim=-1).flatten(1).amax(1)        # [NENV] current non-foot ground force
        up_e = -robot.data.projected_gravity_b[:, 2]          # per-env uprightness
        if t > HOLD:   # lower only while STILL STABLE (upright) and contact gentle -> settles at lowest safe crouch
            ok = (force < F_TARGET) & (up_e > 0.90)
            tau = th.clamp(tau + th.where(ok, 1.0 / DESC, 0.0), 0.0, 1.0)
        tau_f = tau * tau * (3.0 - 2.0 * tau)
        with th.no_grad():
            bal = th.clamp(m.policy._predict(nm.normalize_obs(obs), deterministic=True), -1, 1)
        a = (1.0 - 0.3 * tau)[:, None] * bal + tau_f[:, None] * fold   # 70% balance floor: stay upright on feet
        obs, g, dones, touts, l = env.step_tensor(th.cat([th.clamp(a, -1.5, 1.5), dstb], dim=1))
        zs.append(float(robot.data.root_link_pos_w[:, 2].mean()))
        ups.append(float((-robot.data.projected_gravity_b[:, 2]).mean()))
        ff.append(float(force.max()))
        f = np.asarray(env.render()).copy()
        if HAVE_PIL:
            im = Image.fromarray(f); ImageDraw.Draw(im).text((6, 6), f"GUARDED lie-down {fr*50:.0f}N  z={zs[-1]:.2f} up={ups[-1]:+.2f} tau={float(tau.mean()):.2f}", fill=(255, 255, 0)); f = np.asarray(im)
        frames.append(f)
    env.close()
    return frames, np.array(zs), np.array(ups), np.array(ff)


if __name__ == "__main__":
    frames, zs, ups, ff = run(0.0)
    pk = int(ff.argmax()); rest = ff[-40:].mean()
    print(f"0N guarded descent: base-z {zs[0]:.2f}->min {zs.min():.2f}->end {zs[-1]:.2f} m | up min {ups.min():+.2f} end {ups[-1]:+.2f}")
    print(f"  non-foot force: peak {ff.max():.1f} N @ t={pk*DT:.1f}s | resting(last 0.8s) mean {rest:.1f} N")
    for tt in (HOLD, HOLD+DESC//2, HOLD+DESC, STEPS-1):
        print(f"  t={tt*DT:4.1f}s: z={zs[tt]:.2f} up={ups[tt]:+.2f} force={ff[tt]:.1f}N")
    os.makedirs("results/E031", exist_ok=True); imageio.mimsave("results/E031/handcraft_liedown.mp4", frames, fps=30, macro_block_size=1)
    print("wrote results/go2_payload_runs/handcraft_liedown.mp4")
