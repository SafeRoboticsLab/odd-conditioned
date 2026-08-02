"""E033 — evaluate the trained soft-descent fallback (go2_payload_descent, ReachAvoidSAC single-player).
Training showed safe_rate~1.0 but success_rate~0 (it learned to STAND safely, rarely descends). This renders
what it actually does and measures: base-height trajectory (does it lower?), uprightness, non-foot contact
force (peak/resting), and reach-success frac (l>=0 & g>=0). end_criterion='timeout' so it can settle w/o reset.
"""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, torch as th, imageio.v2 as imageio
sys.path.insert(0, "external/robot-safety-sandbox")
import safety_sb3
from robot_safety_sandbox import make_tensor, spec, algo_name
from safety_sb3.tensor_env import TensorVecNormalize
try:
    from PIL import Image, ImageDraw; HAVE_PIL = True
except Exception: HAVE_PIL = False

RUN = "results/go2_payload_runs/go2_payload_descent_reachavoidsac"; DEV = "cuda:0"; STEPS = 400


def load(nenv, render):
    env = make_tensor("go2_payload_descent", nenv, DEV, adversary=False, end_criterion="timeout",
                      **({"render_mode": "rgb_array"} if render else {}))
    if render:
        try: env.mj.cfg.viewer.max_extra_envs = max(1, nenv - 1)
        except Exception: pass
    Algo = getattr(safety_sb3, algo_name("go2_payload_descent", adversary=False).replace("PPO", "SAC"))
    m = Algo.load(f"{RUN}/final_model.zip", env=env, device=DEV,
                  custom_objects={"_use_lb": False, "_lb_dir": "/tmp/lb", "tensorboard_log": None, "buffer_size": 1})
    nm = TensorVecNormalize.load(f"{RUN}/tensornormalize.pt", env); nm.training = False
    return env, m, nm


def run():
    env, m, nm = load(6, True); robot = env.mj.scene["robot"]; sens = env.mj.scene["nonfoot_ground_touch"]
    obs = env.reset(); frames = []; zs = []; ups = []; ff = []; reach = 0
    for t in range(STEPS):
        with th.no_grad():
            a = th.clamp(m.policy._predict(nm.normalize_obs(obs), deterministic=True), -1, 1)
        obs, g, dones, touts, l = env.step_tensor(a)
        zs.append(float(robot.data.root_link_pos_w[:, 2].mean()))
        ups.append(float((-robot.data.projected_gravity_b[:, 2]).mean()))
        fh = sens.data.force_history if sens.data.force_history is not None else sens.data.force
        ff.append(float(th.norm(fh, dim=-1).flatten(1).amax(1).max()))
        reach += int(((g >= 0) & (l >= 0)).sum())
        f = np.asarray(env.render()).copy()
        if HAVE_PIL:
            im = Image.fromarray(f); ImageDraw.Draw(im).text((6, 6), f"DESCENT policy  z={zs[-1]:.2f} up={ups[-1]:+.2f}", fill=(255, 255, 0)); f = np.asarray(im)
        frames.append(f)
    env.close()
    return frames, np.array(zs), np.array(ups), np.array(ff), reach / (6 * STEPS)


if __name__ == "__main__":
    frames, zs, ups, ff, reach_frac = run()
    print(f"DESCENT policy: base-z {zs[0]:.2f}->min {zs.min():.2f}->end {zs[-1]:.2f} m | up min {ups.min():+.2f} end {ups[-1]:+.2f} "
          f"| non-foot force peak {ff.max():.0f} rest(last 0.8s) {ff[-40:].mean():.0f} N | reach(l>=0&g>=0) frac {reach_frac:.2f}")
    os.makedirs("results/E033", exist_ok=True); imageio.mimsave("results/E033/descent_eval.mp4", frames, fps=30, macro_block_size=1)
    print(f"wrote results/E033/descent_eval.mp4")
