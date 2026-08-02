"""E034 — measure the DROP force/velocity profile of the (near-)converged descent policy, to calibrate a
tighter contact-force gate. The 100M policy reaches ~100% success but FLOPS to the floor (ep_len ~20 steps =
0.4s). Current g gate: non-foot ground force <= 200N. We measure: peak touchdown force, descent speed, and the
settled resting force across 128 envs -> pick a new F_SAFE below the flop peak but above a gentle settle, and
see how much a velocity/impact bound is needed. end_criterion='timeout' (no reset) so the drop+settle plays out.
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

RUN = "results/go2_payload_runs/go2_payload_descent_reachavoidsac"
MODEL = f"{RUN}/checkpoints/model_74999808_steps.zip"
NORM = f"{RUN}/checkpoints/tensornorm_75001344.pt"
DEV = "cuda:0"; DT = 0.02; STEPS = 100


def load(nenv, render):
    env = make_tensor("go2_payload_descent", nenv, DEV, adversary=False, end_criterion="timeout",
                      **({"render_mode": "rgb_array"} if render else {}))
    if render:
        try: env.mj.cfg.viewer.max_extra_envs = max(1, nenv - 1)
        except Exception: pass
    Algo = getattr(safety_sb3, algo_name("go2_payload_descent", adversary=False).replace("PPO", "SAC"))
    m = Algo.load(MODEL, env=env, device=DEV,
                  custom_objects={"_use_lb": False, "_lb_dir": "/tmp/lb", "tensorboard_log": None, "buffer_size": 1})
    nm = TensorVecNormalize.load(NORM, env); nm.training = False
    return env, m, nm


def stats():
    env, m, nm = load(128, False); robot = env.mj.scene["robot"]; sens = env.mj.scene["nonfoot_ground_touch"]
    obs = env.reset(); n = 128
    peak_f = th.zeros(n, device=DEV); prev_z = robot.data.root_link_pos_w[:, 2].clone(); peak_desc = th.zeros(n, device=DEV)
    force_t = []; z_t = []
    for t in range(STEPS):
        with th.no_grad():
            a = th.clamp(m.policy._predict(nm.normalize_obs(obs), deterministic=True), -1, 1)
        obs, g, dones, touts, l = env.step_tensor(a)
        z = robot.data.root_link_pos_w[:, 2]
        fh = sens.data.force_history if sens.data.force_history is not None else sens.data.force
        f = th.norm(fh, dim=-1).flatten(1).amax(1)
        peak_f = th.maximum(peak_f, f)
        desc = (prev_z - z) / DT                              # downward speed (m/s), positive = descending
        peak_desc = th.maximum(peak_desc, desc); prev_z = z.clone()
        force_t.append(float(f.mean())); z_t.append(float(z.mean()))
    settle_f = f                                             # last-step force ~ resting
    env.close()
    def pct(x): return [float(th.quantile(x, q)) for q in (0.5, 0.9, 0.99)]
    print(f"  peak touchdown force  p50/p90/p99 = {pct(peak_f)} N   (max {float(peak_f.max()):.0f})")
    print(f"  peak descent speed    p50/p90/p99 = {pct(peak_desc)} m/s")
    print(f"  settled resting force p50/p90/p99 = {pct(settle_f)} N")
    print(f"  base-z trace (mean): " + " ".join(f"{z_t[i]:.2f}" for i in range(0, STEPS, 10)))
    print(f"  force trace  (mean): " + " ".join(f"{force_t[i]:.0f}" for i in range(0, STEPS, 10)))


def render():
    env, m, nm = load(4, True); robot = env.mj.scene["robot"]; obs = env.reset(); frames = []
    for t in range(STEPS):
        with th.no_grad():
            a = th.clamp(m.policy._predict(nm.normalize_obs(obs), deterministic=True), -1, 1)
        obs, *_ = env.step_tensor(a)
        f = np.asarray(env.render()).copy()
        if HAVE_PIL:
            z = float(robot.data.root_link_pos_w[:, 2].mean())
            im = Image.fromarray(f); ImageDraw.Draw(im).text((6, 6), f"DESCENT 75M  z={z:.2f}", fill=(255, 255, 0)); f = np.asarray(im)
        frames.append(f)
    env.close()
    os.makedirs("results/E034", exist_ok=True); imageio.mimsave("results/E034/descent_75M_eval.mp4", frames, fps=30, macro_block_size=1)
    print(f"wrote results/E034/descent_75M_eval.mp4")


if __name__ == "__main__":
    print("E034 drop force/velocity profile (75M checkpoint, contact gate currently F_SAFE=200N):")
    stats(); render()
