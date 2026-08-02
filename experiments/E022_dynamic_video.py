"""E022 — DYNAMIC ODD video: robot side-by-side with the ODD-change-vs-time graph, one video per arm
(BLIND vs HISTORY). Shows the mid-episode rigidity jump (rigid->sloshy) that shocks the reactive blind
policy but not the tracking history policy. The graph plots the ODD rigidity k(t) (the change the payload
undergoes) and the trunk-upright response, with a moving time cursor synced to the robot frames.

Constant 50 N pull throughout; jnt_stiffness overridden live (300 rigid -> 0 sloshy at t=T_JUMP).
"""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, torch as th, imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
sys.path.insert(0, "external/robot-safety-sandbox")
import safety_sb3
from robot_safety_sandbox import make_tensor, spec, algo_name
from safety_sb3.tensor_env import TensorVecNormalize

R = "results/go2_payload_runs"; DEV = "cuda:0"; PULL = th.tensor([0., 1., 0.]); FR = 1.0
STEPS = 220; T_JUMP = 110; NENV = 2; DT = 0.02
ARMS = {"BLIND": (f"{R}/go2_payload_blind_gameplaysac", "go2_payload_heavy_sloshy"),
        "HISTORY": (f"{R}/go2_payload_history_gameplaysac", "go2_payload_history_heavy_sloshy")}


def jidx(env):
    m = env.mj.sim.mj_model
    js = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"robot/payload_j{i}") for i in range(1, 5)]
    return th.tensor([j for j in js if j >= 0], device=DEV, dtype=th.long)


def rollout(run_dir, task):
    env = make_tensor(task, NENV, DEV, adversary=True, render_mode="rgb_array")
    try: env.mj.cfg.viewer.max_extra_envs = max(1, NENV - 1)
    except Exception: pass
    Algo = getattr(safety_sb3, algo_name(task, adversary=True).replace("PPO", "SAC"))
    m = Algo.load(f"{run_dir}/final_model.zip", env=env, device=DEV,
                  custom_objects={"_use_lb": False, "_lb_dir": "/tmp/lb", "tensorboard_log": None, "buffer_size": 1})
    nm = TensorVecNormalize.load(f"{run_dir}/tensornormalize.pt", env); nm.training = False
    dd = spec(task).dstb_dim; env.force_scale = FR * th.ones(NENV, device=DEV)
    dstb = (PULL / PULL.norm()).to(DEV)[None].expand(NENV, dd).contiguous()
    robot = env.mj.scene["robot"]; ji = jidx(env); obs = env.reset()
    frames, kseries, upseries = [], [], []
    for t in range(STEPS):
        k = 0.0 if t >= T_JUMP else 300.0
        env.mj.sim.model.jnt_stiffness[:, ji] = k
        with th.no_grad():
            a = th.clamp(m.policy._predict(nm.normalize_obs(obs), deterministic=True), -1, 1)
        obs, g, dones, touts, l = env.step_tensor(th.cat([a, dstb], dim=1))
        frames.append(np.asarray(env.render()))
        kseries.append(k)
        upseries.append(float((-robot.data.projected_gravity_b[:, 2]).mean()))
    env.close()
    return frames, np.array(kseries), np.array(upseries)


def graph_frame(k, up, i, H, W):
    t = np.arange(len(k)) * DT
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = fig.add_subplot(111); ax2 = ax.twinx()
    ax.plot(t, k, color="#888", lw=1.2, alpha=0.35)
    ax.plot(t[:i + 1], k[:i + 1], color="#1f77b4", lw=2.4, label="ODD rigidity k")
    ax2.plot(t, up, color="#cc4c3b", lw=1.0, alpha=0.30)
    ax2.plot(t[:i + 1], up[:i + 1], color="#cc4c3b", lw=2.0, label="trunk upright")
    ax.axvline(T_JUMP * DT, color="k", ls="--", lw=1.0, alpha=0.5)
    ax.axvline(t[i], color="#2a9d3a", lw=2.0)
    ax.set_xlim(0, t[-1]); ax.set_ylim(-15, 315); ax2.set_ylim(-0.05, 1.05)
    ax.set_xlabel("time (s)"); ax.set_ylabel("rigidity k (300=rigid, 0=sloshy)", color="#1f77b4")
    ax2.set_ylabel("trunk upright (1=up, <0.4=fallen)", color="#cc4c3b")
    ax.text(T_JUMP * DT, 300, " ODD JUMP\n rigid->sloshy", fontsize=8, va="top")
    fig.tight_layout(pad=0.6)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return buf


if __name__ == "__main__":
    for arm, (rd, task) in ARMS.items():
        frames, k, up = rollout(rd, task)
        H, W = frames[0].shape[:2]; GW = int(W * 0.9)
        comb = []
        for i in range(len(frames)):
            g = graph_frame(k, up, i, H, GW)
            comb.append(np.hstack([frames[i][:H, :W], g[:H, :]]))
        out = f"results/E022/dynamic_odd_{arm.lower()}.mp4"; os.makedirs("results/E022", exist_ok=True)
        imageio.mimsave(out, comb, fps=30, macro_block_size=1)
        print(f"wrote {out} ({len(comb)} frames, {comb[0].shape[1]}x{comb[0].shape[0]})")
