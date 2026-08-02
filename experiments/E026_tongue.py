"""E026 — Ince-Strutt / Arnold TONGUE map: fall rate over (modulation frequency x modulation DEPTH) for the
sinusoidal ODD forcing. E025 showed FULL-depth forcing destabilizes blind at every frequency (tongues merged).
This sweeps depth from ~0 up: at SMALL depth only the resonant frequency should destabilize blind (a narrow
tongue), widening with depth into E025's broadband plateau. The narrowest-depth cell that lights up = the
tongue APEX = the resonant frequency, and depth-threshold ε_min(f) = the parametric-resonance boundary. HISTORY
map should stay flat-low (rejects the forcing at all f, depth) — the punchline overlaid.

k(t) = 150 + depth*150*sin(2*pi*f*(t-T0)*dt), warmup held at 150 (base operating point; sin => continuous).
"""
import os, sys, math
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, torch as th, mujoco
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "external/robot-safety-sandbox")
import safety_sb3
from robot_safety_sandbox import make_tensor, spec, algo_name
from safety_sb3.tensor_env import TensorVecNormalize

R = "results/go2_payload_runs"; DEV = "cuda:0"; PULL = th.tensor([0., 1., 0.]); FR = 1.0
STEPS = 500; T0 = 60; NENV = 128; DT = 0.02; K0 = 150.0
FREQS = [0.4, 0.75, 1.0, 1.4, 2.0, 2.8, 4.0]          # Hz
DEPTHS = [0.0, 0.1, 0.25, 0.4, 0.6, 0.8, 1.0]         # fraction of K0 (1.0 = full 0..300 swing)
ARMS = {"BLIND": (f"{R}/go2_payload_blind_gameplaysac", "go2_payload_heavy_sloshy"),
        "HISTORY": (f"{R}/go2_payload_history_gameplaysac", "go2_payload_history_heavy_sloshy")}


def jidx(env):
    m = env.mj.sim.mj_model
    js = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"robot/payload_j{i}") for i in range(1, 5)]
    return th.tensor([j for j in js if j >= 0], device=DEV, dtype=th.long)


def load(run_dir, task):
    env = make_tensor(task, NENV, DEV, adversary=True)
    Algo = getattr(safety_sb3, algo_name(task, adversary=True).replace("PPO", "SAC"))
    m = Algo.load(f"{run_dir}/final_model.zip", env=env, device=DEV,
                  custom_objects={"_use_lb": False, "_lb_dir": "/tmp/lb", "tensorboard_log": None, "buffer_size": 1})
    nm = TensorVecNormalize.load(f"{run_dir}/tensornormalize.pt", env); nm.training = False
    env.force_scale = FR * th.ones(NENV, device=DEV)
    dstb = (PULL / PULL.norm()).to(DEV)[None].expand(NENV, spec(task).dstb_dim).contiguous()
    return env, m, nm, dstb, jidx(env)


def fall_rate(env, m, nm, dstb, ji, f, depth):
    obs = env.reset(); nfail = 0; A = depth * K0
    for t in range(STEPS):
        k = K0 if t < T0 else K0 + A * math.sin(2 * math.pi * f * (t - T0) * DT)
        env.mj.sim.model.jnt_stiffness[:, ji] = k
        with th.no_grad():
            a = th.clamp(m.policy._predict(nm.normalize_obs(obs), deterministic=True), -1, 1)
        obs, g, dones, touts, l = env.step_tensor(th.cat([a, dstb], dim=1))
        if t >= T0: nfail += int((dones & (~touts)).sum())
    return nfail / (NENV * (STEPS - T0) * DT)


if __name__ == "__main__":
    maps = {}
    for arm, (rd, task) in ARMS.items():
        env, m, nm, dstb, ji = load(rd, task)
        M = np.zeros((len(DEPTHS), len(FREQS)))
        for i, d in enumerate(DEPTHS):
            for j, f in enumerate(FREQS):
                M[i, j] = fall_rate(env, m, nm, dstb, ji, f, d)
        env.close(); maps[arm] = M
        print(f"\n=== {arm} — falls/env-sec  (rows=depth, cols=freq Hz) ===")
        print(f"{'eps.f':>8} | " + " | ".join(f"{f:>6.2f}" for f in FREQS))
        for i, d in enumerate(DEPTHS):
            print(f"{d:8.2f} | " + " | ".join(f"{M[i,j]:6.2f}" for j in range(len(FREQS))))
    # tongue apex: floor = depth-0 row; per freq the min depth where fall exceeds floor+0.10
    B = maps["BLIND"]; floor = B[0].mean()
    apex_d, apex_f = 1e9, None
    for j, f in enumerate(FREQS):
        for i, d in enumerate(DEPTHS):
            if d > 0 and B[i, j] > floor + 0.10:
                if d < apex_d: apex_d, apex_f = d, f
                break
    print(f"\nBLIND static-floor (depth 0) = {floor:.2f}. Tongue APEX (lowest depth that lights up): "
          f"f={apex_f} Hz at depth eps_min={apex_d}" if apex_f else "\nno cell crossed floor+0.10 -> raise depth res")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    vmax = max(maps["BLIND"].max(), maps["HISTORY"].max())
    for ax, arm in zip(axes, ["BLIND", "HISTORY"]):
        im = ax.imshow(maps[arm], origin="lower", aspect="auto", cmap="inferno", vmin=0, vmax=vmax)
        ax.set_xticks(range(len(FREQS))); ax.set_xticklabels([f"{f:.2g}" for f in FREQS])
        ax.set_yticks(range(len(DEPTHS))); ax.set_yticklabels([f"{d:.2g}" for d in DEPTHS])
        ax.set_xlabel("modulation frequency (Hz)"); ax.set_title(f"{arm}  falls/env-sec")
        for i in range(len(DEPTHS)):
            for j in range(len(FREQS)):
                ax.text(j, i, f"{maps[arm][i,j]:.2f}", ha="center", va="center", fontsize=7,
                        color="w" if maps[arm][i, j] < vmax * 0.6 else "k")
    axes[0].set_ylabel("modulation depth eps (fraction of k0=150)")
    fig.colorbar(im, ax=axes, label="falls / env-sec");
    fig.suptitle("Dynamic-ODD parametric-resonance tongue map (rigidity forcing, 50N pull)")
    os.makedirs("results/E026", exist_ok=True); fig.savefig("results/E026/tongue_map.png", dpi=120, bbox_inches="tight")
    np.savez(f"{R}/tongue_map.npz", freqs=np.array(FREQS), depths=np.array(DEPTHS), **maps)
    print(f"wrote {R}/tongue_map.png and .npz")
