"""E025 — theta-CHIRP / frequency sweep: localize the worst modulation frequency of sinusoidal ODD forcing.
E023 (coarse) showed oscillatory rigidity forcing is the worst dynamic ODD, but was still rising at its
low-freq end (period 3s) -> peak unresolved. This sweeps k(t)=150+150cos(2*pi*f*t) (full rigid<->sloshy) over
a wide band 0.2..7 Hz with 10 s episodes (low freqs get real cycles). Reads the fall-rate-vs-frequency curve
per arm: an INTERIOR PEAK => true parametric resonance (Mathieu 2:1 tongue near a closed-loop mode); monotone
rise toward low freq => tracking/quasi-static-drift dominated. The worst f = the strongest dynamic-ODD attack.
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
STEPS = 500; T0 = 60; NENV = 128; DT = 0.02
FREQS = [0.2, 0.3, 0.4, 0.55, 0.75, 1.0, 1.4, 2.0, 2.8, 4.0, 5.5, 7.0]   # Hz
ARMS = {"CONDITIONED": (f"{R}/go2_payload_conditioned_gameplaysac", "go2_payload_conditioned_heavy_sloshy"),
        "BLIND": (f"{R}/go2_payload_blind_gameplaysac", "go2_payload_heavy_sloshy"),
        "HISTORY": (f"{R}/go2_payload_history_gameplaysac", "go2_payload_history_heavy_sloshy")}
COL = {"CONDITIONED": "#9467bd", "BLIND": "#cc4c3b", "HISTORY": "#2a9d3a"}


def jidx(env):
    m = env.mj.sim.mj_model
    js = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"robot/payload_j{i}") for i in range(1, 5)]
    return th.tensor([j for j in js if j >= 0], device=DEV, dtype=th.long)


def fall_rate(run_dir, task, f):
    env = make_tensor(task, NENV, DEV, adversary=True)
    Algo = getattr(safety_sb3, algo_name(task, adversary=True).replace("PPO", "SAC"))
    m = Algo.load(f"{run_dir}/final_model.zip", env=env, device=DEV,
                  custom_objects={"_use_lb": False, "_lb_dir": "/tmp/lb", "tensorboard_log": None, "buffer_size": 1})
    nm = TensorVecNormalize.load(f"{run_dir}/tensornormalize.pt", env); nm.training = False
    dd = spec(task).dstb_dim; env.force_scale = FR * th.ones(NENV, device=DEV)
    dstb = (PULL / PULL.norm()).to(DEV)[None].expand(NENV, dd).contiguous()
    ji = jidx(env); obs = env.reset(); nfail = 0
    for t in range(STEPS):
        k = 300.0 if t < T0 else 150.0 + 150.0 * math.cos(2 * math.pi * f * (t - T0) * DT)
        env.mj.sim.model.jnt_stiffness[:, ji] = k
        with th.no_grad():
            a = th.clamp(m.policy._predict(nm.normalize_obs(obs), deterministic=True), -1, 1)
        obs, g, dones, touts, l = env.step_tensor(th.cat([a, dstb], dim=1))
        if t >= T0: nfail += int((dones & (~touts)).sum())
    env.close()
    return nfail / (NENV * (STEPS - T0) * DT)


if __name__ == "__main__":
    arms = list(ARMS)
    print(f"\nE025 theta-CHIRP freq sweep — falls/env-sec, full rigid<->sloshy forcing, {FR*50:.0f}N pull, {STEPS*DT:.0f}s eps.")
    print(f"{'freq(Hz)':>9} {'period(s)':>10} | " + " | ".join(f"{a:>12}" for a in arms) + " | B-H")
    data = {a: [] for a in arms}
    for f in FREQS:
        cells = {a: fall_rate(ARMS[a][0], ARMS[a][1], f) for a in arms}
        for a in arms: data[a].append(cells[a])
        print(f"{f:9.2f} {1/f:10.2f} | " + " | ".join(f"{cells[a]:12.2f}" for a in arms) + f" | {cells['BLIND']-cells['HISTORY']:+.2f}")
    bi = int(np.argmax(data["BLIND"]))
    print(f"\nWORST freq for BLIND: {FREQS[bi]:.2f} Hz (period {1/FREQS[bi]:.2f} s), fall rate {data['BLIND'][bi]:.2f}")
    interior = 0 < bi < len(FREQS) - 1
    print(f"peak is {'INTERIOR -> parametric resonance' if interior else 'at a BAND EDGE -> re-sweep further out'}")
    fig, ax = plt.subplots(figsize=(8, 5))
    for a in arms:
        ax.plot(FREQS, data[a], "o-", color=COL[a], lw=2, label=a)
    ax.axvline(FREQS[bi], color="#cc4c3b", ls="--", lw=1, alpha=0.5)
    ax.set_xscale("log"); ax.set_xlabel("ODD modulation frequency (Hz)"); ax.set_ylabel("falls / env-sec (lower = safer)")
    ax.set_title(f"Dynamic-ODD frequency response (rigid<->sloshy forcing, {FR*50:.0f}N pull)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); os.makedirs("results/E025", exist_ok=True); fig.savefig("results/E025/chirp_resonance.png", dpi=120)
    np.savez(f"{R}/chirp_resonance.npz", freqs=np.array(FREQS), **{a: np.array(data[a]) for a in arms})
    print(f"wrote {R}/chirp_resonance.png and .npz")
