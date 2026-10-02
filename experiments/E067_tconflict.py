"""E067 — the t_conflict test on LEFT-vs-RIGHT leg death (Buzi's pick). Two modes: FR-leg death vs FL-leg death,
from a shared standing state, NO lateral pull (clean left-right mirror). We measure the two clocks:

  t_identify : how fast a discriminator tells FR-death from FL-death from ACTOR-observable proprio (base IMU +
               joints). (The which-leg signal = base tilt direction.)
  t_conflict : the longest a SYMMETRIC (side-agnostic) action prefix keeps the robot RECOVERABLE, before it must
               commit to an asymmetric shift. By left-right symmetry the best COMMON hedge over {FR,FL} is
               symmetric, so we simulate only the FR branch: apply the SYMMETRIZED specialist action for k steps
               (the specialist's instinct minus its left-right commitment), then hand to the full FR specialist
               (mode revealed). t_conflict = max k with high survival.

m_info = t_identify - t_conflict.  m_info>0 => a window (conditioning load-bearing); <=0 => collapse (blind absorbs).
KEY UNKNOWN the data resolves: if the specialist recovers a dead leg by SITTING STRAIGHT (symmetric), the hedge ==
the recovery and t_conflict is trivially large (common "sit" hedge => collapse). If it recovers by an asymmetric
LATERAL SHIFT off the dead leg, delaying it costs survival => a finite t_conflict.
"""
import os, sys, io, contextlib, mujoco
import torch as th
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "external/robot-safety-sandbox")
import numpy as np
from robot_safety_sandbox import make_tensor, spec
from robot_safety_sandbox.eval.policies import load_twin

DEV, N, DT = "cuda:0", 256, 0.02
T0, STEPS, THETA = 30, 140, 0.2
CK = "results/go2_weak_leg_runs/go2_weak_leg_20_soft_adv/checkpoints/model_49999872_steps.zip"  # FR-20% soft specialist
# left-right action mirror: swap FL<->FR, RL<->RR within each joint group; flip hip signs (lateral).
PERM = th.tensor([1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10], device=DEV)
HIPSIGN = th.tensor([-1., -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1], device=DEV)  # hips (first 4) flip


def mirror_action(a):                       # a:[N,12] -> left-right mirrored action
    return a[:, PERM] * HIPSIGN


def leg_ids(m, leg):
    return th.tensor([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"robot/{leg}_{j}_joint")
                      for j in ("hip", "thigh", "calf")], device=DEV)


def fell(inner, alive):
    d = inner.termination_manager._term_dones
    z = th.zeros(inner.num_envs, device=DEV)
    return (d.get("fell_over", z).bool() | d.get("illegal_contact", z).bool()) & alive


# ---- Part A: t_identify (FR vs FL discriminator) ----
def collect_mode(leg):
    with contextlib.redirect_stdout(io.StringIO()):
        env = make_tensor("go2_stabilize", N, DEV, adversary=True); model, norm = load_twin(CK, DEV, quiet=True)
    inner = env.mj; m = inner.sim.mj_model; tm = inner.sim.model
    ids = leg_ids(m, leg); nom = tm.actuator_forcerange[:, ids, :].clone()
    env.force_scale = 0.0 * th.ones(N, device=DEV); dstb = th.zeros(N, 3, device=DEV)
    obs = env.reset(); obss = []
    for t in range(STEPS):
        if t >= T0:
            tm.actuator_forcerange[:, ids, :] = nom * THETA
        with th.no_grad():
            a = th.clamp(model.policy._predict(norm(obs), deterministic=True), -1, 1)
        obs, g, dones, touts, l = env.step_tensor(th.cat([a, dstb], dim=1))
        obss.append(obs[:, :47].clone())
    env.close(); return th.stack(obss, 1).cpu().numpy()


def auc(sc, y):
    order = np.argsort(sc); r = np.empty(len(sc)); r[order] = np.arange(1, len(sc) + 1)
    P, Q = y.sum(), (1 - y).sum()
    return 0.5 if P == 0 or Q == 0 else float((r[y == 1].sum() - P * (P + 1) / 2) / (P * Q))


def t_identify():
    XF, XL = collect_mode("FR"), collect_mode("FL"); W = 6
    print("t_identify — FR-death vs FL-death discriminator AUC vs τ (6-frame window):")
    dstar = None
    for tau in range(0, 25):
        t = T0 + tau
        if t < W: continue
        A = XF[:, t - W + 1:t + 1].reshape(N, -1); B = XL[:, t - W + 1:t + 1].reshape(N, -1)
        X = np.concatenate([A, B]); y = np.concatenate([np.ones(N), np.zeros(N)])
        idx = np.random.RandomState(0).permutation(2 * N); tr, te = idx[:int(1.6 * N)], idx[int(1.6 * N):]
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6; Xn = (X - mu) / sd
        w = np.zeros(Xn.shape[1]); b = 0.
        for _ in range(250):
            p = 1 / (1 + np.exp(-(Xn[tr] @ w + b))); gr = p - y[tr]
            w -= 0.02 * (Xn[tr].T @ gr / len(tr) + 1e-3 * w); b -= 0.02 * gr.mean()
        a = auc(Xn[te] @ w + b, y[te])
        if a > 0.9 and dstar is None: dstar = tau
        if tau % 2 == 0: print(f"  τ={tau:2d} ({tau*DT:.2f}s)  AUC={a:.2f}")
    print(f"  => t_identify (AUC>0.9) = {dstar} steps = {dstar*DT if dstar else float('nan')}s")
    return dstar


# ---- Part B: t_conflict (symmetric prefix k steps, then full FR specialist) ----
def survival_with_prefix(kpref):
    with contextlib.redirect_stdout(io.StringIO()):
        env = make_tensor("go2_stabilize", N, DEV, adversary=True); model, norm = load_twin(CK, DEV, quiet=True)
    inner = env.mj; m = inner.sim.mj_model; tm = inner.sim.model
    ids = leg_ids(m, "FR"); nom = tm.actuator_forcerange[:, ids, :].clone()
    env.force_scale = 0.0 * th.ones(N, device=DEV); dstb = th.zeros(N, 3, device=DEV)
    obs = env.reset(); alive = th.ones(N, dtype=th.bool, device=DEV); asym_frac = []
    for t in range(STEPS):
        if t >= T0:
            tm.actuator_forcerange[:, ids, :] = nom * THETA
        with th.no_grad():
            a = th.clamp(model.policy._predict(norm(obs), deterministic=True), -1, 1)
        # for the first kpref steps AFTER onset, symmetrize the action (no left-right commitment)
        if T0 <= t < T0 + kpref:
            a_sym = 0.5 * (a + mirror_action(a))
            # log how asymmetric the specialist WANTED to be (|a - a_sym|) = commitment magnitude
            asym_frac.append(float((a - a_sym).abs().mean()))
            a = a_sym
        obs, g, dones, touts, l = env.step_tensor(th.cat([a, dstb], dim=1))
        alive &= ~fell(inner, alive)
    env.close()
    return float(alive.float().mean()), (np.mean(asym_frac) if asym_frac else 0.0)


def t_conflict():
    print("\nt_conflict — survival(symmetric prefix k steps -> full FR specialist). No pull, FR dies at t0.")
    print(f"{'k(steps)':>8} {'k(s)':>6} {'survival':>9} {'|asym want|':>11}")
    base = None; tc = None
    for k in [0, 1, 2, 3, 4, 6, 8, 10, 12, 15, 20]:
        s, asym = survival_with_prefix(k)
        if base is None: base = s
        drop = base - s
        if drop > 0.15 and tc is None and k > 0: tc = k     # survival collapses vs k=0 -> commitment needed
        print(f"{k:>8} {k*DT:>6.2f} {s:>9.2f} {asym:>11.3f}")
    print(f"  => t_conflict (k where survival drops >0.15 vs immediate-commit) = {tc} steps"
          f" = {tc*DT if tc else float('nan')}s  (None/large => symmetric hedge works => COLLAPSE)")
    return tc


if __name__ == "__main__":
    ti = t_identify()
    tc = t_conflict()
    print(f"\n=== m_info = t_identify - t_conflict ===")
    print(f"t_identify={ti} steps, t_conflict={tc} steps")
    if tc is None:
        print("VERDICT: t_conflict large (symmetric hedge recovers both) => m_info<0 => COLLAPSE (blind absorbs).")
    else:
        print(f"m_info = {(ti or 0)-(tc or 0)} steps. >0 => WINDOW (conditioning load-bearing); <=0 => collapse.")
