"""E066 stage-0 — the ALIASING GATE (δ_id): are SHOVE and SLIP distinguishable from ACTOR-observable proprio, and
how fast? The cheap pre-RL kill test for the multi-mode pivot (professor 2026-08-21b).

Context: the stabilizer holds a constant lateral lean-force F0 (feet gripping). At onset t0 BOTH modes cause a
sudden lateral base acceleration (aliased):
  SHOVE : the lateral force JUMPS F0 -> F0+dF for ~15 steps (impulse), friction normal.
  SLIP  : the foot friction DROPS (ice) sustained, force stays F0 -> feet slide.
Actor obs = base IMU (ang vel, projected gravity) + joint pos/vel + last action (NO foot-force sensor; Go2-real).
We train a discriminator on the actor-obs window at each time-since-onset τ and report AUC(τ). δ_id = first τ with
AUC>0.9. If δ_id is ~1-2 steps -> NOT aliased -> basis absorbable -> KILL. If AUC stays low for many steps ->
genuinely confusable -> proceed to specialists (stage 1).
"""
import os, sys, io, contextlib, mujoco
import torch as th
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "external/robot-safety-sandbox")
import numpy as np
from robot_safety_sandbox import make_tensor, spec
from robot_safety_sandbox.eval.policies import load_twin

DEV, N, DT = "cuda:0", 512, 0.02
STEPS, T0 = 160, 60
F0, DF = 0.3, 0.6                      # lean 15N; shove jumps to (F0+DF)*50 = 45N for ~15 steps
SHOVE_LEN = 15
MU_ICE = 0.02
CK = "results/go2_weak_leg_runs/go2_stabilize_adv/checkpoints/model_49999872_steps.zip"
PROPRIO, W = 47, 8                     # actor-obs dim; discriminator window
PULL = th.tensor([0., 1., 0.])


def rollout(mode):
    with contextlib.redirect_stdout(io.StringIO()):
        env = make_tensor("go2_stabilize", N, DEV, adversary=True); model, norm = load_twin(CK, DEV, quiet=True)
    inner = env.mj; m = inner.sim.mj_model; tm = inner.sim.model
    foot = [g for g in range(m.ngeom) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").endswith("foot_collision")]
    mu0 = tm.geom_friction[:, foot, 0].clone()
    dstb = (PULL).to(DEV)[None].expand(N, spec("go2_stabilize").dstb_dim).contiguous()
    # RANDOMIZE magnitudes per-env so the discriminator cannot use size as a mode proxy — only DYNAMICS.
    df_env = (th.rand(N, device=DEV) * 0.5 + 0.25)          # shove jump ∈ [0.25,0.75] -> [12,37]N over F0
    mu_env = (th.rand(N, device=DEV) * 0.05 + 0.01)         # ice μ ∈ [0.01,0.06]
    obs = env.reset(); obss = []; fell = th.zeros(N, dtype=th.bool, device=DEV); vy = []
    for t in range(STEPS):
        fs = F0 * th.ones(N, device=DEV)
        if t >= T0 and mode == "shove" and t < T0 + SHOVE_LEN:
            fs = F0 + df_env
        env.force_scale = fs
        if t == T0 and mode == "slip":
            tm.geom_friction[:, foot, 0] = mu_env[:, None]
        with th.no_grad():
            a = th.clamp(model.policy._predict(norm(obs), deterministic=True), -1, 1)
        obs, g, dones, touts, l = env.step_tensor(th.cat([a, dstb], dim=1))
        obss.append(obs[:, :PROPRIO].clone()); fell |= dones & ~touts
        vy.append(float(inner.scene["robot"].data.root_link_lin_vel_b[:, 1].abs().mean()))
    env.close()
    return th.stack(obss, 1), fell, np.array(vy)   # [N, STEPS, 47]


def auc(scores, labels):
    order = np.argsort(scores)                       # Mann-Whitney U (rank-based AUC)
    ranks = np.empty(len(scores), dtype=float); ranks[order] = np.arange(1, len(scores) + 1)
    P, Neg = labels.sum(), (1 - labels).sum()
    if P == 0 or Neg == 0: return 0.5
    return float((ranks[labels == 1].sum() - P * (P + 1) / 2) / (P * Neg))


if __name__ == "__main__":
    Xs, fs, vs = rollout("shove"); Xl, fl, vl = rollout("slip")
    print(f"onset-aliasing check — |lateral vel| at onset+1: shove={vs[T0+1]:.3f} slip={vl[T0+1]:.3f} (close = aliased)")
    print(f"fall-fraction by end: shove={float(fs.float().mean()):.2f} slip={float(fl.float().mean()):.2f}")
    Xs, Xl = Xs.cpu().numpy(), Xl.cpu().numpy()
    print("\nδ_id — discriminator AUC vs time-since-onset τ (window of last 8 actor-obs frames):")
    print(f"{'τ(steps)':>8} {'τ(s)':>6} {'AUC':>6}")
    dstar = None
    for tau in range(0, 40):
        t = T0 + tau
        if t < W: continue
        # window features [N, W*47]
        fsh = Xs[:, t - W + 1:t + 1, :].reshape(N, -1); fsl = Xl[:, t - W + 1:t + 1, :].reshape(N, -1)
        X = np.concatenate([fsh, fsl], 0); y = np.concatenate([np.ones(N), np.zeros(N)])
        # train/test split
        idx = np.random.RandomState(0).permutation(2 * N); tr, te = idx[:int(1.6 * N)], idx[int(1.6 * N):]
        # standardize + logistic regression (closed-ish via sklearn if available, else simple GD)
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Xn = (X - mu) / sd
        w = np.zeros(Xn.shape[1]); b = 0.0
        for _ in range(300):
            z = Xn[tr] @ w + b; p = 1 / (1 + np.exp(-z)); gr = p - y[tr]
            w -= 0.01 * (Xn[tr].T @ gr / len(tr) + 1e-3 * w); b -= 0.01 * gr.mean()
        sc = Xn[te] @ w + b; a = auc(sc, y[te])
        if a > 0.9 and dstar is None: dstar = tau
        if tau % 2 == 0 or (dstar == tau):
            print(f"{tau:>8} {tau*DT:>6.2f} {a:>6.2f}")
    print(f"\nδ_id (first τ with AUC>0.9) = {dstar} steps = {dstar*DT if dstar is not None else float('nan')}s")
    print("KILL if δ_id ~1-2 steps (instantly separable = not aliased = absorbable).")
