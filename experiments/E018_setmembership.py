"""E018 — SET-MEMBERSHIP belief estimator B̂ over the unobserved payload ODD θ=(rigidity, total_mass).

The deductive core the guarantee story rests on: maintain the SET B̂ of θ CONSISTENT with the observed
(state, action, next-state) transitions under the known model class + a bounded-noise tolerance ε — NOT a
point estimate. Two properties this POC demonstrates on the Go2+payload sim:

  (1) SHRINK — for an in-distribution true θ*, B̂ contracts toward θ* as evidence accumulates and CONTAINS
      θ* (soundness — the truth is never excluded). Report the shrink-time δ* (t at which |B̂| first drops
      below 10% of the grid).
  (2) OOD → EMPTY — for a true θ* OUTSIDE the candidate grid+margin (heavy 12 kg, beyond the training box
      ≤7.5 kg), NO candidate explains the data ⇒ B̂ goes EMPTY. That empty-set event IS the OOD flag — the
      key differentiator from a point estimate (which would silently snap to the nearest wrong θ).

METHOD (trajectory-matching multiple-model set-membership — no per-step state injection):
  * ONE mjlab env holds every candidate θ (a grid over rigidity×mass spanning the training box + a margin),
    each REPLICATED R times, in a single batched rollout. Each env's payload is written directly to the
    per-env model (jnt_stiffness / body_mass — exactly what the training DR events write), RE-APPLIED every
    step so an auto-reset can never corrupt the grid. The "observed" trajectory is: for an IN-DIST θ* (on
    the grid) the θ*-node's own replica group (⇒ the truth's residual is 0 by construction — soundness is
    exact); for an OOD θ* (off-grid) a separate true env carrying θ* (mass 12 kg).
  * All envs are forced to an IDENTICAL canonical initial state (nominal stand, upright, zero vel) so only
    θ differs, then driven by an IDENTICAL scripted excitation (a fast ±y square-wave shake via the
    adversary wrench channel — policy-independent). Divergence of the base response then isolates θ.
  * GPU contact-solver nondeterminism makes even two identical-θ envs diverge chaotically under the shake
    (verified: ~0.07 rad tilt spread over the window). We denoise by AVERAGING the observable over the R
    replicas of each θ: the systematic payload signature survives, the nondeterministic jitter drops ~√R.
  * Observable y = [projected_gravity_b(3), base height(1), base lin/ang-vel(3+3), base lateral
    displacement(2)] — the base signals a hidden payload is felt through. Consistency (cumulative to t):
    candidate i ∈ B̂_t iff max_{s≤t} ‖ȳ_pred_s(i) − ȳ_obs_s‖_W ≤ ε, W = per-channel ensemble-spread norm.

Run LOCALLY: MUJOCO_GL=egl PYTHONPATH=external/robot-safety-sandbox \
             ~/miniconda3/envs/mjlab/bin/python experiments/E018_setmembership.py
"""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import torch as th
sys.path.insert(0, "external/robot-safety-sandbox")
import mujoco
from robot_safety_sandbox import make_tensor, spec

# ── CONFIG (grid + estimator constants) ──────────────────────────────────────────────────────────
DEV = "cuda:0"
TASK = "go2_payload_stabilize"      # blind env (no θ obs); driven open-loop, policy-free
N_RIG, N_MASS = 15, 15              # candidate grid → N_c = 225
RIG_RANGE = (0.0, 320.0)            # rigidity axis: training [0,300] + margin
MASS_RANGE = (1.2, 8.5)            # total-mass axis: training [1.2,7.5] + margin
N_REP = 32                          # replicas per θ (denoise GPU nondeterminism ~√R); (N_c+1)·32 fits 12 GB
HORIZON = 16                        # excitation steps (0.32 s @ 50 Hz) — no falls in this window
FORCE_SCALE = 1.0                   # ×force_max(50 N) = 50 N lateral shake
SHAKE_PERIOD = 2                    # steps per ±y half-cycle (fast reversal ⇒ slosh excitation, no drift-topple)
EPS = 0.22                          # noise tolerance (ensemble-spread units). In-dist truth residual is 0
                                    #   by construction (node-as-observed), so ε only sets shrink tightness;
                                    #   ε=0.22 < every OOD closest-approach (≥0.60 over runs) ⇒ OOD empties.
SHRINK_FRAC = 0.10                  # δ* = first t with |B̂| < SHRINK_FRAC·N_c
INIT_Z = 0.32                       # nominal Go2 stand height (assets_go2 INIT_STATE)
RESULTS = "results/E018"

# true θ* points: (label, stiffness, total_mass, grid_ref | None)
TRUE_POINTS = [
    ("IN-DIST  mid",         None, None, (8, 6)),    # exact grid node → truth residual 0 by construction
    ("IN-DIST  light-rigid", None, None, (12, 2)),
    ("OOD      heavy-12kg",   0.0, 12.0, None),      # mass beyond grid+margin → EMPTY
]


def payload_idx(mj):
    """Global model column indices of the 4 payload hinge joints and 5 payload bodies (by name)."""
    m = mj.sim.mj_model
    jids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"robot/payload_j{i}") for i in range(1, 5)]
    bnames = ["robot/payload_mount"] + [f"robot/payload_b{i}" for i in range(1, 5)]
    bids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in bnames]
    j = th.tensor([i for i in jids if i >= 0], device=DEV, dtype=th.long)
    b = th.tensor([i for i in bids if i >= 0], device=DEV, dtype=th.long)
    return j, b


def build_grid():
    """Candidate θ grid (N_c,2) = (rigidity, mass), flattened row-major (rigidity outer, mass inner)."""
    rg = th.linspace(*RIG_RANGE, N_RIG, device=DEV)
    mg = th.linspace(*MASS_RANGE, N_MASS, device=DEV)
    R, M = th.meshgrid(rg, mg, indexing="ij")
    return th.stack([R.reshape(-1), M.reshape(-1)], dim=1)          # (N_c, 2)


def observe(d, pos0):
    """Base observable through which the hidden payload is felt (N,12). x,y ABSOLUTE position is env-origin
    dependent ⇒ only its DISPLACEMENT from t=0 (origin-invariant) and height/attitude/velocity are used."""
    return th.cat([d.projected_gravity_b, d.root_link_pos_w[:, 2:3], d.root_link_lin_vel_w,
                   d.root_link_ang_vel_w, d.root_link_pos_w[:, :2] - pos0], dim=1)


def run_estimator(true_stiff, true_mass, grid_ref):
    """Batched multiple-model set-membership rollout for one true θ*. Returns per-t |B̂| + diagnostics."""
    grid = build_grid()                                             # (N_c, 2)
    n_c = grid.shape[0]
    # In-dist: θ* lies ON the grid, so the true model is IN the candidate set — the node's own replica
    # group serves as the "observed" reference ⇒ its residual is 0 by construction (soundness is exact,
    # not luck-of-the-noise). OOD: θ* is off-grid, so a separate "true" env carries it (index n_c).
    if grid_ref is not None:
        node = grid_ref[0] * N_MASS + grid_ref[1]
        true_stiff, true_mass = float(grid[node, 0]), float(grid[node, 1])
        theta = grid                                                # observed = candidate group `node`
        obs_idx, n_theta = node, n_c
    else:
        theta = th.cat([grid, th.tensor([[true_stiff, true_mass]], device=DEV)], 0)   # last = true θ*
        obs_idx, n_theta = n_c, n_c + 1
    n = n_theta * N_REP
    stiff = theta[:, 0].repeat_interleave(N_REP)                     # (n,)
    mass = theta[:, 1].repeat_interleave(N_REP)

    env = make_tensor(TASK, n, DEV, adversary=True)
    mj, robot = env.mj, env.mj.scene["robot"]
    jidx, bidx = payload_idx(mj)
    env.reset()

    def apply_theta():
        mj.sim.model.jnt_stiffness[:, jidx] = stiff[:, None]        # hinge spring = rigidity
        mj.sim.model.body_mass[:, bidx] = (mass / bidx.numel())[:, None]   # uniform split over 5 blocks

    apply_theta()
    # identical canonical init: default stand, upright, zero vel (x,y kept per-env origin — irrelevant)
    d = robot.data
    jp = d.default_joint_pos.clone()
    robot.write_joint_state_to_sim(jp, th.zeros_like(jp))
    root = th.zeros(n, 13, device=DEV)
    root[:, :2] = d.root_link_pos_w[:, :2]; root[:, 2] = INIT_Z; root[:, 3] = 1.0
    robot.write_root_state_to_sim(root)

    dd = spec(TASK).dstb_dim
    env.force_scale = FORCE_SCALE * th.ones(n, device=DEV)
    ctrl = th.zeros(n, 12, device=DEV)                              # zero ctrl = hold nominal stand (policy-free)
    py = th.tensor([0.0, 1.0, 0.0], device=DEV)

    pos0, Y, falls = None, [], 0
    for t in range(HORIZON):
        apply_theta()                                              # re-assert grid every step (reset-proof)
        dir_t = py if (t // SHAKE_PERIOD) % 2 == 0 else -py         # ±y square-wave shake
        _, _, term, _, _ = env.step_tensor(th.cat([ctrl, dir_t[None].expand(n, dd).contiguous()], 1))
        falls += int(term.sum())
        if pos0 is None:
            pos0 = d.root_link_pos_w[:, :2].clone()
        Y.append(observe(d, pos0).clone())
    env.close()

    D = Y[0].shape[1]
    # group-average over replicas → (T,n_theta,D); NaN-aware so a lone diverged replica (a heavy payload
    # can blow up under the `dr.body_mass`-keeps-inertia approximation) doesn't poison its θ-group mean.
    Y = th.nanmean(th.stack(Y).reshape(HORIZON, n_theta, N_REP, D), dim=2)
    y_obs = Y[:, obs_idx]                                            # (T, D) the true θ* response
    y_cand = Y[:, :n_c]                                              # (T, N_c, D) — candidate grid
    scale = y_cand.reshape(-1, D).std(0).clamp_min(1e-6)            # per-channel ensemble spread
    resid = (((y_cand - y_obs[:, None]) / scale).pow(2).mean(-1)).sqrt()      # (T, N_c)
    cum = resid.cummax(0).values                                    # cumulative-to-t worst residual
    inB = (cum <= EPS) & th.isfinite(cum)                          # NaN (a diverged replica) ⇒ excluded
    sizeB = inB.sum(1)                                              # (T,) |B̂_t|
    # best full-window fit any candidate achieves (nan-safe) — how far the truth is from the model class
    closest = float(th.nan_to_num(cum[-1], nan=float("inf")).min())

    nn = int(((grid - th.tensor([true_stiff, true_mass], device=DEV))
              / th.tensor([RIG_RANGE[1], MASS_RANGE[1]], device=DEV)).pow(2).sum(1).argmin())
    final = inB[-1]
    if int(final.sum()) > 0:
        surv = grid[final]; centroid = surv.mean(0); extent = surv.max(0).values - surv.min(0).values
    else:
        centroid = extent = th.full((2,), float("nan"), device=DEV)
    return dict(true=(true_stiff, true_mass), sizeB=sizeB.cpu().numpy(), n_c=n_c, falls=falls,
                floor=float(cum[-1, nn]), closest=closest,
                truth_in=bool(inB[-1, nn]), in_grid=grid_ref is not None,
                centroid=centroid.cpu().numpy(), extent=extent.cpu().numpy(),
                grid=grid.cpu().numpy(), inB_final=final.cpu().numpy())


def sparkline(sizeB, n_c):
    b = "▁▂▃▄▅▆▇█"
    return "".join(b[min(7, int(f * 8))] for f in sizeB / n_c)


if __name__ == "__main__":
    os.makedirs(RESULTS, exist_ok=True)
    print("E018 — set-membership belief B̂ over the payload ODD θ=(rigidity, total_mass).")
    print(f"grid {N_RIG}×{N_MASS}={N_RIG*N_MASS} | rigidity{RIG_RANGE} × mass{MASS_RANGE} | {N_REP} replicas/θ")
    print(f"excite {HORIZON} steps (0.02 s) ±y {FORCE_SCALE*50:.0f} N period-{SHAKE_PERIOD} | ε={EPS} (ens-std)\n")
    dump = {}
    for label, ks, ms, ref in TRUE_POINTS:
        r = run_estimator(ks, ms, ref)
        n_c, s = r["n_c"], r["sizeB"]
        d_star = next((t for t, v in enumerate(s) if v < SHRINK_FRAC * n_c), None)
        idx = list(range(0, HORIZON, 2)) + ([HORIZON - 1] if (HORIZON - 1) % 2 else [])
        print(f"── {label}   θ*=(k={r['true'][0]:.0f}, m={r['true'][1]:.2f})   falls={r['falls']}")
        print(f"   |B̂| vs t (of {n_c}): {sparkline(s, n_c)}")
        print("   t   : " + " ".join(f"{t:4d}" for t in idx))
        print("   |B̂|: " + " ".join(f"{int(s[t]):4d}" for t in idx))
        if r["in_grid"]:
            ds = None if d_star is None else f"{d_star} steps ({d_star*0.02:.2f} s)"
            print(f"   soundness: truth ∈ B̂ = {r['truth_in']}  (floor residual {r['floor']:.3f} ≤ ε={EPS})")
            print(f"   shrink-time δ* = {ds or '—'}  (|B̂| < {int(SHRINK_FRAC*n_c)});  final |B̂| = {int(s[-1])}")
            print(f"   surviving set: centroid (k={r['centroid'][0]:.0f}, m={r['centroid'][1]:.2f})  "
                  f"extent (Δk={r['extent'][0]:.0f}, Δm={r['extent'][1]:.2f})")
        else:
            empty = int(s[-1]) == 0
            te = next((t for t, v in enumerate(s) if v == 0), None)
            print(f"   OOD flag: B̂ EMPTY at final t = {empty}   (closest any θ ever got: {r['closest']:.3f} > ε={EPS})"
                  + (f"; first empty at t={te} ({te*0.02:.2f} s)" if te is not None else ""))
            print("   >>> OOD DETECTED (empty consistency set) — no θ explains the data." if empty
                  else f"   >>> NOT empty (|B̂|={int(s[-1])}) — widen grid margin or tighten ε.")
        print()
        dump[label] = dict(sizeB=s, floor=r["floor"], truth_in=r["truth_in"], delta_star=d_star,
                           in_grid=r["in_grid"], centroid=r["centroid"], extent=r["extent"],
                           inB_final=r["inB_final"], true=r["true"])
    np.savez(f"{RESULTS}/setmembership.npz", grid=r["grid"], eps=EPS, n_rep=N_REP, horizon=HORIZON,
             force=FORCE_SCALE, **{f"{k}__{kk}": vv for k, d in dump.items() for kk, vv in d.items()})
    print(f"saved {RESULTS}/setmembership.npz")

    try:  # figure: |B̂|/N_c vs t (all points) + the surviving belief set on the θ-grid (2 in-dist points)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        grid = r["grid"]
        fig, ax = plt.subplots(1, 3, figsize=(13, 4))
        for lab, d in dump.items():
            ax[0].plot(np.arange(HORIZON) * 0.02, d["sizeB"] / N_RIG / N_MASS * 100, marker="o", ms=3, label=lab.strip())
        ax[0].axhline(SHRINK_FRAC * 100, ls="--", c="gray", lw=0.8, label=f"{int(SHRINK_FRAC*100)}% (δ*)")
        ax[0].set(xlabel="time (s)", ylabel="|B̂| (% of grid)", title=f"belief-set size vs evidence (ε={EPS})")
        ax[0].legend(fontsize=7); ax[0].grid(alpha=0.3)
        for a, (lab, d) in zip(ax[1:], [(k, v) for k, v in dump.items() if v["in_grid"]][:2]):
            a.scatter(grid[:, 0], grid[:, 1], s=8, c="#ddd", label="candidates")
            m = d["inB_final"].astype(bool)
            a.scatter(grid[m, 0], grid[m, 1], s=18, c="#1f77b4", label="B̂ (final)")
            a.scatter([d["true"][0]], [d["true"][1]], marker="*", s=260, c="#d62728",
                      edgecolor="white", linewidth=1.2, zorder=10, label="θ* (truth)")
            a.set(xlabel="rigidity (hinge stiffness)", ylabel="total mass (kg)", title=lab.strip(),
                  xlim=(-20, 340), ylim=(0.8, 8.9))
            a.legend(fontsize=7, loc="upper left")
        fig.tight_layout(); fig.savefig(f"{RESULTS}/setmembership.png", dpi=120)
        print(f"saved {RESULTS}/setmembership.png")
    except Exception as e:
        print(f"[plot skipped: {e}]")
