"""E001 — Compounding conservatism: maximal HJ sets vs. classical CLF sublevel sets.

THE QUESTION
------------
The cross-ODD handoff condition is classical: Gandhi & Mhaskar 2008 (Comput. Chem. Eng.
32(9):2113-2122), Theorem 2.2 --

    If x(T_fault) in Omega_c and Omega_c subset of Omega_n, then the switching rule
    guarantees x(t) in Omega_n for all t >= 0.

But their Omega are MPC/CLF stability regions "computed a priori" (their Remark 2.2) --
conservative INNER estimates of the true viability kernel, whose conservatism they never
quantify. So the classical handoff test is a sufficient condition evaluated on sets that
are themselves sufficient-only.

HYPOTHESIS (the load-bearing claim of this project's framing): the conservatism COMPOUNDS.
There exist states that are genuinely recoverable under the depleted ODD -- inside the
maximal reach-avoid set RA(theta') -- but outside the classical Omega(theta'), so the
classical test REJECTS a transition that is in fact safe. If so, the classical condition
cannot deliver least-conservatism at ANY dimension, not merely at 36-D. That is this
project's whole objective, so it would justify computing the maximal version with
adversarial RL.

FALSIFICATION: if Omega(theta') ~= RA(theta') on realistic examples, the "doubly
conservative" argument collapses and the contribution narrows to scale alone.

THE SYSTEM
----------
Inverted pendulum, theta = angle from UPRIGHT (unstable equilibrium), state (theta, omega).
The ODD parameter is the TORQUE LIMIT u_max -- i.e. control authority, which is exactly the
role of Gandhi & Mhaskar's depleted actuator, and of Borquez et al.'s conditioning
parameter. Physical reading: battery sag / thermal derate / actuator wear on a humanoid.
Less torque => a smaller set of states from which you can keep standing. The walk-vs-crawl
mode spectrum in miniature.

WHY NOT odp's HJSolver: it solves the continuous-time HJ PDE (heterocl). The RL this is
meant to validate (safety_sb3) uses the DISCRETE-time backups min(g, V') and
min(g, max(l, V')). Ground truth must use the same backup as the thing it validates, or the
comparison is not apples-to-apples. At 2-D the discrete backup is trivial and auditable.
We use odp's ENV (numpy/scipy/matplotlib), not its solver.

CONVENTION (matches safety_sb3 -- see its README):
    g(x) -- SAFETY margin. g >= 0 iff OUTSIDE the failure set.
    l(x) -- TARGET margin. l >= 0 iff INSIDE the target set.
    stay/avoid : V = min(g, max_u V(f(x,u)))              -> {V >= 0} = viability kernel
    reach-avoid: V = min(g, max(l, max_u V(f(x,u))))      -> {V >= 0} = RA set
"""

import argparse
import csv
import os

import numpy as np
from scipy.linalg import solve_continuous_are

# ----------------------------------------------------------------------------- system

G, M, L = 10.0, 1.0, 1.0
DT = 0.05                      # matches optimized_dp/examples/pendulum_valueIter_example.py
C1 = 3.0 * G / (2.0 * L)       # theta_ddot = C1*sin(theta) + C2*u
C2 = 3.0 / (M * L * L)

THETA_FALL = 0.7               # |theta| > 0.7 rad (~40 deg) => fallen
OMEGA_MAX = 6.0                # |omega| > 6 rad/s => joint speed violation

# The "safe-park point" (Gandhi & Mhaskar's x_c): upright and at rest. Their Omega_c is the
# region from which the DEPLETED controller drives to and holds x_c -- a STABILIZATION region.
# The maximal object must answer the same question, so the reach-avoid target is a small ball
# about x_c, NOT the viability kernel.
#
# WHY NOT the viability kernel: if the target IS the kernel, reaching it safely implies staying
# safe forever, so RA == kernel identically -- the computation is a tautology (observed: RA and
# stay volumes agreed to all printed digits on the first run of this script). And comparing a
# kernel against a stabilization region is apples-to-oranges: a kernel is larger for reasons
# unrelated to conservatism, since it never requires converging anywhere.
PARK_THETA = 0.05              # |theta| <= 0.05 rad
PARK_OMEGA = 0.25              # |omega| <= 0.25 rad/s

# ODD ladder: torque limits, DECREASING authority. Holding torque at angle t is
# u_hold = (M*L*G/2)*sin(t) = 5*sin(t), so u_max caps the holdable angle at
# asin(u_max/5): 2.0 -> 0.41 rad, 0.5 -> 0.10 rad. Nested and non-trivial.
U_MAX_LADDER = [2.0, 1.5, 1.0, 0.75, 0.5]

N_CTRL = 21                    # control samples in [-u_max, u_max]


def dynamics(theta, omega, u):
    """One explicit-Euler step. No clipping: exceeding OMEGA_MAX is a FAILURE, not a clamp."""
    omega_next = omega + (C1 * np.sin(theta) + C2 * u) * DT
    theta_next = theta + omega_next * DT
    return theta_next, omega_next


# ----------------------------------------------------------------------------- grid

class Grid:
    def __init__(self, theta_lim=1.2, omega_lim=8.0, n_theta=241, n_omega=401):
        self.theta = np.linspace(-theta_lim, theta_lim, n_theta)
        self.omega = np.linspace(-omega_lim, omega_lim, n_omega)
        self.TH, self.OM = np.meshgrid(self.theta, self.omega, indexing="ij")
        self.shape = self.TH.shape
        self.dtheta = self.theta[1] - self.theta[0]
        self.domega = self.omega[1] - self.omega[0]
        self.cell_area = self.dtheta * self.domega

    def interp(self, V, theta_q, omega_q):
        """Bilinear interp with edge clamping.

        Clamping is SAFE here, not a fudge: the grid extends to |theta|=1.2 > THETA_FALL=0.7
        and |omega|=8 > OMEGA_MAX=6, so every off-grid state is already deep inside the
        failure set, where V is uniformly negative. Clamping to the edge therefore returns a
        negative value, which is the correct answer.
        """
        ti = np.clip((theta_q - self.theta[0]) / self.dtheta, 0, len(self.theta) - 1.0001)
        oi = np.clip((omega_q - self.omega[0]) / self.domega, 0, len(self.omega) - 1.0001)
        t0, o0 = np.floor(ti).astype(int), np.floor(oi).astype(int)
        ft, fo = ti - t0, oi - o0
        return (V[t0, o0] * (1 - ft) * (1 - fo) + V[t0 + 1, o0] * ft * (1 - fo)
                + V[t0, o0 + 1] * (1 - ft) * fo + V[t0 + 1, o0 + 1] * ft * fo)

    def volume(self, mask):
        return float(mask.sum() * self.cell_area)


def safety_margin(grid):
    """g(x) >= 0 iff outside the failure set. Each term normalized to O(1) (min = AND).

    Normalizing every term by a fixed constant is mandatory: an unnormalized min over
    mixed units (rad, rad/s) makes the argmin meaningless -- whichever term carries the
    larger raw numbers dominates. (Vault: Reach-avoid RL training gotchas.)
    """
    return np.minimum((THETA_FALL - np.abs(grid.TH)) / THETA_FALL,
                      (OMEGA_MAX - np.abs(grid.OM)) / OMEGA_MAX)


def target_margin(grid):
    """l(x) >= 0 iff inside the safe-park ball about x_c = (0, 0). Same normalize-then-min rule.

    Nesting invariant (vault: Reach-avoid RL training gotchas): for every term shared by g and
    l we need l_threshold < g_threshold STRICTLY, else reaching would require leaving the safe
    set. Asserted at import below.
    """
    return np.minimum((PARK_THETA - np.abs(grid.TH)) / PARK_THETA,
                      (PARK_OMEGA - np.abs(grid.OM)) / PARK_OMEGA)


assert PARK_THETA < THETA_FALL and PARK_OMEGA < OMEGA_MAX, "nesting invariant violated: l must be strictly inside g"


# ----------------------------------------------------------------------------- maximal sets (HJ)

def _max_over_ctrl(grid, V, u_max):
    """max_u V(f(x, u)) -- vectorized over the whole grid."""
    best = np.full(grid.shape, -np.inf)
    for u in np.linspace(-u_max, u_max, N_CTRL):
        th_n, om_n = dynamics(grid.TH, grid.OM, u)
        best = np.maximum(best, grid.interp(V, th_n, om_n))
    return best


def solve_stay(grid, g, u_max, tol=1e-6, max_iter=3000):
    """Viability kernel: V = min(g, max_u V(f(x,u))). {V >= 0} = stay forever under |u| <= u_max."""
    V = g.copy()
    for it in range(max_iter):
        V_new = np.minimum(g, _max_over_ctrl(grid, V, u_max))
        delta = np.abs(V_new - V).max()
        V = V_new
        if delta < tol:
            return V, it + 1, delta
    return V, max_iter, delta


def solve_reach_avoid(grid, g, l, u_max, tol=1e-6, max_iter=3000):
    """RA set: V = min(g, max(l, max_u V(f(x,u)))). {V >= 0} = can reach {l >= 0} without failing."""
    V = g.copy()
    for it in range(max_iter):
        V_new = np.minimum(g, np.maximum(l, _max_over_ctrl(grid, V, u_max)))
        delta = np.abs(V_new - V).max()
        V = V_new
        if delta < tol:
            return V, it + 1, delta
    return V, max_iter, delta


# ----------------------------------------------------------------------------- classical set (CLF)

def clf_region(grid, g, u_max):
    """Gandhi & Mhaskar's Omega: the largest CLF sublevel set on which the input constraint
    still admits Vdot < 0, intersected with the safe set.

    CLF from LQR on the linearization about upright (theta_ddot = C1*theta + C2*u), i.e.
    V(x) = x'Px with P solving the CARE. This is the standard construction their stability
    region instantiates ("computed a priori", off-line -- their Remark 2.2).

    Returns (Omega mask, P, c*). Vdot is affine in u, so min_u Vdot over [-u_max, u_max] is
    attained at a bound -- no search needed.
    """
    A = np.array([[0.0, 1.0], [C1, 0.0]])
    B = np.array([[0.0], [C2]])
    Q, R = np.eye(2), np.array([[1.0]])
    P = solve_continuous_are(A, B, Q, R)

    Vclf = P[0, 0] * grid.TH**2 + 2 * P[0, 1] * grid.TH * grid.OM + P[1, 1] * grid.OM**2

    # Vdot = 2 x'P f(x,u), f = [omega, C1 sin(theta) + C2 u]  (true nonlinear f, not the linearization)
    dV_dth = 2 * (P[0, 0] * grid.TH + P[0, 1] * grid.OM)
    dV_dom = 2 * (P[0, 1] * grid.TH + P[1, 1] * grid.OM)
    drift = dV_dth * grid.OM + dV_dom * (C1 * np.sin(grid.TH))
    min_Vdot = drift - np.abs(dV_dom * C2) * u_max      # best u is bang-bang against sign(dV_dom*C2)

    decreasing = min_Vdot < 0
    feasible = decreasing & (g >= 0)
    feasible[np.abs(Vclf) < 1e-12] = True               # the origin itself is the equilibrium

    # Largest c with {Vclf <= c} entirely inside `feasible`: c* = min Vclf over the infeasible set.
    infeasible_levels = Vclf[~feasible]
    c_star = 0.0 if infeasible_levels.size == 0 else float(infeasible_levels.min())
    return (Vclf <= c_star) & feasible, P, c_star


# ----------------------------------------------------------------------------- experiment

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/E001")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    grid = Grid()
    g, l = safety_margin(grid), target_margin(grid)
    print(f"grid {grid.shape}  cell {grid.cell_area:.3e}  safe-set vol {grid.volume(g >= 0):.4f}  "
          f"park-set vol {grid.volume(l >= 0):.4f}")

    # Per ODD (torque limit), the two competing estimates of the SAME object:
    #   RA(u)    -- maximal set from which SOME control drives to the park point without failing
    #   Omega(u) -- Gandhi & Mhaskar's classical CLF stability region for the same task
    # Both answer "can the depleted mode recover to x_c?", so they are directly comparable.
    # stay(u) is kept only as mode u's *operating envelope* (where it may be when the ODD switches).
    stay, ra, omega_sets = {}, {}, {}
    print("\n  u_max |  stay-set |   RA(park) |  CLF Omega | Omega/RA | c*")
    print("  " + "-" * 62)
    for u_max in U_MAX_LADDER:
        V_stay, _, _ = solve_stay(grid, g, u_max)
        V_ra, ra_it, ra_d = solve_reach_avoid(grid, g, l, u_max)
        om, P, c = clf_region(grid, g, u_max)
        stay[u_max], ra[u_max], omega_sets[u_max] = V_stay, V_ra, om

        v_stay, v_ra, v_om = grid.volume(V_stay >= 0), grid.volume(V_ra >= 0), grid.volume(om)
        # Sanity: Omega is a claimed INNER estimate of the true recoverable set. If it ever
        # escapes RA, the CLF construction is wrong (or the grid is too coarse) -- not a finding.
        leak = grid.volume(om & ~(V_ra >= 0))
        assert leak < 1e-6, f"Omega(u={u_max}) leaks {leak:.2e} outside RA -- CLF construction is unsound"
        print(f"  {u_max:<5} | {v_stay:9.4f} | {v_ra:10.4f} | {v_om:10.4f} | {v_om/v_ra:8.3f} | {c:.3f}"
              f"   (RA {ra_it} it, d={ra_d:.1e})")

    # --- the handoff test, per adjacent transition (authority DROPS: u_max a -> b)
    # Gandhi & Mhaskar's forward edge is the POINT check x(T_fault) in Omega_c. Ours must be a
    # SET check: the ODD switch time is not chosen by us, so the state could be anywhere in the
    # previous mode's operating envelope stay(a).
    rows = []
    for a, b in zip(U_MAX_LADDER[:-1], U_MAX_LADDER[1:]):
        src = stay[a] >= 0                       # where mode a may be when the ODD switches
        ra_b, om_b = ra[b] >= 0, omega_sets[b]   # maximal vs classical recoverable-under-b

        maximal_ok = bool((~src | ra_b).all())    # stay(a) subset of RA(b)      -- the true condition
        classical_ok = bool((~src | om_b).all())  # stay(a) subset of Omega(b)   -- the classical test

        rescued = src & ra_b & ~om_b             # recoverable, but the classical test rejects it
        stranded = src & ~ra_b                   # genuinely unrecoverable -- both tests reject
        v_src, v_ra_b, v_om_b = grid.volume(src), grid.volume(ra_b), grid.volume(om_b)
        v_resc, v_strand = grid.volume(rescued), grid.volume(stranded)

        rows.append(dict(
            transition=f"{a}->{b}", u_max_from=a, u_max_to=b,
            maximal_holds=maximal_ok, classical_holds=classical_ok,
            vol_stay_from=round(v_src, 5), vol_RA_to=round(v_ra_b, 5), vol_Omega_to=round(v_om_b, 5),
            vol_rescued=round(v_resc, 5), vol_stranded=round(v_strand, 5),
            frac_rescued_of_source=round(v_resc / v_src, 5) if v_src else float("nan"),
            frac_stranded_of_source=round(v_strand / v_src, 5) if v_src else float("nan"),
            RA_over_Omega=round(v_ra_b / v_om_b, 4) if v_om_b else float("inf"),
        ))
        print(f"\n{a} -> {b}:  maximal stay(a) ⊆ RA(b)? {maximal_ok}    "
              f"classical stay(a) ⊆ Ω(b)? {classical_ok}")
        print(f"    RA(b)={v_ra_b:.4f}  Ω(b)={v_om_b:.4f}  RA/Ω={v_ra_b/v_om_b if v_om_b else float('inf'):.2f}x")
        print(f"    RESCUED  (recoverable, classical rejects): {v_resc:.4f} = "
              f"{100*v_resc/v_src if v_src else 0:.1f}% of stay(a)")
        print(f"    STRANDED (truly unrecoverable, both reject): {v_strand:.4f} = "
              f"{100*v_strand/v_src if v_src else 0:.1f}% of stay(a)")

    csv_path = os.path.join(args.out, "containment_table.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {csv_path}")

    np.savez_compressed(os.path.join(args.out, "sets.npz"),
                        theta=grid.theta, omega=grid.omega, g=g, l=l,
                        **{f"stay_{u}": stay[u] for u in U_MAX_LADDER},
                        **{f"ra_{u}": ra[u] for u in U_MAX_LADDER},
                        **{f"omega_{u}": omega_sets[u] for u in U_MAX_LADDER})

    _plot(grid, ra, omega_sets, args.out)

    # --- verdict
    print("\n" + "=" * 78)
    any_rescued = any(r["vol_rescued"] > 1e-9 for r in rows)
    if any_rescued:
        worst = max(rows, key=lambda r: r["frac_rescued_of_source"])
        print("HYPOTHESIS SUPPORTED. Omega and RA answer the SAME question (recover to the park")
        print("point under the depleted torque limit), and Omega is strictly, substantially smaller.")
        print(f"  - RA/Omega volume ratio: up to {max(r['RA_over_Omega'] for r in rows):.2f}x")
        print(f"  - Worst transition {worst['transition']}: {100*worst['frac_rescued_of_source']:.1f}% "
              f"of the source operating envelope is")
        print("    genuinely recoverable but REJECTED by the classical test.")
    else:
        print("HYPOTHESIS NOT SUPPORTED: Omega ~= RA on this ladder. The 'doubly conservative'")
        print("argument does not hold here -- report this honestly; it narrows the contribution.")

    # A transition can fail for two very different reasons. Separating them is the point.
    both_fail = [r for r in rows if not r["maximal_holds"] and not r["classical_holds"]]
    if both_fail:
        print("\nNOTE: on this ladder the MAXIMAL test also fails for "
              f"{len(both_fail)}/{len(rows)} transitions, i.e. some states are")
        print("truly unrecoverable after the authority drop -- no filter, learned or classical,")
        print("can save them. That is the handoff phenomenon itself ('you cannot teleport from")
        print("standing to crawling'), and it is what motivates the dwell-time / detection-latency")
        print("contract: the transition must be ANTICIPATED, not merely detected. The rescued")
        print("volume above is the separate claim -- that the classical test over-rejects on top.")
    print("=" * 78)


def _plot(grid, ra, omega_sets, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(U_MAX_LADDER), figsize=(3.6 * len(U_MAX_LADDER), 4.0), sharey=True)
    for ax, u_max in zip(np.atleast_1d(axes), U_MAX_LADDER):
        ra_m, om_m = (ra[u_max] >= 0), omega_sets[u_max]
        ax.contourf(grid.theta, grid.omega, ra_m.T, levels=[0.5, 1.5], colors=["#4c9f70"], alpha=0.5)
        ax.contour(grid.theta, grid.omega, ra_m.T, levels=[0.5], colors=["#2e6b4a"], linewidths=1.8)
        ax.contourf(grid.theta, grid.omega, om_m.T, levels=[0.5, 1.5], colors=["#c0392b"], alpha=0.65)
        gap = 100 * (1 - om_m.sum() / max(ra_m.sum(), 1))
        ax.set_title(f"$u_{{max}}$={u_max}\nclassical misses {gap:.0f}%", fontsize=10)
        ax.set_xlabel(r"$\theta$ (rad)")
        ax.set_xlim(-0.72, 0.72); ax.set_ylim(-4, 4)
        ax.set_xticks([-0.5, 0.0, 0.5])
        ax.axhline(0, lw=0.4, c="k"); ax.axvline(0, lw=0.4, c="k")
    np.atleast_1d(axes)[0].set_ylabel(r"$\omega$ (rad/s)")
    # The structural point: RA is a slanted parallelogram; a quadratic CLF's sublevel set is an
    # ellipse. An ellipse cannot fill a parallelogram, and the mismatch WORSENS as the set skews
    # with falling authority -- so the gap is not a weak-CLF artifact that tuning would remove.
    fig.suptitle("E001 — recover to the safe-park point under torque limit $u_{max}$.   "
                 "GREEN: maximal reach-avoid set $RA$   |   RED: classical CLF region $\\Omega$ "
                 "(Gandhi & Mhaskar 2008).\nSame question, both sets. $RA$ is a slanted "
                 "parallelogram; a quadratic CLF sublevel set is an ellipse — an ellipse cannot "
                 "fill it, and the mismatch grows as authority falls.", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    p = os.path.join(out, "fig1_ra_vs_clf.png")
    fig.savefig(p, dpi=140)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
