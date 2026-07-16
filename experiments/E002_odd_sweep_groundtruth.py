"""E002 — The ODD-swept family of maximal safe sets: continuous ground truth + visualization.

THE QUESTION E002 SETS UP (E003 answers it)
-------------------------------------------
E001 showed the classical CLF/CBF region is 1.6-3.8x smaller than the maximal set answering
the same question, and that the cause is structural (an ellipse cannot fill a parallelogram).
So the maximal family is what we want. The open question is COMPUTATIONAL:

    Can ONE neural critic represent the whole family {V(.; odd)} as the ODD varies
    continuously -- smoothly, without collapsing to the worst case?

E002 builds the reference that question is scored against: the TRUE family, densely swept,
as a continuous object. E003 overlays a single learned ODD-conditioned critic on it.

THE ODD AXES (--axis)
---------------------
`mass` -- THE HEADLINE (Buzi, 2026-07-16). The pendulum's mass = a PAYLOAD. Visual (the bob
    grows), physical (you pick up a box), and named first in the lab's own humanoid proposal:
    "a humanoid may carry objects with different shapes, masses, centers of mass, and
    inertial properties". Mass also fixes the transition model on physical grounds: payloads
    JUMP (you grab a box, you don't ramp into it) -- so this is the adjacency-graph + dwell
    case, which is exactly the exogenous unpredictable switching Gandhi & Mhaskar model.

    Monotonicity is NOT free here, and the reason is worth knowing. Gravity is mass-INDEPENDENT
    ((g/l) sin theta -- m cancels between torque mgl sin th and inertia ml^2), while control
    scales 1/(ml^2) and disturbance 1/(ml). Net restoring authority is

        (1/(m l)) * (u_max/l - F_bar*|cos theta|)

    so heavier => weaker control against unchanged gravity => smaller safe set. Monotone --
    BUT ONLY while the bracket is positive. If F_bar*|cos theta| > u_max/l the adversary
    out-powers the control, the bracket flips sign, and MORE mass HELPS (it damps the
    adversary faster than it damps you). So mass is monotone on a REGION, not globally: the
    (mass, F_bar) corner is a non-monotone nuisance region, concretely instantiating the
    professor's caveat that Theta = ordered cone x non-monotone nuisance. With the lab's
    config (u_max=20, l=1, F_bar=2) we are far from the flip: 20 >> 2.

`F_bar` -- THE SECOND AXIS. The adversary's power: varying it changes THE GAME, not merely
    the specification -- the regime this project claims is unclaimed (AnySafe is max_a with
    no adversary; Borquez sets beta_dot = 0). It is also what Borquez et al. and Lin et al.
    both condition on, so it is the direct comparison to prior art.

Together they make the ODD a genuine PRODUCT PARTIAL ORDER, which is what actually tests the
monotone architecture (proposal 5.5) -- and their corner is where the regime flip lives.

PHYSICS: Alan et al., "Parameterized Barrier Functions to Guarantee Safety Under
Uncertainty", L-CSS 2023 (Example 1, Eq. 14) -- the same system as the lab's R-CBF pendulum
(safe_adaptation_dev/simulators/dynamics/inverted_pendulum_dynamics.py), so results are
directly comparable to that paper's figure.

    theta_ddot = (g/l) sin(theta) + u/(m l^2) + F/(m l) cos(theta)

NB: PBF's own barrier is ALREADY parameterized by the uncertainty bound (p = F_bar/(m l)) --
it is an analytic, conservative, ellipsoidal ODD-conditioned certificate. This experiment
computes the MAXIMAL ODD-conditioned family instead. Cite PBF; do not claim conditioning.

CONVENTION (matches safety_sb3): g(x) >= 0 iff outside the failure set;
avoid/stay backup V = min(g, max_u min_d V(f(x,u,d))). Two-player (Isaacs), matching what
ISAACS solves -- E001 was single-player.
"""

import argparse
import os
import time

import numpy as np

# --- PBF Example 1 / lab R-CBF pendulum parameters -------------------------------------
G, L = 10.0, 1.0
M_NOMINAL = 2.0                  # kg, bare pendulum (the lab's config)
F_BAR_NOMINAL = 2.0              # N, disturbance bound (the lab's config)
DT = 0.025                       # matches optimized_dp/invpend2d.yaml
THETA_MAX = np.pi / 3            # ~1.047 rad; |theta| beyond this = failed
U_MAX = 20.0                     # torque bound (Nm)

# Control and disturbance both enter the Hamiltonian LINEARLY (the dynamics are affine in
# u and F), so the optimizers are bang-bang. Dense sampling is wasted work: 5 points per
# player brackets the bounds and keeps a mid-point for grid-artifact safety.
N_CTRL, N_DSTB = 5, 5

AXES = {
    # name:   (default range, unit, label)
    "mass":  ((2.0, 8.0), "kg", r"payload: pendulum mass $m$"),
    "F_bar": ((0.0, 6.0), "N", r"adversary power: disturbance bound $\bar F$"),
}


def dynamics(theta, omega, u, F, m):
    """One explicit-Euler step of PBF Eq. 14.

    theta_ddot = (g/l) sin(theta) + u/(m l^2) + F/(m l) cos(theta)

    Note the gravity term has NO mass: m cancels between the gravity torque (m g l sin th)
    and the inertia (m l^2). That asymmetry is the whole reason mass is an interesting ODD
    axis -- it weakens control and disturbance together, against an unchanged destabilizer.
    """
    acc = (G / L) * np.sin(theta) + u / (m * L * L) + (F / (m * L)) * np.cos(theta)
    omega_n = omega + acc * DT
    theta_n = theta + omega_n * DT
    return theta_n, omega_n


class Grid:
    def __init__(self, n_theta=161, n_omega=161, theta_lim=1.5, omega_lim=6.5):
        # Limits match optimized_dp/invpend2d.yaml so the sweep is comparable to the lab's HJI run.
        self.theta = np.linspace(-theta_lim, theta_lim, n_theta)
        self.omega = np.linspace(-omega_lim, omega_lim, n_omega)
        self.TH, self.OM = np.meshgrid(self.theta, self.omega, indexing="ij")
        self.shape = self.TH.shape
        self.dth = self.theta[1] - self.theta[0]
        self.dom = self.omega[1] - self.omega[0]
        self.cell = self.dth * self.dom

    def interp(self, V, tq, oq):
        ti = np.clip((tq - self.theta[0]) / self.dth, 0, len(self.theta) - 1.0001)
        oi = np.clip((oq - self.omega[0]) / self.dom, 0, len(self.omega) - 1.0001)
        t0, o0 = np.floor(ti).astype(int), np.floor(oi).astype(int)
        ft, fo = ti - t0, oi - o0
        return (V[t0, o0] * (1 - ft) * (1 - fo) + V[t0 + 1, o0] * ft * (1 - fo)
                + V[t0, o0 + 1] * (1 - ft) * fo + V[t0 + 1, o0 + 1] * ft * fo)

    def volume(self, mask):
        return float(mask.sum() * self.cell)


def safety_margin(grid):
    """g(x) = theta_max - |theta|, normalized to O(1). Matches the lab's HJI initial value
    (optimized_dp/gen_hji_inverted_pendulum.py::make_initial_value) -- avoid-only, no
    speed term, so the swept family is directly comparable to that figure."""
    return (THETA_MAX - np.abs(grid.TH)) / THETA_MAX


def solve_stay(grid, g, m, F_bar, V_init=None, tol=1e-6, max_iter=4000, patience=60):
    """Maximal robust stay-set: V = min(g, max_u min_d V(f(x,u,d))).

    V_init enables WARM-STARTING along the sweep. The family is smooth in the ODD, so the
    previous rung's solution is a good initializer -- and the iteration count needed to
    re-converge is itself evidence of that smoothness (reported per rung).

    CONVERGENCE IS ON THE SET, NOT THE VALUE. We want the zero-superlevel set; the value
    keeps creeping (linearly, and slowly -- the pendulum is near-marginal, control accel
    u_max/(m l^2) barely exceeds gravity (g/l) at large theta) long after the boundary has
    stopped moving. Stopping on |dV| < 1e-6 cost 1296-4000 iterations per rung and computed
    digits nobody reads. We stop once the MASK is unchanged for `patience` consecutive
    iterations, and still report the value delta so the value-convergence state is visible.
    """
    V = g.copy() if V_init is None else V_init.copy()
    us = np.linspace(-U_MAX, U_MAX, N_CTRL)
    ds = np.linspace(-F_bar, F_bar, N_DSTB) if F_bar > 0 else np.zeros(1)

    # Precompute successor states: they depend on (u, F, m) but NOT on V, so they are
    # constant across the fixed-point iteration. Hoisting this out of the loop is the
    # difference between a ~30 min sweep and a ~3 min one.
    succ = [[dynamics(grid.TH, grid.OM, u, F, m) for F in ds] for u in us]

    mask, stable, delta = V >= 0, 0, np.inf
    for it in range(max_iter):
        # max_u min_d V(f(x,u,d)) -- Isaacs: the adversary responds to the control.
        best_u = np.full(grid.shape, -np.inf)
        for row in succ:
            worst_d = np.full(grid.shape, np.inf)
            for th_n, om_n in row:
                worst_d = np.minimum(worst_d, grid.interp(V, th_n, om_n))
            best_u = np.maximum(best_u, worst_d)
        V_new = np.minimum(g, best_u)
        delta = np.abs(V_new - V).max()
        V = V_new

        new_mask = V >= 0
        stable = stable + 1 if np.array_equal(new_mask, mask) else 0
        mask = new_mask
        if stable >= patience or delta < tol:
            return V, it + 1, delta
    return V, max_iter, delta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", choices=list(AXES), default="mass",
                    help="which ODD parameter to sweep (default: mass — the headline payload axis)")
    ap.add_argument("--out", default=None, help="default: results/E002_<axis>")
    ap.add_argument("--lo", type=float, default=None)
    ap.add_argument("--hi", type=float, default=None)
    ap.add_argument("--n-sweep", type=int, default=61)
    ap.add_argument("--no-warmstart", action="store_true",
                    help="cold-start every rung (slower; use to verify warm-start didn't bias the fixed point)")
    args = ap.parse_args()

    (lo_d, hi_d), unit, label = AXES[args.axis]
    lo = lo_d if args.lo is None else args.lo
    hi = hi_d if args.hi is None else args.hi
    out = args.out or f"results/E002_{args.axis}"
    os.makedirs(out, exist_ok=True)

    grid = Grid()
    g = safety_margin(grid)
    sweep = np.linspace(lo, hi, args.n_sweep)

    def params(v):
        """(mass, F_bar) for a sweep value on the chosen axis; the other axis stays nominal."""
        return (v, F_BAR_NOMINAL) if args.axis == "mass" else (M_NOMINAL, v)

    print(f"grid {grid.shape}  cell {grid.cell:.3e}")
    print(f"ODD axis = {args.axis} in [{lo}, {hi}] {unit} x{args.n_sweep}   "
          f"(held: {'F_bar=' + str(F_BAR_NOMINAL) + ' N' if args.axis == 'mass' else 'm=' + str(M_NOMINAL) + ' kg'}, "
          f"u_max={U_MAX} Nm)")
    print(f"warm-start: {'OFF' if args.no_warmstart else 'ON'}\n")

    Vs, vols, iters = [], [], []
    V_prev, t0 = None, time.time()
    for i, v in enumerate(sweep):
        m, F_bar = params(v)
        V, it, d = solve_stay(grid, g, m, F_bar, V_init=V_prev)
        V_prev = None if args.no_warmstart else V
        vol = grid.volume(V >= 0)
        Vs.append(V.copy()); vols.append(vol); iters.append(it)
        if i % 10 == 0 or i == len(sweep) - 1:
            print(f"  {args.axis}={v:5.2f} {unit}  vol={vol:7.4f}  ({it:3d} it, d={d:.1e})  "
                  f"[{time.time()-t0:5.1f}s]")

    Vs = np.stack(Vs)
    vols = np.array(vols)
    F_sweep = sweep  # keep downstream names

    # --- Monotonicity check. The project's monotone-architecture claim (proposal 5.5) needs
    # the family nested in the ODD parameter: more adversary power => smaller safe set.
    # This is the empirical test of that assumption on the disturbance axis.
    diffs = np.diff(vols)
    monotone = bool((diffs <= 1e-9).all())
    n_viol = int((diffs > 1e-9).sum())

    # Nestedness is the STRONGER claim (volume ordering does not imply set inclusion).
    nested = True
    for i in range(len(F_sweep) - 1):
        if ((Vs[i + 1] >= 0) & ~(Vs[i] >= 0)).any():
            nested = False
            break

    np.savez_compressed(os.path.join(out, "sweep.npz"),
                        theta=grid.theta, omega=grid.omega, sweep=sweep, axis=args.axis,
                        unit=unit, g=g, V=Vs.astype(np.float32), vols=vols,
                        iters=np.array(iters))

    print(f"\nvolume: {vols[0]:.4f} ({args.axis}={sweep[0]:.1f}) -> {vols[-1]:.4f} ({args.axis}={sweep[-1]:.1f})")
    print(f"monotone in {args.axis}: {monotone}" + ("" if monotone else f"  ({n_viol} violations — grid noise, or the regime flip?)"))
    print(f"NESTED (set inclusion, not just volume): {nested}"
          f"   <- the premise the monotone architecture needs")
    print(f"warm-start iterations: first={iters[0]}, median-after-first={int(np.median(iters[1:]))}"
          f"  <- low median = the family is smooth in the ODD")
    print(f"total {time.time()-t0:.1f}s")

    _plot(grid, sweep, Vs, vols, out, args.axis, unit, label)
    print("\nGround truth family ready. E003 overlays a single ODD-conditioned learned critic on this.")


def _draw_pendulum(ax, m, F_bar, axis):
    """Cartoon: rod + bob, bob AREA proportional to mass. Makes the ODD legible at a glance --
    the payload grows and you watch the safe set shrink in the panel beside it."""
    ax.clear()
    th = 0.42                                        # a fixed, tilted pose (purely illustrative)
    x, y = L * np.sin(th), L * np.cos(th)
    ax.plot([0, x], [0, y], color="#333", lw=3, solid_capstyle="round", zorder=2)
    r = 0.085 * np.sqrt(m / M_NOMINAL)               # area ∝ m
    ax.add_patch(plt.Circle((x, y), r, color="#c0392b" if axis == "mass" else "#777",
                            zorder=3, ec="#5a1a12", lw=1.5))
    ax.add_patch(plt.Circle((0, 0), 0.035, color="#333", zorder=4))
    ax.plot([-0.45, 0.45], [0, 0], color="#333", lw=2)
    if F_bar > 0:                                    # disturbance force arrow on the bob
        ax.arrow(x + r + 0.03, y, 0.055 * F_bar / max(F_BAR_NOMINAL, 1e-9), 0,
                 head_width=0.05, head_length=0.04,
                 fc="#2980b9", ec="#2980b9", lw=1.4, zorder=5, length_includes_head=True)
    ax.set_xlim(-0.62, 0.72); ax.set_ylim(-0.16, 1.32)
    ax.set_aspect("equal"); ax.axis("off")
    ax.text(0.5, 0.02, f"m = {m:.2f} kg     $\\bar F$ = {F_bar:.2f} N",
            transform=ax.transAxes, ha="center", fontsize=10, family="monospace")


def _plot(grid, sweep, Vs, vols, out, axis, unit, label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    globals()["plt"] = plt

    def params(v):
        return (v, F_BAR_NOMINAL) if axis == "mass" else (M_NOMINAL, v)

    # --- (1) the sweep as a GIF: what the ODD IS | the true set | the family as a curve
    fig, (axp, ax, axv) = plt.subplots(1, 3, figsize=(14.5, 4.5),
                                       gridspec_kw={"width_ratios": [0.72, 1.25, 1.0]})

    def frame(i):
        m, F_bar = params(sweep[i])
        _draw_pendulum(axp, m, F_bar, axis)
        axp.set_title("the ODD", fontsize=11)

        ax.clear(); axv.clear()
        # every rung ghosted -- makes the nesting visible at a glance
        for j in range(0, len(sweep), 6):
            ax.contour(grid.theta, grid.omega, (Vs[j] >= 0).T, levels=[0.5],
                       colors=["#999"], linewidths=0.5, alpha=0.5)
        ax.contourf(grid.theta, grid.omega, (Vs[i] >= 0).T, levels=[0.5, 1.5],
                    colors=["#2e6b4a"], alpha=0.55)
        ax.contour(grid.theta, grid.omega, (Vs[i] >= 0).T, levels=[0.5],
                   colors=["#1d4430"], linewidths=2.0)
        for s in (-THETA_MAX, THETA_MAX):
            ax.axvline(s, color="#c0392b", ls=":", lw=1.2)
        ax.set_xlim(grid.theta[0], grid.theta[-1]); ax.set_ylim(grid.omega[0], grid.omega[-1])
        ax.set_xlabel(r"$\theta$ (rad)"); ax.set_ylabel(r"$\omega$ (rad/s)")
        ax.set_title("maximal robust safe set", fontsize=11)

        axv.plot(sweep, vols, color="#1d4430", lw=1.8)
        axv.plot(sweep[i], vols[i], "o", color="#c0392b", ms=8)
        axv.set_xlim(sweep[0], sweep[-1])
        axv.set_xlabel(f"{label}  ({unit})")
        axv.set_ylabel("safe-set volume")
        axv.set_title("the whole family, as one curve", fontsize=11)
        axv.grid(alpha=0.25)

        fig.suptitle("E002 — ground truth: the maximal safe set as the ODD sweeps continuously   "
                     "(pendulum, PBF Eq. 14 physics)", fontsize=12.5)
        fig.tight_layout(rect=[0, 0, 1, 0.92])

    anim = FuncAnimation(fig, frame, frames=len(sweep), interval=90)
    p = os.path.join(out, "sweep.gif")
    anim.save(p, writer=PillowWriter(fps=11))
    plt.close(fig)
    print(f"wrote {p}")

    # --- (2) the family as ONE 3D object: stacked zero-level contours over the ODD axis.
    # This is the picture of what a single conditioned critic must represent.
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(9.5, 7.5))
    ax3 = fig.add_subplot(111, projection="3d")
    cmap = plt.get_cmap("viridis")
    scratch = plt.figure()
    for i in range(0, len(sweep), 2):
        cs = scratch.gca().contour(grid.theta, grid.omega, (Vs[i] >= 0).T, levels=[0.5])
        for path in cs.collections[0].get_paths():
            v = path.vertices
            ax3.plot(v[:, 0], v[:, 1], zs=sweep[i], zdir="z",
                     color=cmap(i / max(len(sweep) - 1, 1)), lw=1.1, alpha=0.85)
        scratch.gca().clear()
    plt.close(scratch)
    ax3.set_xlabel(r"$\theta$ (rad)"); ax3.set_ylabel(r"$\omega$ (rad/s)")
    ax3.set_zlabel(f"ODD: {label} ({unit})")
    ax3.set_title("E002 — the ODD-swept family as ONE object\n"
                  "each slice is a maximal safe set; the solid is what a single conditioned critic must learn",
                  fontsize=11)
    ax3.view_init(elev=22, azim=-58)
    p = os.path.join(out, "family_3d.png")
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
