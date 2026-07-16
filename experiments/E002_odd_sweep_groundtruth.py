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

THE ODD PARAMETER: F_bar, the disturbance bound
-----------------------------------------------
Not the torque limit (E001's axis). F_bar is the ADVERSARY'S POWER, so varying it changes
THE GAME, not merely the specification -- which is the regime this project claims is
unclaimed (AnySafe is max_a with no adversary; Borquez sets beta_dot = 0). It is also what
Borquez et al. and Lin et al. both condition on, so it is the direct comparison to prior art.

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
G, M, L = 10.0, 2.0, 1.0
DT = 0.025                       # matches optimized_dp/invpend2d.yaml
THETA_MAX = np.pi / 3            # ~1.047 rad; |theta| beyond this = failed
U_MAX = 20.0                     # torque bound (Nm), FIXED -- F_bar is the ODD axis here

N_CTRL, N_DSTB = 21, 9


def dynamics(theta, omega, u, F):
    """One explicit-Euler step of PBF Eq. 14."""
    acc = (G / L) * np.sin(theta) + u / (M * L * L) + (F / (M * L)) * np.cos(theta)
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


def solve_stay(grid, g, F_bar, V_init=None, tol=1e-6, max_iter=4000):
    """Maximal robust stay-set: V = min(g, max_u min_d V(f(x,u,d))).

    V_init enables WARM-STARTING along the sweep. The family is smooth in F_bar, so the
    previous rung's solution is a good initializer -- and the iteration count needed to
    re-converge is itself evidence of that smoothness (reported per rung).
    """
    V = g.copy() if V_init is None else V_init.copy()
    us = np.linspace(-U_MAX, U_MAX, N_CTRL)
    ds = np.linspace(-F_bar, F_bar, N_DSTB) if F_bar > 0 else np.zeros(1)

    for it in range(max_iter):
        # max_u min_d V(f(x,u,d)) -- the adversary responds to the control (Isaacs, not simultaneous).
        best_u = np.full(grid.shape, -np.inf)
        for u in us:
            worst_d = np.full(grid.shape, np.inf)
            for F in ds:
                th_n, om_n = dynamics(grid.TH, grid.OM, u, F)
                worst_d = np.minimum(worst_d, grid.interp(V, th_n, om_n))
            best_u = np.maximum(best_u, worst_d)
        V_new = np.minimum(g, best_u)
        delta = np.abs(V_new - V).max()
        V = V_new
        if delta < tol:
            return V, it + 1, delta
    return V, max_iter, delta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/E002")
    ap.add_argument("--f-min", type=float, default=0.0)
    ap.add_argument("--f-max", type=float, default=6.0)
    ap.add_argument("--n-sweep", type=int, default=61)
    ap.add_argument("--no-warmstart", action="store_true",
                    help="cold-start every rung (slower; use to verify warm-start didn't bias the fixed point)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    grid = Grid()
    g = safety_margin(grid)
    F_sweep = np.linspace(args.f_min, args.f_max, args.n_sweep)

    print(f"grid {grid.shape}  cell {grid.cell:.3e}  |  u_max={U_MAX} FIXED, "
          f"ODD = F_bar in [{args.f_min}, {args.f_max}] x{args.n_sweep}")
    print(f"warm-start: {'OFF' if args.no_warmstart else 'ON'}\n")

    Vs, vols, iters = [], [], []
    V_prev, t0 = None, time.time()
    for i, F_bar in enumerate(F_sweep):
        V, it, d = solve_stay(grid, g, F_bar, V_init=V_prev)
        V_prev = None if args.no_warmstart else V
        vol = grid.volume(V >= 0)
        Vs.append(V.copy()); vols.append(vol); iters.append(it)
        if i % 10 == 0 or i == len(F_sweep) - 1:
            print(f"  F_bar={F_bar:5.2f}  vol={vol:7.4f}  ({it:3d} it, d={d:.1e})  "
                  f"[{time.time()-t0:5.1f}s]")

    Vs = np.stack(Vs)
    vols = np.array(vols)

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

    np.savez_compressed(os.path.join(args.out, "sweep.npz"),
                        theta=grid.theta, omega=grid.omega, F_sweep=F_sweep, g=g,
                        V=Vs.astype(np.float32), vols=vols, iters=np.array(iters))

    print(f"\nvolume: {vols[0]:.4f} (F_bar={F_sweep[0]:.1f}) -> {vols[-1]:.4f} (F_bar={F_sweep[-1]:.1f})")
    print(f"monotone in F_bar: {monotone}" + ("" if monotone else f"  ({n_viol} violations, grid noise?)"))
    print(f"NESTED (set inclusion, not just volume): {nested}")
    print(f"warm-start iterations: first={iters[0]}, median-after-first={int(np.median(iters[1:]))}"
          f"  <- low median = the family is smooth in the ODD")
    print(f"total {time.time()-t0:.1f}s")

    _plot(grid, F_sweep, Vs, vols, args.out)
    print("\nGround truth family ready. E003 overlays a single ODD-conditioned learned critic on this.")


def _plot(grid, F_sweep, Vs, vols, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    # --- (1) the sweep as a GIF: the true safe set morphing as the ODD changes
    fig, (ax, axv) = plt.subplots(1, 2, figsize=(11.5, 4.6),
                                  gridspec_kw={"width_ratios": [1.35, 1]})

    def frame(i):
        ax.clear(); axv.clear()
        ax.contourf(grid.theta, grid.omega, (Vs[i] >= 0).T, levels=[0.5, 1.5],
                    colors=["#2e6b4a"], alpha=0.55)
        ax.contour(grid.theta, grid.omega, (Vs[i] >= 0).T, levels=[0.5],
                   colors=["#1d4430"], linewidths=2.0)
        # every rung, ghosted -- makes the nesting visible
        for j in range(0, len(F_sweep), 6):
            ax.contour(grid.theta, grid.omega, (Vs[j] >= 0).T, levels=[0.5],
                       colors=["#999"], linewidths=0.5, alpha=0.5)
        for s in (-THETA_MAX, THETA_MAX):
            ax.axvline(s, color="#c0392b", ls=":", lw=1.2)
        ax.set_xlim(grid.theta[0], grid.theta[-1]); ax.set_ylim(grid.omega[0], grid.omega[-1])
        ax.set_xlabel(r"$\theta$ (rad)"); ax.set_ylabel(r"$\omega$ (rad/s)")
        ax.set_title(f"maximal robust safe set   $\\bar F$ = {F_sweep[i]:.2f} N", fontsize=11)

        axv.plot(F_sweep, vols, color="#1d4430", lw=1.8)
        axv.plot(F_sweep[i], vols[i], "o", color="#c0392b", ms=8)
        axv.set_xlabel(r"ODD parameter $\bar F$ (N) — adversary power")
        axv.set_ylabel("safe-set volume"); axv.set_title("the family, as one curve", fontsize=11)
        axv.grid(alpha=0.25)
        fig.suptitle("E002 — ground truth: the maximal safe set as the ODD sweeps continuously "
                     "(pendulum, PBF Eq. 14 physics)", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.93])

    anim = FuncAnimation(fig, frame, frames=len(F_sweep), interval=90)
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
    for i in range(0, len(F_sweep), 2):
        cs = plt.figure().gca().contour(grid.theta, grid.omega, (Vs[i] >= 0).T, levels=[0.5])
        for path in cs.collections[0].get_paths():
            v = path.vertices
            ax3.plot(v[:, 0], v[:, 1], zs=F_sweep[i], zdir="z",
                     color=cmap(i / max(len(F_sweep) - 1, 1)), lw=1.1, alpha=0.85)
        plt.close(plt.gcf())
    ax3.set_xlabel(r"$\theta$ (rad)"); ax3.set_ylabel(r"$\omega$ (rad/s)")
    ax3.set_zlabel(r"ODD  $\bar F$ (N)")
    ax3.set_title("E002 — the ODD-swept family as one object\n"
                  "each slice is a maximal safe set; the solid is what ONE conditioned critic must learn",
                  fontsize=11)
    ax3.view_init(elev=22, azim=-58)
    p = os.path.join(out, "family_3d.png")
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
