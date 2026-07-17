"""E005 — Oscillating ODD: the safe set as a function of HOW FAST the world changes.

THE MODEL (Buzi, 2026-07-16)
----------------------------
The ODD is not just a set of admissible conditions -- it is a set PLUS an admissible law of
variation. Here the payload oscillates:

    m(phi) = m0 + A*sin(phi),    phi_dot = w   (rad/s),    phi in [0, 2pi)

with (A, w) = bounded MAGNITUDE and FREQUENCY of ODD change. Physically: a sloshing liquid
container, or a suspended swinging load (a crane -- cf. the lab's CLINC).

Why this and not rate-bounded drift: a rate-bounded ADVERSARY parks m at the worst value and
stays there, so infinite-horizon safety collapses to stay(m_max) and the rate bound buys
nothing. A bounded-amplitude bounded-frequency oscillation MUST come back -- the excursion to
heavy is temporary by construction, so the uncertainty class is genuinely smaller. Formally
sine is a subset of rate-bounded drift with rho = A*w, so the gap between them is the value
of knowing the structure.

NB the naming: this is "rate-bounded parameter variation" in LPV / gain-scheduling terms,
where assuming arbitrarily-fast variation forces a single Lyapunov function (quadratic
stability, conservative) and rate bounds permit parameter-dependent ones (less conservative).
Same thesis as this project, stated in the 1990s. Cite LPV; do not call this "meta ODD".

PHASE GOES IN THE STATE, (A, w) ARE THE CONDITIONING PARAMETERS
--------------------------------------------------------------
State (theta, omega, phi); m is a function of phi, so the problem is Markov and autonomous.
The ODD parameters we condition on are (A, w) -- magnitude and frequency of ODD change.
Note: with phi OBSERVED there is no ODD adversary at all (m(t) is then known); only F remains
adversarial. With phi UNOBSERVED the adversary picks it. We compute BOTH, and the gap is the
value of observing the ODD phase -- the estimation story, quantified, with no RL.

THE HYPOTHESIS
--------------
Monotone in A (bigger swings, smaller set) but NON-MONOTONE in w:
  * slow  -> quasi-static: you must survive a long dwell at m0+A  ->  set -> stay(m0+A)
  * fast  -> the pendulum cannot track the oscillation and averages it. What averages is
             <1/m> (control accel is u/(m l^2)), and by Jensen <1/m> >= 1/<m>, so fast
             oscillation gives MORE effective authority than the mean mass suggests.
  * between, near the pendulum's own timescale sqrt(g/l) ~ 3.16 rad/s -> plausibly worst.
If so the safe set is worst at an INTERMEDIATE frequency -- resonance in the ODD -- and the
ODD is monotone in one coordinate and non-monotone in the other: exactly the
"ordered cone x non-monotone nuisance" structure the monotone architecture (proposal 5.5)
must handle. This is a PREDICTION, not an assertion; the pendulum is unstable so "resonance"
is not textbook here. The experiment is cheap, which is the point.

CONVENTION (matches safety_sb3): g >= 0 iff outside the failure set;
avoid/stay backup V = min(g, max_u min_F V(f)).
"""

import argparse
import os
import time

import numpy as np

# --- PBF Example 1 / lab R-CBF pendulum parameters -------------------------------------
G, L = 10.0, 1.0
M0 = 5.0                         # mean mass (kg). With A in [0,3] the swing spans m in [2,8]
                                 # -- exactly E003's swept range, so the two are comparable.
F_BAR = 2.0                      # N, disturbance bound (the lab's config), held fixed
DT = 0.025
THETA_MAX = np.pi / 3
U_MAX = 20.0
N_CTRL, N_DSTB = 5, 5            # Hamiltonian is linear in u and F -> bang-bang; 5 brackets it

W_NATURAL = np.sqrt(G / L)       # ~3.16 rad/s -- the pendulum's own timescale, for reference


def mass(phi, A, m0=M0):
    return m0 + A * np.sin(phi)


class Grid3:
    """(theta, omega, phi) with phi PERIODIC."""

    def __init__(self, n_th=101, n_om=101, n_phi=24, th_lim=1.5, om_lim=6.5):
        self.theta = np.linspace(-th_lim, th_lim, n_th)
        self.omega = np.linspace(-om_lim, om_lim, n_om)
        self.phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)   # periodic: no endpoint
        self.TH, self.OM, self.PH = np.meshgrid(self.theta, self.omega, self.phi, indexing="ij")
        self.shape = self.TH.shape
        self.dth = self.theta[1] - self.theta[0]
        self.dom = self.omega[1] - self.omega[0]
        self.dphi = 2 * np.pi / n_phi
        self.n_phi = n_phi
        self.cell_area = self.dth * self.dom       # area in (theta, omega); phi is not a volume axis

    def interp(self, V, tq, oq, pq):
        """Trilinear; clamped in (theta, omega), WRAPPED in phi.

        Clamping in (theta, omega) is sound: the grid extends past every failure threshold
        (|theta|<=1.5 > THETA_MAX=1.047, |omega|<=6.5), so off-grid states are already deep in
        the failure set where V < 0 -- clamping returns a negative value, the right answer.
        """
        ti = np.clip((tq - self.theta[0]) / self.dth, 0, len(self.theta) - 1.0001)
        oi = np.clip((oq - self.omega[0]) / self.dom, 0, len(self.omega) - 1.0001)
        pi_ = (pq % (2 * np.pi)) / self.dphi
        t0, o0, p0 = np.floor(ti).astype(int), np.floor(oi).astype(int), np.floor(pi_).astype(int)
        ft, fo, fp = ti - t0, oi - o0, pi_ - p0
        p0 = p0 % self.n_phi
        p1 = (p0 + 1) % self.n_phi                 # wrap, don't clamp
        t1, o1 = t0 + 1, o0 + 1

        def bil(pidx):
            return (V[t0, o0, pidx] * (1 - ft) * (1 - fo) + V[t1, o0, pidx] * ft * (1 - fo)
                    + V[t0, o1, pidx] * (1 - ft) * fo + V[t1, o1, pidx] * ft * fo)

        return bil(p0) * (1 - fp) + bil(p1) * fp


def safety_margin(grid):
    """g = (theta_max - |theta|)/theta_max, normalized to O(1). Matches E003 and the lab's HJI
    initial value. Independent of phi -- the ODD changes the DYNAMICS, not the failure set."""
    return (THETA_MAX - np.abs(grid.TH)) / THETA_MAX


def solve_stay(grid, g, A, w, m0=M0, V_init=None, tol=1e-6, max_iter=2000, patience=40):
    """V = min(g, max_u min_F V(f(x,u,F))) on (theta, omega, phi).

    A=0 makes m constant at m0 and phi irrelevant -- that is how the static references are
    computed, with this exact code, so they are comparable to the oscillating runs by
    construction rather than by argument.

    Converges on the SET, not the value (see E003: the pendulum is near-marginal so the value
    creeps linearly long after the boundary stops moving -- 1296 iters vs 170 for the same set).
    """
    V = g.copy() if V_init is None else V_init.copy()
    m = mass(grid.PH, A, m0)                               # mass varies over the phi axis
    us = np.linspace(-U_MAX, U_MAX, N_CTRL)
    ds = np.linspace(-F_BAR, F_BAR, N_DSTB)

    # Successor states depend on (u, F) and the grid but NOT on V -> hoist out of the loop.
    succ = []
    for u in us:
        row = []
        for F in ds:
            acc = (G / L) * np.sin(grid.TH) + u / (m * L * L) + (F / (m * L)) * np.cos(grid.TH)
            om_n = grid.OM + acc * DT
            th_n = grid.TH + om_n * DT
            ph_n = grid.PH + w * DT                        # phi_dot = w, constant
            row.append((th_n, om_n, ph_n))
        succ.append(row)

    mask, stable, delta = V >= 0, 0, np.inf
    for it in range(max_iter):
        best_u = np.full(grid.shape, -np.inf)
        for row in succ:
            worst_d = np.full(grid.shape, np.inf)
            for th_n, om_n, ph_n in row:
                worst_d = np.minimum(worst_d, grid.interp(V, th_n, om_n, ph_n))
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


def measure(grid, V):
    """Two volumes, and their gap is a result in itself.

    observed   -- the filter SEES the phase (it can watch the slosh): mean over phi of the
                  per-phase safe area. This is what an ODD-conditioned filter with a phase
                  observation can certify.
    unobserved -- the filter does NOT see the phase: it must be safe for EVERY phase, so the
                  certifiable set is the intersection over phi. Worst-case over the ODD's
                  hidden state.

    observed - unobserved = the value of observing the ODD phase, with no RL involved.
    """
    per_phase = [(V[:, :, k] >= 0).sum() * grid.cell_area for k in range(grid.n_phi)]
    observed = float(np.mean(per_phase))
    unobserved = float((V.min(axis=2) >= 0).sum() * grid.cell_area)
    return observed, unobserved, np.array(per_phase)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/E005")
    ap.add_argument("--A", type=float, default=3.0,
                    help="oscillation amplitude (kg). 3.0 => m sweeps [2,8], matching E003's range")
    ap.add_argument("--n-w", type=int, default=21, help="frequency sweep points (log-spaced)")
    ap.add_argument("--w-min", type=float, default=0.15)
    ap.add_argument("--w-max", type=float, default=25.0)
    ap.add_argument("--n-th", type=int, default=101)
    ap.add_argument("--n-om", type=int, default=101)
    ap.add_argument("--n-phi", type=int, default=24)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    grid = Grid3(args.n_th, args.n_om, args.n_phi)
    g = safety_margin(grid)
    print(f"grid {grid.shape} = {np.prod(grid.shape):,} pts   "
          f"m(phi) = {M0} + {args.A}*sin(phi)  => m in [{M0-args.A}, {M0+args.A}] kg")
    print(f"pendulum natural rate sqrt(g/l) = {W_NATURAL:.2f} rad/s\n")

    # --- static references: A=0 at the min/mean/max of the swing. Computed with the SAME 3-D
    # code (A=0 makes m constant and phi irrelevant) so they are exactly comparable.
    print("static references (A=0, m constant):")
    refs = {}
    for m_fixed in (M0 - args.A, M0, M0 + args.A):
        V, it, d = solve_stay(grid, g, A=0.0, w=0.0, m0=m_fixed)
        obs, unobs, _ = measure(grid, V)
        refs[m_fixed] = obs
        print(f"   m = {m_fixed:.1f} kg  ->  safe area = {obs:7.4f}   ({it} it)")
    worst, mean_m, best = refs[M0 + args.A], refs[M0], refs[M0 - args.A]

    # --- the frequency sweep
    ws = np.logspace(np.log10(args.w_min), np.log10(args.w_max), args.n_w)
    print(f"\nfrequency sweep: {args.n_w} points, w in [{args.w_min}, {args.w_max}] rad/s (log-spaced)")
    print("\n     w (rad/s) | w/w_nat |  observed |  unobserved | gap (cost of not seeing phase)")
    print("     " + "-" * 74)

    obs_v, unobs_v, Vs, iters = [], [], [], []
    V_prev, t0 = None, time.time()
    for w in ws:
        V, it, d = solve_stay(grid, g, args.A, w, V_init=V_prev)
        V_prev = V
        obs, unobs, _ = measure(grid, V)
        obs_v.append(obs); unobs_v.append(unobs); iters.append(it)
        Vs.append(V.astype(np.float32))
        print(f"     {w:9.3f} | {w/W_NATURAL:7.2f} | {obs:9.4f} | {unobs:11.4f} | "
              f"{obs-unobs:8.4f}   ({it:3d} it) [{time.time()-t0:5.0f}s]")

    obs_v, unobs_v = np.array(obs_v), np.array(unobs_v)
    np.savez_compressed(os.path.join(args.out, "sweep.npz"),
                        theta=grid.theta, omega=grid.omega, phi=grid.phi, w=ws, A=args.A,
                        observed=obs_v, unobserved=unobs_v, V=np.stack(Vs),
                        ref_worst=worst, ref_mean=mean_m, ref_best=best, M0=M0)

    # --- verdict
    i_min = int(np.argmin(obs_v))
    interior = 0 < i_min < len(ws) - 1
    print("\n" + "=" * 78)
    print(f"static refs:  best (m={M0-args.A:.0f}) = {best:.4f}   "
          f"mean (m={M0:.0f}) = {mean_m:.4f}   worst (m={M0+args.A:.0f}) = {worst:.4f}")
    print(f"oscillating:  min = {obs_v[i_min]:.4f} at w = {ws[i_min]:.2f} rad/s "
          f"({ws[i_min]/W_NATURAL:.2f} x natural)   max = {obs_v.max():.4f} at w = {ws[int(np.argmax(obs_v))]:.2f}")
    if interior:
        print("\n>>> NON-MONOTONE IN FREQUENCY: the safe set is worst at an INTERMEDIATE frequency.")
        print("    RESONANCE IN THE ODD. The frequency axis is non-monotone while the amplitude")
        print("    axis is monotone => the ODD is 'ordered cone x non-monotone nuisance' exactly as")
        print("    proposal 5.5 anticipates. Monotone structure must NOT be imposed on this axis.")
    else:
        print("\n>>> MONOTONE IN FREQUENCY (min at an endpoint). The resonance prediction FAILS on")
        print("    this ladder -- record it honestly. Monotone structure may apply to both axes.")
    if obs_v.max() > worst * 1.02:
        print(f"\n    Oscillation beats the static worst case by up to "
              f"{100*(obs_v.max()/worst - 1):.0f}% -- the temporary-excursion argument holds:")
        print("    a bounded oscillation is a strictly smaller uncertainty class than parking at m_max.")
    print("=" * 78)

    _plot(grid, ws, obs_v, unobs_v, Vs, args.A, refs, args.out)


def _plot(grid, ws, obs_v, unobs_v, Vs, A, refs, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    worst, mean_m, best = refs[M0 + A], refs[M0], refs[M0 - A]

    # --- (1) THE MONEY FIGURE: safe area vs frequency of ODD change
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.semilogx(ws, obs_v, "o-", color="#1d4430", lw=2, ms=5, label="oscillating ODD (phase observed)")
    ax.semilogx(ws, unobs_v, "s--", color="#2980b9", lw=1.6, ms=4, alpha=0.85,
                label="oscillating ODD (phase NOT observed)")
    ax.axhline(worst, color="#c0392b", ls=":", lw=1.8,
               label=f"static worst case: $m=${M0+A:.0f} kg (what a fixed-ODD filter must assume)")
    ax.axhline(mean_m, color="#7f8c8d", ls=":", lw=1.4, label=f"static mean: $m=${M0:.0f} kg")
    ax.axhline(best, color="#27ae60", ls=":", lw=1.4, label=f"static best: $m=${M0-A:.0f} kg")
    ax.axvline(W_NATURAL, color="#8e44ad", ls="-.", lw=1.4, alpha=0.8,
               label=r"pendulum natural rate $\sqrt{g/\ell}$")
    ax.set_xlabel(r"frequency of ODD change  $\omega$  (rad/s)  →  faster world")
    ax.set_ylabel(r"maximal safe area in $(\theta,\omega)$")
    ax.set_title(f"E005 — how fast the ODD changes decides how safe you can be\n"
                 f"payload oscillates $m(\\phi)={M0:.0f}+{A:.0f}\\sin\\phi$ kg  "
                 f"(slosh / swinging load)", fontsize=12)
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8.5, loc="best")
    fig.tight_layout()
    p = os.path.join(out, "fig_resonance.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print(f"wrote {p}")

    # --- (2) the set pulsing with phase, at the worst frequency
    i_w = int(np.argmin(obs_v))
    V = Vs[i_w]
    fig, (axp, ax) = plt.subplots(1, 2, figsize=(11, 4.6),
                                  gridspec_kw={"width_ratios": [0.75, 1.25]})

    def frame(k):
        phi = grid.phi[k]
        m = M0 + A * np.sin(phi)
        axp.clear()
        th_draw = 0.42
        x, y = L * np.sin(th_draw), L * np.cos(th_draw)
        axp.plot([0, x], [0, y], color="#333", lw=3, solid_capstyle="round", zorder=2)
        axp.add_patch(plt.Circle((x, y), 0.085 * np.sqrt(m / M0), color="#c0392b",
                                 zorder=3, ec="#5a1a12", lw=1.5))
        axp.add_patch(plt.Circle((0, 0), 0.035, color="#333", zorder=4))
        axp.plot([-0.45, 0.45], [0, 0], color="#333", lw=2)
        axp.set_xlim(-0.62, 0.72); axp.set_ylim(-0.16, 1.32)
        axp.set_aspect("equal"); axp.axis("off")
        axp.set_title(f"m = {m:.2f} kg     $\\phi$ = {phi:.2f}", fontsize=10, family="monospace")

        ax.clear()
        for kk in range(grid.n_phi):
            ax.contour(grid.theta, grid.omega, (V[:, :, kk] >= 0).T, levels=[0.5],
                       colors=["#bbb"], linewidths=0.5, alpha=0.6)
        ax.contourf(grid.theta, grid.omega, (V[:, :, k] >= 0).T, levels=[0.5, 1.5],
                    colors=["#2e6b4a"], alpha=0.55)
        ax.contour(grid.theta, grid.omega, (V[:, :, k] >= 0).T, levels=[0.5],
                   colors=["#1d4430"], linewidths=2.0)
        ax.contour(grid.theta, grid.omega, (V.min(axis=2) >= 0).T, levels=[0.5],
                   colors=["#2980b9"], linewidths=1.8, linestyles="--")
        for s in (-THETA_MAX, THETA_MAX):
            ax.axvline(s, color="#c0392b", ls=":", lw=1.2)
        ax.set_xlim(grid.theta[0], grid.theta[-1]); ax.set_ylim(grid.omega[0], grid.omega[-1])
        ax.set_xlabel(r"$\theta$ (rad)"); ax.set_ylabel(r"$\omega$ (rad/s)")
        ax.set_title("green: safe set at this phase   |   blue dashed: safe for EVERY phase",
                     fontsize=10)
        fig.suptitle(f"E005 — the safe set breathing with the ODD   "
                     f"($\\omega$ = {ws[i_w]:.2f} rad/s, the worst frequency)", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.91])

    anim = FuncAnimation(fig, frame, frames=grid.n_phi, interval=110)
    p = os.path.join(out, "breathing.gif")
    anim.save(p, writer=PillowWriter(fps=9))
    plt.close(fig)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
