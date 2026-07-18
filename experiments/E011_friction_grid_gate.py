"""E011 v6 — the RL-FREE grid gate for the friction-circle bicycle. Gates ALL downstream RL.

Professor (consult-bicycle-task-design, 2026-07-18): v1–v5 failed volume-sensitivity NOT because
friction is a strategy-only knob (that diagnosis was WRONG) but because all five sat in a regime
where mu is dynamically INERT. Friction IS the canonical car FEASIBILITY knob: stopping distance
v^2/(2 mu G) and min turn radius v^2/(mu G) both scale 1/mu. The three v1–v5 errors:
  (a) STEERING-LIMITED, not friction-limited: at v=1.6, mu*G/v^2 = 3.83*mu > KAPPA_MAX=1.42 for all
      mu>0.37, so turning was identical across mu in [0.37,1.0] — the steering limit bound, not mu.
  (b) The v_min CLAMP deleted the volume axis: the mu-doomed set {v^2 >~ 2 mu G * room} has measure
      ~ v_max^3 / mu; clamping v in [1.6,2.0] compressed kinetic energy to 1.56x. VOLUME LIVES IN
      THE FAST STATES — slow states being safe at every mu is fine (it gives nesting).
  (c) under-resolved thin shell (0.5 m envelope ~ 3 cells).

FIX — keep mu as the SINGLE ODD; fix the REGIME. Design law: Pi(mu) = v_max^2 / (2 mu G R) (kinetic
energy / dissipation room, R = clearance). Need Pi(mu_hi) <~ 1 and Pi(mu_lo) >~ 3. Friction-limited
cruise needs v >~ 2.7 m/s. Geometry = "fast-approach dead-end with a tight exit": a committed
corridor (v up to ~5, NO v_min) ending in a DEAD-END wall, with one side-exit gap whose turn radius
r_gap ~ 1.2 m sits ABOVE the steering-limit radius 0.70 m, so FRICTION binds (not geometry).

Gate (four checks, replaces the old three):
  1. SENSITIVITY   — safe-set volume varies >=3x across mu  (+ grid-convergence of the ratio).
  2. NESTING       — safe sets nest in mu (structural: U(mu) grows with mu; a failure is a numerics bug).
  3. TURN-SIGN FLIP— positive-measure region where the optimal-curvature SIGN flips with mu (bonus).
  4. BRAKE-vs-THREAD FLIP — positive-measure region where the optimal action switches between
                     BRAKE (hard a_long<0, straight) and THREAD (near-KAPPA_MAX turn) as mu changes.
                     The brake/swerve boundary speed v*(mu)=2*sqrt(2 d mu G) tracks sqrt(mu); threading
                     the gap is feasible iff v <= sqrt(mu G r_gap) — an ANALYTIC prediction to validate
                     the grid against (v1–v5 had none).

Why 4-D [x,y,psi,v]: constant-speed Dubins DELETES the brake mode; v is load-bearing — the friction
circle couples brake and turn, and that coupling IS the bifurcation. Dynamics (psidot=v*kappa, no
v=0 turn singularity):  xdot=v cos psi   ydot=v sin psi   psidot=v*kappa   vdot=a_long.

Reach-avoid (single-player, control MAXIMIZES — no adversary yet):
    V(s) = min( g(s), max( l(s), max_u V(f(s,u)) ) )     discrete-time, anchor min(l,g).
"""
import argparse
import os

import numpy as np

# --- physical constants (SI-ish) ---------------------------------------------------------------
G = 9.8
V_MIN, V_MAX = 0.0, 8.0      # 🔑 v6: NO v_min, and HIGH v_max. Volume lives in the FAST states;
                             # friction binds on FEASIBILITY (stopping/turning room) precisely at high
                             # v. v6.0 (v_max=5, L=5.4) gave Pi_brake(0.3)=0.79<1 => stopping always
                             # fit => FLAT volume. v6.1: v_max=8 + SHORT corridor (L~3.4) => cliff.
                             # KEY LAW: every friction limit scales sqrt(mu) (stop v<=sqrt(2 mu G room),
                             # turn v<=sqrt(mu G r), wall v<=sqrt(2 mu G w)) => the safe-area RATIO is
                             # bounded by sqrt(mu_hi/mu_lo). For mu in [0.3,1.0] that ceiling is 1.8x
                             # (< 3x!). So sweep the PHYSICALLY HONEST ice->dry range mu in [~0.12,1.0]:
                             # sqrt(1/0.12)=2.9x. This is the real reason v1–v5 (mu>=0.3) capped low.
DT = 0.05
WHEELBASE, DELTA_LIM = 0.257, 0.35
KAPPA_MAX = np.tan(DELTA_LIM) / WHEELBASE          # ~1.42 /m  (min STEERING radius ~0.70 m)
CAR_HALF = 0.16
R_GAP = 1.20                                       # target exit-turn radius (> 0.70 => FRICTION binds)
# Regime dials (R ~ corridor length to dead-end / wall clearance):
#   stopping dist v^2/(2 mu G): at v=5 -> 4.25 m (mu=0.3) vs 1.28 m (mu=1.0)  => fast states doom w/ mu.
#   thread-at-speed feasible iff v <= sqrt(mu G r_gap): 1.88 m/s (mu=0.3) vs 3.43 m/s (mu=1.0).
#   => band v in [1.9,3.4] FLIPS thread-vs-brake with mu; that is gate 4.


def _hwall(y, x0, x1, r=0.28, step=0.26):
    return [[x, y, r] for x in np.arange(x0, x1 + 1e-9, step)]


def _vwall(x, y0, y1, r=0.28, step=0.26):
    return [[x, y, r] for y in np.arange(y0, y1 + 1e-9, step)]


# --- v6 geometry: fast-approach dead-end corridor with ONE tight side exit ----------------------
# Corridor along +x, half-width ~1.0 (walls y=+-1.0). Bottom wall SOLID (no lower escape). Top wall
# ends at x=3.3 -> the ONLY exit up is x in [3.3, dead-end]. Dead-end wall at x=5.4 blocks straight.
# Goal sits up-and-forward of the gap. A fast car must THREAD the up-turn (needs v <= sqrt(mu G r_gap))
# or BRAKE before the dead-end (needs stopping room v^2/(2 mu G)); low mu loses BOTH => volume shrinks.
# v6.1: SHORT corridor (L~3.4 spawn->dead-end) so Pi_brake(mu_lo) >= 3 (cliff regime).
GOAL = (2.70, 2.45, 0.45)
X_DEAD = 3.60
X_TOPEND = 1.55
OBSTACLES = np.array(
    _hwall(-1.00, -0.4, X_DEAD) +                  # bottom wall — SOLID full length (no escape down)
    _hwall(+1.00, -0.4, X_TOPEND) +                # top wall ENDS at x=1.55 -> the up-exit opens
    _vwall(X_DEAD, -1.0, 1.0) +                     # DEAD-END wall: straight-ahead = death
    _vwall(X_DEAD, 1.0, 2.9, r=0.30) +             # right boundary above the gap (goal is to its left)
    [[1.30, 2.75, 0.34], [1.70, 2.9, 0.30]],       # short lip left of the goal (keeps the exit a real turn)
    dtype=np.float64)


class Grid4:
    def __init__(self, nx=55, ny=41, npsi=24, nv=21,
                 xlim=(-0.6, 4.2), ylim=(-1.4, 3.1), vlim=(V_MIN, V_MAX)):
        self.x = np.linspace(*xlim, nx)
        self.y = np.linspace(*ylim, ny)
        self.psi = np.linspace(-np.pi, np.pi, npsi, endpoint=False)   # periodic
        self.v = np.linspace(*vlim, nv)
        self.shape = (nx, ny, npsi, nv)
        self.X, self.Y, self.PS, self.V = np.meshgrid(self.x, self.y, self.psi, self.v, indexing="ij")
        self.dx = self.x[1] - self.x[0]; self.dy = self.y[1] - self.y[0]
        self.dpsi = 2 * np.pi / npsi; self.dv = self.v[1] - self.v[0]
        self.cell_xy = self.dx * self.dy

    def interp(self, Vg, xq, yq, pq, vq):
        """Vectorized 4-D linear interp. Clamp x,y,v (off-arena = failed, V<0); WRAP psi."""
        def idx(q, ax, d, n, wrap=False):
            f = (q - ax[0]) / d
            if wrap:
                f = f % n
                i0 = np.floor(f).astype(int) % n
                return i0, (i0 + 1) % n, f - np.floor(f)
            f = np.clip(f, 0, n - 1.0001)
            i0 = np.floor(f).astype(int)
            return i0, i0 + 1, f - i0
        ix0, ix1, fx = idx(xq, self.x, self.dx, len(self.x))
        iy0, iy1, fy = idx(yq, self.y, self.dy, len(self.y))
        ip0, ip1, fp = idx(pq, self.psi, self.dpsi, len(self.psi), wrap=True)
        iv0, iv1, fv = idx(vq, self.v, self.dv, len(self.v))
        out = np.zeros_like(xq)
        for bx, wx in ((ix0, 1 - fx), (ix1, fx)):
            for by, wy in ((iy0, 1 - fy), (iy1, fy)):
                for bp, wp in ((ip0, 1 - fp), (ip1, fp)):
                    for bv, wv in ((iv0, 1 - fv), (iv1, fv)):
                        out += Vg[bx, by, bp, bv] * (wx * wy * wp * wv)
        return out


def margins(grid):
    """g (avoid, >=0 outside inflated obstacles) and l (reach, >=0 inside goal)."""
    g = np.full(grid.shape, 3.0)
    for ox, oy, r in OBSTACLES:
        d = np.hypot(grid.X - ox, grid.Y - oy) - (r + CAR_HALF)
        g = np.minimum(g, d)
    g = np.clip(g, -1.0, 1.0)
    dgoal = np.hypot(grid.X - GOAL[0], grid.Y - GOAL[1])
    l = np.clip(GOAL[2] - dgoal, -1.0, 1.0)
    return g, l


def controls(v, mu):
    """Admissible (a_long, kappa) ON/IN the friction circle at speed v, friction mu.

    THE CRUX (v6.2 bugfix): a turn kappa demands LATERAL accel a_lat = v^2 |kappa|. If a_lat alone
    exceeds the friction budget mu*G, that curvature is INFEASIBLE — the car skids — REGARDLESS of
    throttle. The old code only zeroed a_long and STILL APPLIED the illegal kappa, so friction never
    limited turning and mu was inert for the safe SET (the v1–v6 flat-volume bug). Fix: cap the actual
    curvature to the friction-feasible |kappa| <= mu*G / v^2 (per cell), THEN spend the remaining
    budget on a_long. Now the friction circle binds on BOTH brake and turn — the whole point.
    Returns (a_long_array, kappa_array) pairs; kappa is per-cell (depends on v)."""
    a_fric = mu * G
    kmax_fric = a_fric / np.maximum(v ** 2, 1e-6)          # friction-limited curvature (per cell)
    out = []
    for frac in (-1.0, -0.5, 0.0, 0.5, 1.0):              # desired curvature as a fraction of steering limit
        k_des = frac * KAPPA_MAX
        k = np.clip(k_des, -kmax_fric, kmax_fric)          # cap by friction (and |k_des|<=KAPPA_MAX already)
        a_lat = (v ** 2) * np.abs(k)
        rem = np.sqrt(np.maximum(a_fric ** 2 - a_lat ** 2, 0.0))   # a_long budget left (>=0)
        for a in (rem, np.zeros_like(v), -rem):            # accel / coast / brake
            out.append((a, k))
    return out


def solve(grid, g, l, mu, V_init=None, tol=1e-5, max_iter=300, patience=12, verbose=False):
    """Reach-avoid HJ + track the optimal (curvature, a_long) — the 'mode' — at each state.

    Precompute the per-control next-state indices/weights ONCE (dynamics don't depend on V): the
    hot loop then only gathers V at cached corners — ~3x faster than recomputing trig+interp indices
    every sweep. Convergence: mask-change FRACTION < 1e-4 for `patience` sweeps (exact equality never
    triggers — boundary cells jitter under interpolation)."""
    V = np.minimum(g, l) if V_init is None else V_init.copy()
    N = int(np.prod(grid.shape))
    # cache each control's 16-corner (flat-index, weight) stencil
    stencils = []
    for a_long, k in controls(grid.V, mu):
        xn = grid.X + grid.V * np.cos(grid.PS) * DT
        yn = grid.Y + grid.V * np.sin(grid.PS) * DT
        pn = grid.PS + grid.V * k * DT
        vn = np.clip(grid.V + a_long * DT, V_MIN, V_MAX)
        stencils.append((_stencil(grid, xn, yn, pn, vn), k, a_long))
    mask, stable = V >= 0, 0
    best_k = np.zeros(grid.shape); best_a = np.zeros(grid.shape)
    for it in range(max_iter):
        Vf = V.reshape(-1)
        best = np.full(grid.shape, -np.inf)
        bk = np.zeros(grid.shape); ba = np.zeros(grid.shape)
        for (corners, k, a_long) in stencils:
            nvf = np.zeros(Vf.shape)
            for flat, w in corners:
                nvf += Vf[flat] * w
            nv = nvf.reshape(grid.shape)
            better = nv > best
            best = np.where(better, nv, best)
            bk = np.where(better, k, bk); ba = np.where(better, a_long, ba)
        V_new = np.minimum(g, np.maximum(l, best))
        d = np.abs(V_new - V).max(); V = V_new; best_k = bk; best_a = ba
        nm = V >= 0
        changed = np.count_nonzero(nm ^ mask) / N
        stable = stable + 1 if changed < 1e-4 else 0
        mask = nm
        if verbose and (it % 20 == 0 or stable >= patience):
            print(f"    it{it:3d}  dV={d:.3e}  mask-change={changed:.2e}  safe={nm.mean():.3f}", flush=True)
        if stable >= patience or d < tol:
            return V, best_k, best_a, it + 1
    return V, best_k, best_a, max_iter


def _stencil(grid, xq, yq, pq, vq):
    """Return the 16 (flat-index-array, weight-array) corners for a batch of query points — cached so
    the value-iteration hot loop is pure gather+add (no trig / index recompute per sweep)."""
    nx, ny, npsi, nv = grid.shape

    def idx(q, ax, d, n, wrap=False):
        f = (q - ax[0]) / d
        if wrap:
            f = f % n
            i0 = np.floor(f).astype(int) % n
            return i0, (i0 + 1) % n, f - np.floor(f)
        f = np.clip(f, 0, n - 1.0001)
        i0 = np.floor(f).astype(int)
        return i0, i0 + 1, f - i0
    ix0, ix1, fx = idx(xq, grid.x, grid.dx, nx)
    iy0, iy1, fy = idx(yq, grid.y, grid.dy, ny)
    ip0, ip1, fp = idx(pq, grid.psi, grid.dpsi, npsi, wrap=True)
    iv0, iv1, fv = idx(vq, grid.v, grid.dv, nv)
    corners = []
    for bx, wx in ((ix0, 1 - fx), (ix1, fx)):
        for by, wy in ((iy0, 1 - fy), (iy1, fy)):
            for bp, wp in ((ip0, 1 - fp), (ip1, fp)):
                for bv, wv in ((iv0, 1 - fv), (iv1, fv)):
                    flat = ((bx * ny + by) * npsi + bp) * nv + bv
                    corners.append((flat.reshape(-1), (wx * wy * wp * wv).reshape(-1)))
    return corners


def slice_area(grid, mask_xypv):
    """4-D safe-cell count * xy-cell / (npsi*nv) — reads as a 2-D footprint (a proxy volume)."""
    return float(mask_xypv.sum() * grid.cell_xy / (grid.shape[2] * grid.shape[3]))


def mode_label(grid, K, A, mu):
    """Categorize the optimal action: THREAD (near-max turn) vs BRAKE (hard decel, ~straight)."""
    a_fric = mu * G
    thread = np.abs(K) > 0.7 * KAPPA_MAX
    brake = (A < -0.4 * a_fric) & (np.abs(K) < 0.4 * KAPPA_MAX)
    return thread, brake


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/E011")
    ap.add_argument("--mus", type=float, nargs="+", default=[1.0, 0.5, 0.25, 0.12])  # ICE->DRY range
    ap.add_argument("--nx", type=int, default=55); ap.add_argument("--ny", type=int, default=41)
    ap.add_argument("--npsi", type=int, default=24); ap.add_argument("--nv", type=int, default=21)
    ap.add_argument("--refine", action="store_true", help="also solve at ~1.4x res; check volume-ratio convergence")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    grid = Grid4(args.nx, args.ny, args.npsi, args.nv)
    g, l = margins(grid)
    print(f"grid {grid.shape} = {np.prod(grid.shape):,} pts  |  v in [{V_MIN},{V_MAX}]  |  mu sweep {args.mus}")

    Vs, Ks, As, vols = {}, {}, {}, {}
    Vp = None
    import time; t0 = time.time()
    for i, mu in enumerate(args.mus):                    # warm-start high->low mu
        V, K, A, it = solve(grid, g, l, mu, V_init=Vp, verbose=(i == 0))
        Vp = V; Vs[mu] = V; Ks[mu] = K; As[mu] = A
        vols[mu] = slice_area(grid, V >= 0)
        # Pi dial + analytic thread-speed at this mu (diagnostics)
        v_thread = np.sqrt(mu * G * R_GAP)
        print(f"  mu={mu:.2f}  safe-vol={vols[mu]:.4f}  v_thread*={v_thread:.2f}  ({it} it, {time.time()-t0:.0f}s)")

    mus = args.mus
    vr = max(vols.values()) / max(min(vols.values()), 1e-9)
    # nesting: safe(mu_lo) subset safe(mu_hi). Report the VIOLATION FRACTION (tiny => numerical).
    nest_viol = max((((Vs[mus[i]] >= 0) & ~(Vs[mus[i - 1]] >= 0)).mean()) for i in range(1, len(mus)))
    nested = nest_viol < 1e-3
    hi, lo = mus[0], mus[-1]
    both = (Vs[hi] >= 0) & (Vs[lo] >= 0)

    # gate 3: turn-SIGN flip
    flip3 = both & (np.sign(Ks[hi]) != np.sign(Ks[lo])) & (np.abs(Ks[hi]) > 0.1) & (np.abs(Ks[lo]) > 0.1)
    flip3_frac = float(flip3.sum() / max(both.sum(), 1))

    # gate 4: BRAKE-vs-THREAD flip (the v6 mode structure)
    th_hi, br_hi = mode_label(grid, Ks[hi], As[hi], hi)
    th_lo, br_lo = mode_label(grid, Ks[lo], As[lo], lo)
    flip4 = both & ((th_hi & br_lo) | (br_hi & th_lo))
    flip4_frac = float(flip4.sum() / max(both.sum(), 1))

    # gate 4 analytic: does the mode-boundary speed track sqrt(mu)?  Extract v*(mu) = highest safe v
    # that still THREADS, at approach states near the gap mouth (x in [X_TOPEND-0.3, X_TOPEND+0.6],
    # y in [-0.3,0.6], psi in [0, 0.9] heading toward the up-exit).
    xm = (grid.X >= X_TOPEND - 0.3) & (grid.X <= X_TOPEND + 0.6)
    ym = (grid.Y >= -0.3) & (grid.Y <= 0.6)
    pm = (grid.PS >= 0.0) & (grid.PS <= 0.9)
    approach = xm & ym & pm
    vstar = {}
    for mu in mus:
        th, _ = mode_label(grid, Ks[mu], As[mu], mu)
        sel = approach & (Vs[mu] >= 0) & th
        vstar[mu] = float(grid.V[sel].max()) if sel.any() else float("nan")
    # log-log slope of v* vs mu (predict ~0.5 for the sqrt law)
    vv = [(m, vstar[m]) for m in mus if np.isfinite(vstar[m]) and vstar[m] > 0]
    sqrt_slope = float("nan")
    if len(vv) >= 2:
        lm = np.log([m for m, _ in vv]); lv = np.log([x for _, x in vv])
        sqrt_slope = float(np.polyfit(lm, lv, 1)[0])

    np.savez_compressed(os.path.join(args.out, "gate.npz"),
                        x=grid.x, y=grid.y, psi=grid.psi, v=grid.v, mus=np.array(mus), g=g, l=l,
                        **{f"V_{m}": Vs[m] for m in mus}, **{f"K_{m}": Ks[m] for m in mus},
                        **{f"A_{m}": As[m] for m in mus})

    # gate 1 grid-convergence: rerun at ~1.4x (x,y,v) res, compare volume ratio
    vr_fine = None
    if args.refine:
        gf = Grid4(int(args.nx * 1.4), int(args.ny * 1.4), args.npsi, int(args.nv * 1.4))
        gg, ll = margins(gf); Vpp = None; volf = {}
        print("refine:")
        for mu in mus:
            Vf, _, _, itf = solve(gf, gg, ll, mu, V_init=Vpp); Vpp = Vf
            volf[mu] = slice_area(gf, Vf >= 0)
            print(f"  mu={mu:.2f}  safe-vol={volf[mu]:.4f}  ({itf} it)")
        vr_fine = max(volf.values()) / max(min(volf.values()), 1e-9)

    print("\n" + "=" * 78)
    print(f"GATE 1 SENSITIVITY : volume range {vr:.2f}x   ({'PASS' if vr >= 3 else 'FAIL'}; need >=3x)")
    if vr_fine is not None:
        conv = abs(vr_fine - vr) / max(vr, 1e-9)
        print(f"       convergence : coarse {vr:.2f}x vs fine {vr_fine:.2f}x  "
              f"(rel-diff {conv:.0%}; {'STABLE' if conv < 0.25 else 'UNSTABLE — refine more'})")
    print(f"GATE 2 NESTING     : violation {100*nest_viol:.3f}% of cells   "
          f"({'PASS' if nested else 'FAIL'}; structural, so <0.1% = numerical boundary jitter)")
    print(f"GATE 3 TURN-SIGN   : {100*flip3_frac:.1f}% of jointly-safe states flip optimal-turn SIGN "
          f"({'PASS' if flip3_frac > 0.02 else 'fail (bonus)'}; want >2%)")
    print(f"GATE 4 BRAKE/THREAD: {100*flip4_frac:.1f}% flip BRAKE<->THREAD with mu "
          f"({'PASS' if flip4_frac > 0.02 else 'FAIL'}; need >2%)")
    print(f"       v*(mu) thread-boundary: " + "  ".join(f"mu={m}:{vstar[m]:.2f}" for m in mus)
          + f"   [log-log slope {sqrt_slope:+.2f}; sqrt-law predicts +0.50]")
    passed = vr >= 3 and nested and flip4_frac > 0.02
    print("-" * 78)
    print("VERDICT: " + ("volume + brake/thread BOTH pass -> the friction family bifurcates AND is "
                         "least-conservative-per-ODD; RL is warranted."
                         if passed else
                         "FAIL on >=1 core gate (1 or 4) -> tune the Pi-dials (v_max / corridor length "
                         "/ r_gap), NOT the topology (professor: iterate scalars, not geometry)."))
    if vr < 3:
        print(f"  fix sensitivity: raise V_MAX or shorten the corridor so Pi(mu_lo)>=3 "
              f"(currently V_MAX={V_MAX}, dead-end x={X_DEAD}).")
    if flip4_frac <= 0.02:
        print("  fix brake/thread: move r_gap / dead-end so the band v in [sqrt(mu_lo G r_gap), "
              "sqrt(mu_hi G r_gap)] lands in the visited, jointly-safe region.")
    print("=" * 78)

    _plot(grid, mus, Vs, Ks, flip4, args.out)


def _plot(grid, mus, Vs, Ks, flip4, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(mus) + 1, figsize=(3.4 * (len(mus) + 1), 3.6))
    for ax, mu in zip(axes[:-1], mus):
        safe_xy = (Vs[mu] >= 0).any(axis=(2, 3))
        turn_xy = (np.sign(Ks[mu]) * (Vs[mu] >= 0)).sum(axis=(2, 3)) / np.maximum((Vs[mu] >= 0).sum(axis=(2, 3)), 1)
        ax.contourf(grid.x, grid.y, safe_xy.T, levels=[0.5, 1.5], colors=["#dfe"], alpha=0.6)
        ax.contourf(grid.x, grid.y, turn_xy.T, levels=np.linspace(-1, 1, 11), cmap="coolwarm", alpha=0.6)
        for ox, oy, r in OBSTACLES:
            ax.add_patch(plt.Circle((ox, oy), r, color="#333", alpha=0.8))
        ax.add_patch(plt.Circle(GOAL[:2], GOAL[2], color="#2a2", alpha=0.5))
        ax.set_title(f"$\\mu$={mu}\n(footprint safe-for-some-$(\\psi,v)$)", fontsize=9)
        ax.set_xlim(grid.x[0], grid.x[-1]); ax.set_ylim(grid.y[0], grid.y[-1]); ax.set_aspect("equal")
    flip_xy = flip4.any(axis=(2, 3))
    ax = axes[-1]
    ax.contourf(grid.x, grid.y, flip_xy.T, levels=[0.5, 1.5], colors=["#c0392b"], alpha=0.7)
    for ox, oy, r in OBSTACLES:
        ax.add_patch(plt.Circle((ox, oy), r, color="#333", alpha=0.5))
    ax.add_patch(plt.Circle(GOAL[:2], GOAL[2], color="#2a2", alpha=0.4))
    ax.set_title("BRAKE<->THREAD flip region\n(optimal mode switches with $\\mu$)", fontsize=9)
    ax.set_xlim(grid.x[0], grid.x[-1]); ax.set_ylim(grid.y[0], grid.y[-1]); ax.set_aspect("equal")
    fig.suptitle("E011 v6 — friction-circle grid gate: volume shrinks AND mode flips with $\\mu$?", fontsize=11)
    fig.tight_layout()
    p = os.path.join(out, "gate.png"); fig.savefig(p, dpi=130); plt.close(fig)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
