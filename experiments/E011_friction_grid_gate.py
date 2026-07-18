"""E011 — the RL-FREE grid gate for the friction-circle bicycle. Gates ALL downstream RL.

Professor (consult-bicycle-task-design): before training anything, verify on 4-D reach-avoid grid
truth that the friction-ODD family is well-posed AND genuinely bifurcates. Three checks:
  1. SENSITIVITY   — safe-set volume varies >=3x across mu.
  2. NESTING       — safe sets nest in mu (grippier ⊇ icier).
  3. MODE-FLIP     — a POSITIVE-MEASURE region of states where the optimal safety action's
                     SIDE (sign of optimal curvature) flips as mu changes, and that region MOVES
                     with mu. This is the brake-vs-swerve / which-gap bifurcation the whole toy
                     exists to exhibit. If it FAILS, redesign the geometry BEFORE any RL.

Why 4-D [x,y,psi,v] (not 3-D Dubins): constant-speed Dubins DELETES the brake mode. v is
load-bearing — the friction circle couples brake and turn, and that coupling IS the bifurcation.

THE FRICTION CIRCLE (the crux). Controls (a_long, kappa):
    |kappa| <= kappa_max                          (steering geometry)
    a_long^2 + (v^2 kappa)^2 <= (mu * G)^2         (friction: brake/accel TRADES OFF against turn)
  a_lat = v^2 kappa is centripetal accel; at the limit you cannot brake hard AND turn hard, so
  brake-vs-swerve is a real either/or, and the crossover MOVES with mu. Dynamics (no v=0 turn
  singularity since psidot = v*kappa):
    xdot=v cos psi   ydot=v sin psi   psidot=v*kappa   vdot=a_long

Reach-avoid (single-player, control MAXIMIZES — per Buzi, no adversary yet):
    V(s) = min( g(s), max( l(s), max_u V(f(s,u)) ) )     discrete-time, anchor min(l,g).
  g>=0 outside obstacles (inflated by car half-length); l>=0 inside goal.
"""
import argparse
import os

import numpy as np

# --- physical constants (SI-ish; arena ~4x3 m, v<=2 m/s) ---------------------------------------
G = 9.8
V_MAX = 2.0
DT = 0.05
WHEELBASE, DELTA_LIM = 0.257, 0.35
KAPPA_MAX = np.tan(DELTA_LIM) / WHEELBASE          # ~1.42 /m  (min turn radius ~0.7 m)
CAR_HALF = 0.21                                     # inflate obstacles by this (point-car grid)

# --- asymmetric two-gap geometry: near-narrow (upper) gap vs far-wide (lower) detour -----------
# high mu: thread the tight upper gap (sharp turn, needs grip).  low mu: can't -> wide lower arc.
GOAL = (3.5, 0.0, 0.35)
OBSTACLES = np.array([
    [1.8,  0.00, 0.45],     # central: blocks straight
    [1.8,  0.95, 0.35],     # upper: with the central one, forms a NARROW upper gap (~0.15 m)
    [1.8, -1.30, 0.55],     # lower-far: leaves a WIDE but longer lower route
], dtype=np.float64)


class Grid4:
    def __init__(self, nx=31, ny=31, npsi=24, nv=12,
                 xlim=(-0.2, 4.0), ylim=(-1.9, 1.9), vlim=(0.0, V_MAX)):
        self.x = np.linspace(*xlim, nx)
        self.y = np.linspace(*ylim, ny)
        self.psi = np.linspace(-np.pi, np.pi, npsi, endpoint=False)   # periodic
        self.v = np.linspace(*vlim, nv)
        self.shape = (nx, ny, npsi, nv)
        self.X, self.Y, self.PS, self.V = np.meshgrid(self.x, self.y, self.psi, self.v, indexing="ij")
        self.dx = self.x[1] - self.x[0]; self.dy = self.y[1] - self.y[0]
        self.dpsi = 2 * np.pi / npsi; self.dv = self.v[1] - self.v[0]
        self.cell_xy = self.dx * self.dy                              # for (x,y) area at fixed (psi,v)

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
    """Admissible (a_long, kappa) samples at speed v under friction mu. Bang-ish: for each of a few
    kappa, the friction-limited a_long extremes + coast. Returns list of (a_long_array, kappa)."""
    a_fric = mu * G
    ks = np.array([-1.0, -0.5, 0.0, 0.5, 1.0]) * KAPPA_MAX
    out = []
    for k in ks:
        a_lat = (v ** 2) * abs(k)
        rem = np.sqrt(np.maximum((a_fric ** 2) - a_lat ** 2, 0.0))   # a_long budget left after turning
        for a in (rem, np.zeros_like(v), -rem):                       # accel / coast / brake
            out.append((a, k))
    return out


def solve(grid, g, l, mu, V_init=None, tol=1e-5, max_iter=400, patience=30):
    """Reach-avoid HJ + track the optimal curvature sign (the 'mode') at each state."""
    V = np.minimum(g, l) if V_init is None else V_init.copy()
    ctrls = controls(grid.V, mu)                         # depends on v-grid, not V -> hoist
    mask, stable = V >= 0, 0
    best_k = np.zeros(grid.shape)
    for it in range(max_iter):
        best = np.full(grid.shape, -np.inf)
        bk = np.zeros(grid.shape)
        for a_long, k in ctrls:
            xn = grid.X + grid.V * np.cos(grid.PS) * DT
            yn = grid.Y + grid.V * np.sin(grid.PS) * DT
            pn = grid.PS + grid.V * k * DT
            vn = np.clip(grid.V + a_long * DT, 0.0, V_MAX)
            nv = grid.interp(V, xn, yn, pn, vn)
            better = nv > best
            best = np.where(better, nv, best)
            bk = np.where(better, k, bk)
        V_new = np.minimum(g, np.maximum(l, best))
        d = np.abs(V_new - V).max(); V = V_new; best_k = bk
        nm = V >= 0; stable = stable + 1 if np.array_equal(nm, mask) else 0; mask = nm
        if stable >= patience or d < tol:
            return V, best_k, it + 1
    return V, best_k, max_iter


def slice_area(grid, mask_xypv):
    """(x,y)-area of a boolean set, integrated over (psi,v) then normalized to a per-(psi,v) mean
    so it reads as a 2-D footprint. Here: total 4-D cell-count * xy-cell / (npsi*nv)."""
    return float(mask_xypv.sum() * grid.cell_xy / (grid.shape[2] * grid.shape[3]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/E011")
    ap.add_argument("--mus", type=float, nargs="+", default=[1.0, 0.7, 0.5, 0.35])
    ap.add_argument("--nx", type=int, default=31); ap.add_argument("--ny", type=int, default=31)
    ap.add_argument("--npsi", type=int, default=24); ap.add_argument("--nv", type=int, default=12)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    grid = Grid4(args.nx, args.ny, args.npsi, args.nv)
    g, l = margins(grid)
    print(f"grid {grid.shape} = {np.prod(grid.shape):,} pts  |  mu sweep {args.mus}")

    Vs, Ks, vols = {}, {}, {}
    Vp = None
    import time; t0 = time.time()
    for mu in args.mus:                                  # warm-start high->low mu
        V, K, it = solve(grid, g, l, mu, V_init=Vp)
        Vp = V; Vs[mu] = V; Ks[mu] = K
        vols[mu] = slice_area(grid, V >= 0)
        print(f"  mu={mu:.2f}  safe-vol={vols[mu]:.4f}  ({it} it, {time.time()-t0:.0f}s)")

    mus = args.mus
    vr = max(vols.values()) / max(min(vols.values()), 1e-9)
    # nesting: grippier (higher mu) should CONTAIN icier (lower mu)
    nested = all(not (((Vs[mus[i]] >= 0) & ~(Vs[mus[i - 1]] >= 0)).any()) for i in range(1, len(mus)))
    # mode-flip: states SAFE under both extremes where the optimal-curvature SIGN differs
    hi, lo = mus[0], mus[-1]
    both = (Vs[hi] >= 0) & (Vs[lo] >= 0)
    flip = both & (np.sign(Ks[hi]) != np.sign(Ks[lo])) & (np.abs(Ks[hi]) > 0.1) & (np.abs(Ks[lo]) > 0.1)
    flip_frac = float(flip.sum() / max(both.sum(), 1))
    flip_area = slice_area(grid, flip)

    np.savez_compressed(os.path.join(args.out, "gate.npz"),
                        x=grid.x, y=grid.y, psi=grid.psi, v=grid.v, mus=np.array(mus), g=g, l=l,
                        **{f"V_{m}": Vs[m] for m in mus}, **{f"K_{m}": Ks[m] for m in mus})

    print("\n" + "=" * 74)
    print(f"GATE 1 SENSITIVITY : volume range {vr:.2f}x   ({'PASS' if vr >= 3 else 'FAIL'}; need >=3x)")
    print(f"GATE 2 NESTING     : {nested}   ({'PASS' if nested else 'FAIL'})")
    print(f"GATE 3 MODE-FLIP   : {100*flip_frac:.1f}% of jointly-safe states flip optimal-turn SIGN "
          f"between mu={hi} and mu={lo}  ({'PASS' if flip_frac > 0.02 else 'FAIL'}; need >2% positive measure)")
    passed = vr >= 3 and nested and flip_frac > 0.02
    print("-" * 74)
    print("VERDICT: " + ("ALL PASS -> the friction family bifurcates; RL is warranted."
                         if passed else
                         "FAIL on >=1 gate -> REDESIGN GEOMETRY/mu before any RL (this is the point of the gate)."))
    if vr < 3:
        print("  fix sensitivity: widen mu range or tighten the arena so friction binds.")
    if flip_frac <= 0.02:
        print("  fix mode-flip: the two gaps aren't forcing a side-switch with mu — adjust obstacle")
        print("  offsets/radii so the tight gap becomes infeasible at low mu (the whole design point).")
    print("=" * 74)

    _plot(grid, mus, Vs, Ks, flip, args.out)


def _plot(grid, mus, Vs, Ks, flip, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # project onto (x,y): a state is 'safe-ish' if safe for SOME (psi,v); mode = mean optimal-k sign
    kv = grid.shape[2] * grid.shape[3]
    fig, axes = plt.subplots(1, len(mus) + 1, figsize=(3.4 * (len(mus) + 1), 3.6))
    for ax, mu in zip(axes[:-1], mus):
        safe_xy = (Vs[mu] >= 0).any(axis=(2, 3))               # (x,y): safe for some (psi,v)
        turn_xy = (np.sign(Ks[mu]) * (Vs[mu] >= 0)).sum(axis=(2, 3)) / np.maximum((Vs[mu] >= 0).sum(axis=(2, 3)), 1)
        ax.contourf(grid.x, grid.y, safe_xy.T, levels=[0.5, 1.5], colors=["#dfe"], alpha=0.6)
        ax.contourf(grid.x, grid.y, turn_xy.T, levels=np.linspace(-1, 1, 11), cmap="coolwarm", alpha=0.65)
        for ox, oy, r in OBSTACLES:
            ax.add_patch(plt.Circle((ox, oy), r, color="#333", alpha=0.8))
        ax.add_patch(plt.Circle(GOAL[:2], GOAL[2], color="#2a2", alpha=0.5))
        ax.set_title(f"$\\mu$={mu}\nblue=turn-left red=right", fontsize=9)
        ax.set_xlim(grid.x[0], grid.x[-1]); ax.set_ylim(grid.y[0], grid.y[-1]); ax.set_aspect("equal")
    flip_xy = flip.any(axis=(2, 3))
    ax = axes[-1]
    ax.contourf(grid.x, grid.y, flip_xy.T, levels=[0.5, 1.5], colors=["#c0392b"], alpha=0.7)
    for ox, oy, r in OBSTACLES:
        ax.add_patch(plt.Circle((ox, oy), r, color="#333", alpha=0.5))
    ax.add_patch(plt.Circle(GOAL[:2], GOAL[2], color="#2a2", alpha=0.4))
    ax.set_title("MODE-FLIP region\n(optimal turn-side flips with $\\mu$)", fontsize=9)
    ax.set_xlim(grid.x[0], grid.x[-1]); ax.set_ylim(grid.y[0], grid.y[-1]); ax.set_aspect("equal")
    fig.suptitle("E011 — friction-circle grid gate: does the safe strategy bifurcate with $\\mu$?", fontsize=11)
    fig.tight_layout()
    p = os.path.join(out, "gate.png"); fig.savefig(p, dpi=130); plt.close(fig)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
