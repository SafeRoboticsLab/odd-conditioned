"""E012 (Rung 0.5) — the guarantee-horizon KILL-TEST. RL-free. Gates the whole belief-robust demo.

Professor (consult 2026-07-18, env fork): because friction μ enters ONLY through the nested control
set U(μ), a field adversary just parks at μ_min ahead ⇒ the unconstrained worst-case value is exactly
the blind slice V(·;μ_min). So **knowing the local μ is worth ZERO for a guarantee unless the ODD has
structure linking current μ to near-future μ** — a guaranteed-μ horizon `w` (seconds). Sources of w:
patch length L (w=L/v−δ), spatial rate bound (w=(μ_cur−μ_floor)/(ρv)), preview (w=d_obs/v); detector
latency δ subtracts. blind = w→0 pessimist; point-estimate = w→∞ optimist; belief-robust = honest w.

THE OBJECT: V*(x, μ_cur, w) = finite-horizon reach-avoid, under μ_cur-admissible controls, INTO the
fallback stay-set {V∞(·;μ_min) ≥ 0}, within N=w/DT steps, avoiding obstacles throughout. Worst case is
provably "μ drops to μ_min at the window's end" (nestedness), so reaching the μ_min-safe set by time w
certifies safety for the composite "μ_cur for w, then μ_min forever". Backup:
    W_0 = min(g, V∞(·;μ_min))
    W_k = min( g, max( V∞(·;μ_min), max_u W_{k-1}(f(x,u; μ_cur)) ) )
    V*(·,μ_cur, w=k·DT) = W_k.

THE KILL-TEST: gap(w) = vol{V*(·, μ_hi, w) ≥ 0} − vol{V∞(·;μ_min) ≥ 0}  (value of local knowledge).
  gap(0)=0 (blind).  gap(∞) → the full knowledge value (≈ vol{V∞(·;μ_hi)} − vol{V∞(·;μ_min)}).
  If gap(w) is negligible for physically sensible w (patch ≳ maneuver length), belief-robust has NO
  room over blind on this geometry ⇒ REDESIGN (icier μ / longer corridor) BEFORE training anything.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import E011_friction_grid_gate as E  # noqa: E402


def build_stencils(grid, mu):
    """Cache the 16-corner next-state stencil for every friction-limited control at friction mu."""
    st = []
    for a_long, k in E.controls(grid.V, mu):
        xn = grid.X + grid.V * np.cos(grid.PS) * E.DT
        yn = grid.Y + grid.V * np.sin(grid.PS) * E.DT
        pn = grid.PS + grid.V * k * E.DT
        vn = np.clip(grid.V + a_long * E.DT, E.V_MIN, E.V_MAX)
        st.append(E._stencil(grid, xn, yn, pn, vn))
    return st


def backup(W, g, target, stencils, shape):
    """One finite-horizon reach-avoid step: W <- min(g, max(target, max_u interp(W, f(x,u))))."""
    Wf = W.reshape(-1)
    best = np.full(shape, -np.inf)
    for corners in stencils:
        nvf = np.zeros(Wf.shape)
        for flat, w in corners:
            nvf += Wf[flat] * w
        best = np.maximum(best, nvf.reshape(shape))
    return np.minimum(g, np.maximum(target, best))


def horizon_sweep(grid, g, target, mu_cur, N_max):
    """Return vols[k] = vol{V*(·,mu_cur, w=k·DT) >= 0} for k=0..N_max (and the v>=2 band)."""
    st = build_stencils(grid, mu_cur)
    W = np.minimum(g, target)                       # W_0
    vols = [E.slice_area(grid, W >= 0)]
    vols_op = [_op_vol(grid, W >= 0)]
    for k in range(1, N_max + 1):
        W = backup(W, g, target, st, grid.shape)
        vols.append(E.slice_area(grid, W >= 0))
        vols_op.append(_op_vol(grid, W >= 0))
    return np.array(vols), np.array(vols_op), W


def _op_vol(grid, mask):
    """Volume restricted to the operationally-relevant speed band v>=2 (E011: full vol is creep-diluted)."""
    m = mask & (grid.V >= 2.0)
    return float(m.sum() * grid.cell_xy / (grid.shape[2] * grid.shape[3]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/E012")
    ap.add_argument("--nx", type=int, default=41); ap.add_argument("--ny", type=int, default=31)
    ap.add_argument("--npsi", type=int, default=20); ap.add_argument("--nv", type=int, default=21)
    ap.add_argument("--mu-min", type=float, default=0.1)
    ap.add_argument("--mu-curs", type=float, nargs="+", default=[1.0, 0.6, 0.35])
    ap.add_argument("--nmax", type=int, default=60)     # max window = nmax*DT seconds
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    grid = E.Grid4(args.nx, args.ny, args.npsi, args.nv)
    g, l = E.margins(grid)
    print(f"grid {grid.shape}  DT={E.DT}  max window w={args.nmax*E.DT:.2f}s  mu_min={args.mu_min}")

    # fallback stay-set target = infinite-horizon mu_min reach-avoid value (the blind slice)
    import time; t0 = time.time()
    V_min = E.solve(grid, g, l, args.mu_min)[0]
    V_hi = E.solve(grid, g, l, max(args.mu_curs), V_init=V_min)[0]
    blind_vol = E.slice_area(grid, V_min >= 0); blind_op = _op_vol(grid, V_min >= 0)
    ceil_vol = E.slice_area(grid, V_hi >= 0);  ceil_op = _op_vol(grid, V_hi >= 0)
    print(f"slices: blind vol{{V_min>=0}}={blind_vol:.3f} (op {blind_op:.3f}); "
          f"ceiling vol{{V_hi>=0}}={ceil_vol:.3f} (op {ceil_op:.3f})  [{time.time()-t0:.0f}s]")

    ws = np.arange(args.nmax + 1) * E.DT
    curves = {}
    for mu_cur in args.mu_curs:
        vols, vols_op, _ = horizon_sweep(grid, g, V_min, mu_cur, args.nmax)
        curves[mu_cur] = (vols, vols_op)
        gap = vols - blind_vol
        gmax = vols[-1] - blind_vol
        # w to reach 50% / 90% of this mu_cur's asymptotic gap
        def w_at(frac):
            thr = blind_vol + frac * gmax
            idx = np.argmax(vols >= thr) if (vols >= thr).any() else -1
            return ws[idx] if idx >= 0 else np.nan
        print(f"  mu_cur={mu_cur:.2f}: gap(w) 0 -> {gmax:.3f} (full) ; "
              f"50% at w={w_at(0.5):.2f}s, 90% at w={w_at(0.9):.2f}s ; "
              f"gap@0.5s={np.interp(0.5,ws,gap):.3f} @1.0s={np.interp(1.0,ws,gap):.3f} "
              f"@1.5s={np.interp(1.5,ws,gap):.3f}")

    np.savez_compressed(os.path.join(args.out, "horizon.npz"),
                        ws=ws, blind_vol=blind_vol, ceil_vol=ceil_vol,
                        **{f"vols_{m}": curves[m][0] for m in args.mu_curs},
                        **{f"volsop_{m}": curves[m][1] for m in args.mu_curs})

    # --- VERDICT: is the value of local knowledge realized at physically-sensible windows? ---
    # a realistic patch (L~2-4 m) traversed near the corner (v_thread ~ sqrt(mu G r_gap) ~ 2-3 m/s)
    # gives w ~ L/v ~ 0.7-1.5 s. Kill-test: gap in that window as a fraction of the full-knowledge gap.
    mu_hi = max(args.mu_curs)
    vols_hi = curves[mu_hi][0]; gap_hi = vols_hi - blind_vol
    full_gap = ceil_vol - blind_vol
    g10 = np.interp(1.0, ws, gap_hi)
    frac_at_1s = g10 / full_gap if full_gap > 1e-9 else 0.0
    print("\n" + "=" * 78)
    print(f"VALUE OF LOCAL KNOWLEDGE (mu_cur={mu_hi}): full-horizon gap {gap_hi[-1]:.3f} "
          f"(= {100*gap_hi[-1]/max(blind_vol,1e-9):.0f}% over blind); ceiling gap {full_gap:.3f}")
    print(f"  realized at w=1.0s: {g10:.3f}  = {100*frac_at_1s:.0f}% of the full-knowledge gap")
    verdict = (gap_hi[-1] > 0.15 * blind_vol) and (frac_at_1s > 0.3)
    print("-" * 78)
    if verdict:
        print("VERDICT: PASS -> local-mu knowledge buys a substantial, physically-reachable safe-set")
        print("  expansion over blind. The belief-robust-vs-blind demo has room. Proceed to env + R1.")
    else:
        print("VERDICT: FAIL -> gap negligible at sensible windows. Belief-robust ~ blind on THIS")
        print("  geometry/mu-range. REDESIGN (icier mu_min, longer corridor, sustained-grip hazard)")
        print("  BEFORE training anything. (This is exactly what the kill-test is for.)")
    print("=" * 78)

    _plot(ws, blind_vol, ceil_vol, curves, args.mu_curs, args.out)


def _plot(ws, blind_vol, ceil_vol, curves, mu_curs, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    cols = {1.0: "#1a9850", 0.6: "#f1a340", 0.35: "#d73027"}
    for ax, band, title in ((ax1, 0, "full safe-set volume"), (ax2, 1, "operational band (v>=2)")):
        base = blind_vol if band == 0 else None
        for mu in mu_curs:
            vols = curves[mu][band]
            c = cols.get(mu, "#333")
            ax.plot(ws, vols, "-", color=c, lw=2, label=f"$\\mu_{{cur}}$={mu} (know local grip)")
        b = blind_vol if band == 0 else _blind_op(curves)  # blind line
        ax.axhline(b, ls="--", color="#666", lw=1.4, label="blind = $V(\\cdot;\\mu_{min})$ ($w\\to0$)")
        if band == 0:
            ax.axhline(ceil_vol, ls=":", color="#1a9850", lw=1.4, label="point-est ceiling $V(\\cdot;\\mu_{hi})$ ($w\\to\\infty$)")
        ax.axvspan(0.7, 1.5, color="#cce", alpha=0.3, label="realistic patch window $L/v$")
        ax.set_xlabel("guaranteed-$\\mu$ horizon $w$ (s)"); ax.set_ylabel("safe-set volume")
        ax.set_title(title, fontsize=10); ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("E012 (Rung 0.5) — value-of-local-knowledge: does knowing local $\\mu$ for $w$ seconds "
                 "buy safe set over blind worst-case?", fontsize=11)
    fig.tight_layout()
    p = os.path.join(out, "guarantee_horizon.png"); fig.savefig(p, dpi=130); plt.close(fig)
    print(f"wrote {p}")


def _blind_op(curves):
    # operational blind volume = the w=0 point of any curve's op band (all share W_0=min(g,V_min))
    return next(iter(curves.values()))[1][0]


if __name__ == "__main__":
    main()
