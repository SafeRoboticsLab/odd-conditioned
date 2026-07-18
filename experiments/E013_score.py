"""E013 scoring — IoU-vs-μ of each Rung-1 critic's safe set {V(x;μ)≥0} against the E011 grid truth,
re-solved with the ENV's GRADED l (value-parity). The E008c verdict, on the friction toy:
  conditioned tracks the family (IoU high across μ) ; blind frozen at one set ; spec = per-μ ceiling.

V(x;μ) = min over twin critics of Q(obs, actor(obs, deterministic)) — the reach-avoid value under the
learned policy. {V≥0} = reach-avoidable (same sign as the grid). IoU = |A∩B|/|A∪B| of the safe sets.
"""
import argparse
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np  # noqa: E402
import torch  # noqa: E402
torch.set_num_threads(4)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))
import E011_friction_grid_gate as E  # noqa: E402
from odd_conditioned.envs import friction_bicycle as FB  # noqa: E402

ARMS = {"conditioned": "mu_local", "blind": "blind", "spec_hi": "blind", "spec_lo": "blind"}


def graded_margins(grid):
    """g = E011's; l = the env's GRADED l (so the grid truth matches what the critic was trained on)."""
    g, _ = E.margins(grid)
    d = np.hypot(grid.X - FB.GOAL[0], grid.Y - FB.GOAL[1]); gr = FB.GOAL[2]
    inside = 0.3 * (gr - d) / gr
    outside = (gr - d) / 4.0
    l = np.clip(np.where(d <= gr, inside, outside), -1.0, 1.0)
    return g, l


def grid_truth(grid, mus):
    g, l = graded_margins(grid)
    V = {}; Vp = None
    for mu in mus:
        Vp = E.solve(grid, g, l, mu, V_init=Vp)[0]; V[mu] = Vp.copy()
    return V


def critic_value_on_grid(zip_path, obs_mode, grid, mu, batch=200_000):
    from safety_sb3 import ReachAvoidSAC
    m = ReachAvoidSAC.load(zip_path, device="cpu")
    X, Y, PS, Vv = (a.reshape(-1) for a in (grid.X, grid.Y, grid.PS, grid.V))
    cols = [X, Y, np.sin(PS), np.cos(PS), Vv]
    if obs_mode == "mu_local":
        lo, hi = FB.MU_RANGE
        cols.append(np.full_like(X, 2 * (mu - lo) / (hi - lo + 1e-9) - 1))
    obs = np.stack(cols, axis=1).astype(np.float32)
    out = np.empty(obs.shape[0], dtype=np.float32)
    for i in range(0, obs.shape[0], batch):
        o = torch.as_tensor(obs[i:i + batch])
        with torch.no_grad():
            a = m.actor(o, deterministic=True) if "deterministic" in m.actor.forward.__code__.co_varnames \
                else m.actor(o)
            q = torch.cat(m.critic(o, a), dim=1).min(dim=1).values
        out[i:i + batch] = q.numpy()
    return out.reshape(grid.shape)


def iou(a, b):
    inter = (a & b).sum(); union = (a | b).sum()
    return float(inter / union) if union else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/E013")
    ap.add_argument("--mus", type=float, nargs="+", default=[1.0, 0.6, 0.35, 0.2, 0.1])
    ap.add_argument("--nx", type=int, default=41); ap.add_argument("--ny", type=int, default=31)
    ap.add_argument("--npsi", type=int, default=20); ap.add_argument("--nv", type=int, default=21)
    args = ap.parse_args()

    grid = E.Grid4(args.nx, args.ny, args.npsi, args.nv)
    print(f"grid {grid.shape}  |  scoring arms {list(ARMS)}  |  μ {args.mus}")
    import time; t0 = time.time()
    Vgt = grid_truth(grid, args.mus)
    gt = {mu: (Vgt[mu] >= 0) for mu in args.mus}
    print(f"grid truth (graded l) solved [{time.time()-t0:.0f}s]  safe-frac: "
          + " ".join(f"{mu}:{gt[mu].mean():.3f}" for mu in args.mus))

    results = {}
    for arm, obs_mode in ARMS.items():
        p = os.path.join(args.out, f"{arm}.zip")
        if not os.path.exists(p):
            print(f"  [{arm}] missing {p} — skip"); continue
        row = {}
        for mu in args.mus:
            V = critic_value_on_grid(p, obs_mode, grid, mu)
            safe = V >= 0
            row[mu] = dict(iou=iou(safe, gt[mu]), frac=float(safe.mean()))
        results[arm] = row

    print("\nIoU vs grid truth   @ μ = " + "  ".join(f"{mu:>5}" for mu in args.mus))
    for arm in results:
        print(f"  {arm:>12} | " + "  ".join(f"{results[arm][mu]['iou']:5.2f}" for mu in args.mus))
    print("\nsafe-fraction (is blind FROZEN across μ? conditioned should TRACK gt):")
    print(f"  {'grid-truth':>12} | " + "  ".join(f"{gt[mu].mean():5.2f}" for mu in args.mus))
    for arm in results:
        print(f"  {arm:>12} | " + "  ".join(f"{results[arm][mu]['frac']:5.2f}" for mu in args.mus))

    import json
    json.dump({a: {str(mu): v for mu, v in r.items()} for a, r in results.items()},
              open(os.path.join(args.out, "score.json"), "w"), indent=2)
    print("\n" + "=" * 70)
    if "conditioned" in results and "blind" in results:
        c = results["conditioned"]; b = results["blind"]
        c_iou = np.mean([c[mu]["iou"] for mu in args.mus])
        b_spread = max(b[mu]["frac"] for mu in args.mus) - min(b[mu]["frac"] for mu in args.mus)
        c_spread = max(c[mu]["frac"] for mu in args.mus) - min(c[mu]["frac"] for mu in args.mus)
        gt_spread = max(gt[mu].mean() for mu in args.mus) - min(gt[mu].mean() for mu in args.mus)
        print(f"conditioned mean IoU {c_iou:.2f}; safe-frac spread — gt {gt_spread:.2f}, "
              f"conditioned {c_spread:.2f}, blind {b_spread:.2f}")
        print("VERDICT: conditioned tracks the family (high IoU, spread ~ gt) AND blind is frozen "
              "(spread << gt) => the thesis holds on the friction toy." if c_iou > 0.7 and b_spread < 0.5 * gt_spread
              else "VERDICT: inconclusive — inspect (more steps? sign convention? coverage?).")
    print("=" * 70)


if __name__ == "__main__":
    main()
