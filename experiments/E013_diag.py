"""E013 diagnostic (professor step 1): does the conditioned critic track the per-μ FAMILY in shape,
without a discounted grid re-solve? Two ground-truth-light checks:

  (A) THRESHOLD SWEEP — for each μ, find c*≥0 maximizing IoU({V_RL(·;μ)≥0}, {V_grid(·;μ)≥c}) on the
      SUPERVISED SUPPORT. If IoU peaks HIGH at a c*>0 that is roughly μ-consistent, the RL set is a
      clean INNER approximation (discount level-shift; shape is right) ⇒ family-tracking holds in spirit.
  (B) MONOTONICITY — conditioned composed-volume vs μ should be monotone↑ (grippier=bigger); blind
      should be ~FLAT (frozen at one averaged set). Needs NO ground truth.

Support = free space (g≥0) within the spawn box. Grid is the graded-l UNDISCOUNTED E011 slices (the
discount level-shift is absorbed by c in the sweep — that is the point of the sweep).
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import torch; torch.set_num_threads(4)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))
import E011_friction_grid_gate as E
from E013_score import graded_margins, critic_value_on_grid

MUS = [1.0, 0.6, 0.35, 0.2, 0.1]
DIR = os.environ.get("E013_DIR", "results/E013")


def main():
    grid = E.Grid4(41, 31, 20, 21)
    g, l = graded_margins(grid); g011 = E.margins(grid)[0]
    support = (g011 >= 0) & (grid.X >= -0.4) & (grid.X <= 3.2) & (grid.Y >= -0.9) & (grid.Y <= 2.7)
    supp_hi = support & (grid.V >= 2.0)
    print(f"support cells {support.sum():,} (v≥2: {supp_hi.sum():,})")

    Vgt = {}; Vp = None
    import time; t0 = time.time()
    for mu in MUS:
        Vp = E.solve(grid, g, l, mu, V_init=Vp)[0]; Vgt[mu] = Vp.copy()
    print(f"grid (graded-l, undiscounted) solved [{time.time()-t0:.0f}s]")

    def iou(a, b):
        u = (a | b).sum(); return float((a & b).sum() / u) if u else 1.0

    cs = np.linspace(0.0, 0.4, 21)
    for arm, mode in (("conditioned", "mu_local"), ("blind", "blind")):
        print(f"\n=== {arm} ===")
        vols = {}
        print(f"  {'μ':>5} | {'peakIoU':>7} {'c*':>5} | {'IoU@c0':>7} | {'RLvol':>7} {'GTvol':>7} (support)")
        for mu in MUS:
            V = critic_value_on_grid(f"{DIR}/{arm}.zip", mode, grid, mu)
            rl = (V >= 0) & support
            best_iou, best_c = -1, 0.0
            for c in cs:
                ii = iou(rl, (Vgt[mu] >= c) & support)
                if ii > best_iou:
                    best_iou, best_c = ii, c
            iou0 = iou(rl, (Vgt[mu] >= 0) & support)
            vols[mu] = rl.sum()
            print(f"  {mu:>5} | {best_iou:7.2f} {best_c:5.2f} | {iou0:7.2f} | "
                  f"{rl.sum():7d} {((Vgt[mu] >= 0) & support).sum():7d}")
        vv = [vols[m] for m in MUS]
        mono = all(vv[i] >= vv[i + 1] - max(1, 0.02 * vv[0]) for i in range(len(vv) - 1))  # ↓ as μ↓
        spread = (max(vv) - min(vv)) / max(max(vv), 1)
        print(f"  composed-vol vs μ (hi→lo): {vv}   monotone↓:{mono}  spread:{spread:.2f}")


if __name__ == "__main__":
    main()
