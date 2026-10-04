"""Acceptance test for a COMPOUND STAND checkpoint (the leg-dies-while-loaded demo, E078).

    python scripts/check_compound.py <compound_stand checkpoint .zip> [...]

Runs the E078 fixed-θ protocol (constant 80 N load at 0.25 m, FR-leg torque θ, 10 N push + one 25 N gust,
N=128 × 6 s) on the candidate at θ ∈ {1.0, 0.6, 0.4} (where the stance should hold) and θ = 0.2 (where it should
not). PASS = stance fraction >= 0.90 and tip rate <= 0.10 at every θ >= 0.4. Shipped reference: stance 0.98-0.99,
tip 0.02-0.05 for θ >= 0.4; tip 0.49 at θ = 0.2.
"""
import argparse
import sys

sys.path.insert(0, "experiments")

THETAS_OK, THETA_FAIL = (1.0, 0.6, 0.4), 0.2
STAND_PASS, TIP_PASS = 0.90, 0.10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    a = ap.parse_args()
    import E078_matrix as M
    for ck in a.ckpts:
        cells = {t: M.cell(ck, t) for t in (*THETAS_OK, THETA_FAIL)}
        ok = all(cells[t]["stand"] >= STAND_PASS and cells[t]["tip"] <= TIP_PASS for t in THETAS_OK)
        desc = "  ".join(f"θ={t}: stand {c['stand']:.2f} tip {c['tip']:.2f}" for t, c in cells.items())
        print(f"{'PASS' if ok else 'FAIL'}  {desc}  | {ck}", flush=True)


if __name__ == "__main__":
    main()
