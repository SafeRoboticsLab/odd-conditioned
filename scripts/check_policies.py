"""Acceptance tests for (re)trained policies: does a candidate support the reported results?

    python scripts/check_policies.py stand runs/stand/go2_weight_stand_hi_adv/checkpoints/model_49999872_steps.zip
    python scripts/check_policies.py compound checkpoints/compound_stand/model.zip

stand      T1  the candidate alone, standing under a constant load of 40 N and 100 N (CoM 0.25 m), 20 s, no push:
               survival. A usable stance expert keeps nearly every robot up at 40 N.
           T2  the payload walk (period profile, no push) with the candidate as the STAND expert and the
               installed rest / getup / descend policies: ODD-conditioned and ONE-WAY success / safe. The STAND
               expert only flies the BRAKE there and the payload triggers do not read V_stand, so this isolates
               the stand policy and needs no recalibration.
           PASS = T1 survival at 40 N >= 0.90 and T2 ODD-conditioned success >= 0.60.
compound   the compound STAND policy at fixed leg derating θ while carrying 80 N (10 N push + one 35 N gust):
           PASS = standing fraction >= 0.90 and tip rate <= 0.10 at every θ in {1.0, 0.6, 0.4} (θ = 0.2 is shown
           for contrast: there the stance should fail).

Training is high-variance for the stance-type policies (docs/TRAINING.md): train several seeds, keep a passer.
"""
import argparse
import contextlib
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from odd_conditioned.automaton import rollout  # noqa: E402
from odd_conditioned.certificates import compound_cell  # noqa: E402
from odd_conditioned.scenarios import SCENARIOS as SC  # noqa: E402
from odd_conditioned.sim import LOAD_H  # noqa: E402

T1_PASS, T2_PASS = 0.90, 0.60
THETAS_OK, THETA_FAIL, STAND_PASS, TIP_PASS = (1.0, 0.6, 0.4), 0.2, 0.90, 0.10


def _quiet(*a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return rollout(*a, **k)


def check_stand(ckpt, n):
    out = {"ckpt": ckpt}
    for W in (40.0, 100.0):
        sc = SC["standing-square"].with_(odd=lambda t, _w=W: (_w, LOAD_H, None))
        out[f"T1_{int(W)}N"] = round(_quiet(sc, "task-only", push="none", n=n, checkpoints={"stand": ckpt})["safe"], 3)
    for m in ("odd", "one-way"):
        r = _quiet(SC["payload-period"], m, push="none", n=n, checkpoints={"stand": ckpt})
        out[f"T2_{m}"] = [round(r["success"], 3), round(r["safe"], 3)]
        out[f"T2_{m}_brake_deaths"] = r["deaths"]["BRAKE"]
    out["PASS"] = bool(out["T1_40N"] >= T1_PASS and out["T2_odd"][0] >= T2_PASS)
    print(f"{'PASS' if out['PASS'] else 'FAIL'}  T1 stand@40N {out['T1_40N']:.2f} @100N {out['T1_100N']:.2f}  | "
          f"T2 ODD-conditioned {out['T2_odd'][0]:.2f}/{out['T2_odd'][1]:.2f} (brake deaths "
          f"{out['T2_odd_brake_deaths']})  ONE-WAY {out['T2_one-way'][0]:.2f}/{out['T2_one-way'][1]:.2f}  | {ckpt}",
          flush=True)
    return out


def check_compound(ckpt, n):
    cells = {t: compound_cell(ckpt, t) for t in (*THETAS_OK, THETA_FAIL)}
    ok = all(cells[t]["stand"] >= STAND_PASS and cells[t]["tip"] <= TIP_PASS for t in THETAS_OK)
    desc = "  ".join(f"θ={t}: stand {c['stand']:.2f} tip {c['tip']:.2f}" for t, c in cells.items())
    print(f"{'PASS' if ok else 'FAIL'}  {desc}  | {ckpt}", flush=True)
    return {"ckpt": ckpt, "cells": cells, "PASS": ok}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=("stand", "compound"))
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--json", default=None, help="append results as JSON lines to this file")
    a = ap.parse_args()
    for c in a.ckpts:
        r = (check_stand if a.kind == "stand" else check_compound)(c, a.n)
        if a.json:
            with open(a.json, "a") as f:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
