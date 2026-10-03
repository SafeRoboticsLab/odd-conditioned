"""Acceptance test for a STAND expert checkpoint — does a (re)trained stand policy support the paper's results?

    python scripts/check_stand.py <stand checkpoint .zip> [<more checkpoints> ...] [--n 256]

Run from the repo root after `source activate.sh`. Two tests, no scripted push (deployment conditions):

  T1  stand expert alone, standing under a constant carried load W = 40 N and 100 N (CoM 0.25 m up), 20 s:
      survival. A usable stance expert keeps nearly everyone up at 40 N.
  T2  the payload-swap walk (E092, smooth ramp) with the SHIPPED rest / get-up / descent policies and the
      candidate as the stand expert: ODD-conditioned (V2) and ONE-WAY success / safe. The stand expert only flies
      the BRAKE there and E092's triggers do not read V_stand, so this isolates the stand policy's effect and
      needs no threshold recalibration.

PASS = T1 survival at 40 N >= 0.90 and T2 V2 success >= 0.60. Reference (shipped stand_hi recal): see
docs/TRAINING.md. The other three automaton checkpoints must be present (scripts/fetch_bundles.sh or train_all.sh).
"""
import argparse
import io
import json
import os
import sys
import contextlib

sys.path.insert(0, "experiments")

T1_PASS, T2_PASS = 0.90, 0.60


def check(ckpt, n):
    import E084_automaton as E84
    import E092_payload_walk as E92
    if not os.path.exists(ckpt):
        raise SystemExit(f"no such checkpoint: {ckpt}")
    saved_ck, saved_w = dict(E84.CK), E84.W_of
    E84.CK["stand"] = ckpt
    out = {"ckpt": ckpt}
    try:
        for W in (40.0, 100.0):                       # T1: stand expert alone, constant load, no push
            E84.W_of = lambda sched, t, _w=W: _w
            with contextlib.redirect_stdout(io.StringIO()):
                r = E84.rollout("square", "none", "STAND-ONLY")
            out[f"T1_stand_{int(W)}N"] = round(r["final"], 3)
        E84.W_of = saved_w
        E92.PUSH_SCALE = 0.0                          # T2: E092 ramp, shipped automaton + candidate stand
        for arm in ("V2", "ONE-WAY"):
            with contextlib.redirect_stdout(io.StringIO()):
                r = E92.rollout("period", arm, n=n)
            out[f"T2_{arm}"] = [round(r["success"], 3), round(r["safe"], 3)]
            out[f"T2_{arm}_brake_deaths"] = r["deaths"]["BRAKE"]
    finally:
        E84.CK.clear(); E84.CK.update(saved_ck); E84.W_of = saved_w
    out["PASS"] = bool(out["T1_stand_40N"] >= T1_PASS and out["T2_V2"][0] >= T2_PASS)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--json", default=None, help="append results as JSON lines to this file")
    a = ap.parse_args()
    for c in a.ckpts:
        r = check(c, a.n)
        print(f"{'PASS' if r['PASS'] else 'FAIL'}  T1 stand@40N {r['T1_stand_40N']:.2f}  @100N {r['T1_stand_100N']:.2f}  "
              f"| T2 V2 {r['T2_V2'][0]:.2f}/{r['T2_V2'][1]:.2f} (brake deaths {r['T2_V2_brake_deaths']})  "
              f"ONE-WAY {r['T2_ONE-WAY'][0]:.2f}/{r['T2_ONE-WAY'][1]:.2f}  | {c}", flush=True)
        if a.json:
            with open(a.json, "a") as f:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
