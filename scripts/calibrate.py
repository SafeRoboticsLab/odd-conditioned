"""Recalibrate the automata's switching thresholds for (re)trained policies.

    python scripts/calibrate.py                       # every group, 3 pooled repetitions, ~7 min on an RTX 4070
    python scripts/calibrate.py --only standing walk   # a subset: standing walk leg payload demos
    python scripts/calibrate.py --reps 1               # one repetition, ~2.5 min (noisier tails)

Run from the repo root after `source activate.sh`, with the checkpoints you want to calibrate in results/.
The thresholds compare learned values (and one residual) against constants at the top of the experiment
scripts. Those constants were placed on the SHIPPED networks at particular operating points, e.g. "the descent
trigger fires on 7 % of light-load standing states". This script measures the same distributions on whatever
networks are in results/ and recommends the value that sits at the same operating point. It changes nothing:
it prints what to set, where, and the evidence. Run it on the shipped checkpoints and the recommendations
reproduce the shipped constants (up to sampling noise) — that is the check that the rules are right.

Rules (docs/TRAINING.md §6 explains each):
  EPS_DN        standing descent trigger      = 7th percentile of V̄_stand, robots standing under 40 N
  EPS_UP        standing get-up gate          = 94th percentile of V̄_up, settled rest after the load clears
  EPS_ABORT     standing get-up abort         = EPS_UP - 0.12
  V1_UP         V1's blind return             = 21st percentile of V̄_stand on the same settled-rest robots
  EPS_DN_WALK   walking descent trigger (E089) = highest value with <= 1.6 % false descents per 30 s of healthy
                                                walking (K_DN_WALK-step sustain; robots that survive the 30 s)
  ERR_DEG       leg-fault residual (E091)     = keep while it sits between the healthy and derated bands
  EPS_UP_LEG    leg-fault get-up gate         = 12th percentile of V̄_up, settled rest after the leg heals; abort -0.10
  EPS_UP_92     payload-walk get-up gate      = 35th percentile of V̄_up, settled rest after the load clears; abort -0.15
  demos         one-way demo triggers         = reported with their value-sweep midpoint and discrimination
"""
import argparse
import contextlib
import io
import json
import os
import sys

import torch as th

sys.path.insert(0, "experiments")

# Operating points of the shipped calibration (measured on the shipped networks, 2026-10-03).
OP = {"EPS_DN": 0.07, "EPS_UP": 0.94, "V1_UP": 0.21, "EPS_DN_WALK": 0.016, "EPS_UP_LEG": 0.12, "EPS_UP_92": 0.35}
ABORT_OFFSET = {"EPS_ABORT": 0.12, "EPS_ABORT_LEG": 0.10, "EPS_ABORT_92": 0.15}

rows = []   # (file, constant, current, recommended, evidence)
REPS = 1    # set by --reps: every probe is repeated and its samples pooled (tail percentiles are noisy)


def _flat(vals):
    return th.cat([v.flatten() for v in vals]).float().cpu() if vals else th.zeros(0)


def _q(x, p):
    return float(th.quantile(x, p)) if x.numel() else float("nan")


def _frac(x, thr):
    return float((x >= thr).float().mean()) if x.numel() else float("nan")


def _pct(x):
    return "  ".join(f"p{int(p * 100)} {_q(x, p):+.3f}" for p in (0.05, 0.25, 0.5, 0.75, 0.95))


def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def standing():
    import E084_automaton as E
    saved = E.W_of
    dist = {}
    for W in (40.0, 130.0, 220.0):
        E.W_of = lambda sched, t, _w=W: _w
        dist[W] = th.cat([_flat(_quiet(E.rollout, "square", "benign", "STAND-ONLY", cal="dn")["cal"])
                          for _ in range(REPS)])
    E.W_of = saved
    rec = _q(dist[40.0], OP["EPS_DN"])
    fire = " / ".join(f"{float((dist[w] < rec).float().mean()):.0%}" for w in dist)
    rows.append(("E084_automaton.py", "EPS_DN", E.EPS_DN, rec,
                 f"V̄_stand standing @40 N: {_pct(dist[40.0])}; recommended fires {fire} at 40/130/220 N"))
    runs = [_quiet(E.rollout, "square", "benign", "V2", cal="up") for _ in range(REPS)]
    up = th.cat([_flat(r["cal"]) for r in runs])
    v1 = th.cat([_flat(r["cal_v1"]) for r in runs])
    rec_up = _q(up, OP["EPS_UP"])
    rows.append(("E084_automaton.py", "EPS_UP", E.EPS_UP, rec_up,
                 f"V̄_up settled rest, load cleared: {_pct(up)}; current passes {_frac(up, E.EPS_UP):.0%}"))
    rows.append(("E084_automaton.py", "EPS_ABORT", E.EPS_ABORT, rec_up - ABORT_OFFSET["EPS_ABORT"], "EPS_UP - 0.12"))
    rows.append(("E084_automaton.py", "V1_UP", E.V1_UP, _q(v1, OP["V1_UP"]),
                 f"V̄_stand on the same robots: {_pct(v1)}; current passes {_frac(v1, E.V1_UP):.0%}"))


def walk():
    import E089_goal_walk as G
    saved = G.W_of
    G.W_of = lambda sched, t: 0.0                       # healthy walking, no load, 30 s
    parts = []
    for _ in range(REPS):
        r = _quiet(G.rollout, "pulse", "benign", "WALK-ONLY", cal=True)
        parts.append(th.stack(r["cal"]["vs"])[:, r["cal"]["alive"][-1]])   # robots that survive the 30 s
    G.W_of = saved
    vs = th.cat(parts, dim=1)                           # (T, n_healthy)
    grid = th.arange(-0.60, 0.0001, 0.005, device=vs.device)
    below = vs[:, :, None] < grid[None, None, :]        # (T, n, E)
    c = th.zeros(below.shape[1:], device=vs.device)
    trig = th.zeros(below.shape[1:], dtype=th.bool, device=vs.device)
    for t in range(below.shape[0]):
        c = th.where(below[t], c + 1, th.zeros_like(c))
        if t >= G.WARMUP:
            trig |= c >= G.K_DN_WALK
    false_rate = trig.float().mean(0)                   # per threshold, fraction of robots per 30 s
    ok = (false_rate <= OP["EPS_DN_WALK"]).nonzero()
    rec = float(grid[ok.max()]) if ok.numel() else float("nan")
    cur = float(false_rate[(grid - G.EPS_DN_WALK).abs().argmin()])
    rows.append(("E089_goal_walk.py", "EPS_DN_WALK", G.EPS_DN_WALK, rec,
                 f"false descents per 30 s of healthy walking: current {cur:.1%}; V̄_stand {_pct(vs.flatten().cpu())}"))


def leg():
    import E091_leg_walk as E
    r = _quiet(E.rollout, "legonly", "WALK-ONLY", n=64, cal=True)
    healthy = max(x[2] for x in r["cal"][100:249])
    derated = [x[1] for x in r["cal"][500:740]]
    der_med = sorted(derated)[len(derated) // 2]
    ok = healthy < E.ERR_DEG < max(x[2] for x in r["cal"][500:740])
    rows.append(("E091_leg_walk.py", "ERR_DEG", E.ERR_DEG, E.ERR_DEG if ok else (healthy + der_med) / 2,
                 f"residual healthy p99 {healthy:.4f}, derated median {der_med:.4f} "
                 f"({'keep' if ok else 'outside the gap — midpoint'})"))
    up = th.cat([_flat(_quiet(E.rollout, "legonly", "V2-REUSE", cal_up=True)["cal_up"]) for _ in range(REPS)])
    rec = _q(up, OP["EPS_UP_LEG"])
    rows.append(("E091_leg_walk.py", "EPS_UP_LEG", E.EPS_UP_LEG, rec,
                 f"V̄_up settled rest, leg healed: {_pct(up)}; current passes {_frac(up, E.EPS_UP_LEG):.0%}"))
    rows.append(("E091_leg_walk.py", "EPS_ABORT_LEG", E.EPS_ABORT_LEG, rec - ABORT_OFFSET["EPS_ABORT_LEG"],
                 "EPS_UP_LEG - 0.10"))


def payload():
    import E092_payload_walk as E
    up = th.cat([_flat(_quiet(E.rollout, "period", "V2", cal_up=True)["cal"]) for _ in range(REPS)])
    rec = _q(up, OP["EPS_UP_92"])
    rows.append(("E092_payload_walk.py", "EPS_UP_92", E.EPS_UP_92, rec,
                 f"V̄_up settled rest, load cleared: {_pct(up)}; current passes {_frac(up, E.EPS_UP_92):.0%}"))
    rows.append(("E092_payload_walk.py", "EPS_ABORT_92", E.EPS_ABORT_92, rec - ABORT_OFFSET["EPS_ABORT_92"],
                 "EPS_UP_92 - 0.15"))


def demos():
    from _paths import _ART
    art = os.path.expanduser(_ART)
    for script, const, path, rule in (
            ("E074_ramp.py", "EPS", "E074-hicom-demo/task2_value.json", "forced-switch window (run E074_tune.py)"),
            ("E075_ramp.py", "EPS", "E075-recal-eval/partA_value.json", "forced-switch window (run E074_tune.py)"),
            ("E078_ramp.py", "EPS", "E078-compound-demo/task2_value.json", "value-sweep midpoint")):
        f = os.path.join(art, path)
        if not os.path.exists(f):
            rows.append((script, const, None, None, f"no {path} — run: bash scripts/reproduce.sh certificates"))
            continue
        d = json.load(open(f))
        disc = d.get("discrim", d.get("discrim_recal"))
        flat = disc is not None and disc < 1.0
        rec = d["eps"] if rule == "value-sweep midpoint" and not flat else None
        ev = f"value-sweep midpoint {d['eps']:+.3f}"
        if disc is not None:
            ev += f", discrimination {disc:.2f}"
        if flat:
            ev += " — FLAT: no usable certificate, retrain"
        rows.append((script, const, None, rec, f"{ev}; rule: {rule}"))


GROUPS = {"standing": standing, "walk": walk, "leg": leg, "payload": payload, "demos": demos}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", choices=list(GROUPS), default=list(GROUPS))
    ap.add_argument("--reps", type=int, default=3, help="repeat each probe and pool the samples (default 3)")
    a = ap.parse_args()
    global REPS
    REPS = a.reps
    for g in a.only:
        print(f"[calibrate] {g} ...", flush=True)
        GROUPS[g]()
    print("\nconstant (file)                         current   recommended   evidence")
    for f, c, cur, rec, ev in rows:
        cs = "—" if cur is None else f"{cur:+.3f}"
        rs = "—" if rec is None else f"{rec:+.3f}"
        print(f"{c:13s} ({f:22s})  {cs:>8s}   {rs:>10s}   {ev}")
    print("\nTo apply: edit each constant at the top of the named file in experiments/, then rerun the targets.")


if __name__ == "__main__":
    main()
