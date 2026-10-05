"""Recalibrate the switching thresholds for (re)trained policies.

    python scripts/calibrate.py                        # every group, 3 pooled seeds, ~7 min on an RTX 4070
    python scripts/calibrate.py --only standing walk   # a subset of: standing walk leg payload demos
    python scripts/calibrate.py --reps 1               # one seed, ~2.5 min (noisier tails)

The thresholds compare learned values (and one residual) against constants in odd_conditioned/scenarios.py,
automaton.py and certificates.py. Those constants sit at particular OPERATING POINTS of the published
networks — e.g. "the standing descent trigger fires on 7 % of robots standing under 40 N". This script
measures the same distributions on the networks in $ODD_CHECKPOINTS and recommends the value at the same
operating point. It changes nothing: it prints what to set, where, and the evidence. On the published
checkpoints the recommendations reproduce the published constants up to sampling noise — that is the check
that the rules are right.

Rules (docs/TRAINING.md, "Calibrating the switching thresholds"):
  standing descent trigger   7th percentile of V̄_stand, robots standing under 40 N
  standing get-up gate       94th percentile of V̄_up, settled REST after the load clears; abort = gate - 0.12
  direct return              21st percentile of V̄_stand on the same settled-REST robots
  weight-walk trigger        highest value with <= 1.6 % false descents per 30 s of healthy walking
  leg residual trigger       keep while it sits between the healthy and derated bands
  leg get-up gate            12th percentile of V̄_up, settled REST after the leg heals; abort = gate - 0.10
  payload get-up gate        35th percentile of V̄_up, settled REST after the load clears; abort = gate - 0.15
  demo ramps                 reported with their value-sweep midpoint and discrimination
"""
import argparse
import contextlib
import io
import json
import os
import sys

import torch as th

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from odd_conditioned import automaton as A  # noqa: E402
from odd_conditioned import certificates as C  # noqa: E402
from odd_conditioned.automaton import REST, TASK, rollout  # noqa: E402
from odd_conditioned.paths import OUTPUTS  # noqa: E402
from odd_conditioned.scenarios import SCENARIOS as SC  # noqa: E402
from odd_conditioned.sim import LOAD_H  # noqa: E402

# Operating points of the published calibration (measured on the published networks).
OP = {"standing_dn": 0.07, "standing_up": 0.94, "direct_up": 0.21, "walk_false": 0.016, "leg_up": 0.12,
      "payload_up": 0.35}
ABORT_OFFSET = {"standing": 0.12, "leg": 0.10, "payload": 0.15}

rows = []   # (where, constant, current, recommended, evidence)
REPS = 3


def _q(x, p):
    return float(th.quantile(x, p)) if x.numel() else float("nan")


def _frac(x, thr):
    return float((x >= thr).float().mean()) if x.numel() else float("nan")


def _pct(x):
    return "  ".join(f"p{int(p * 100)} {_q(x, p):+.3f}" for p in (0.05, 0.25, 0.5, 0.75, 0.95))


def _run(*a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return rollout(*a, trace=True, **k)


def _return_population(sc, rep):
    """V̄_up and V̄_stand of robots settled in REST while the belief reads the nominal ODD (return disabled):
    exactly the population the return gate sees."""
    tr = _run(sc, "odd", seed=rep, allow_return=False)["trace"]
    ok = th.tensor(tr["belief_ok"])[:, None]
    m = ok & tr["alive_pre"] & (tr["st_pre"] == REST) & (tr["settle"] >= A.SETTLE)
    return tr["vu_bar"][m], tr["vs_bar"][m]


def standing():
    base = SC["standing-square"]
    dist = {}
    for W in (40.0, 130.0, 220.0):
        sc = base.with_(odd=lambda t, _w=W: (_w, LOAD_H, None))
        parts = []
        for rep in range(REPS):
            tr = _run(sc, "task-only", seed=rep)["trace"]
            m = tr["alive"] & (tr["st"] == TASK)
            m[:base.arm_after] = False
            parts.append(tr["vs_bar"][m])
        dist[W] = th.cat(parts)
    rec = _q(dist[40.0], OP["standing_dn"])
    fire = " / ".join(f"{float((dist[w] < rec).float().mean()):.0%}" for w in dist)
    rows.append(("scenarios.py standing", "trigger_eps", base.trigger_eps, rec,
                 f"V̄_stand standing @40 N: {_pct(dist[40.0])}; recommended fires {fire} at 40/130/220 N"))
    pops = [_return_population(base, rep) for rep in range(REPS)]
    up, vs = th.cat([p[0] for p in pops]), th.cat([p[1] for p in pops])
    rec_up = _q(up, OP["standing_up"])
    rows.append(("scenarios.py standing", "eps_up", base.eps_up, rec_up,
                 f"V̄_up settled REST, load cleared: {_pct(up)}; current passes {_frac(up, base.eps_up):.0%}"))
    rows.append(("scenarios.py standing", "eps_abort", base.eps_abort, rec_up - ABORT_OFFSET["standing"],
                 f"eps_up - {ABORT_OFFSET['standing']}"))
    rows.append(("automaton.py", "DIRECT_EPS_UP", A.DIRECT_EPS_UP, _q(vs, OP["direct_up"]),
                 f"V̄_stand on the same robots: {_pct(vs)}; current passes {_frac(vs, A.DIRECT_EPS_UP):.0%}"))


def walk():
    base = SC["weight-walk-pulse"]
    sc = base.with_(odd=lambda t: (0.0, LOAD_H, None))             # healthy walking, no load, 30 s
    parts = []
    for rep in range(REPS):
        tr = _run(sc, "task-only", seed=rep)["trace"]
        parts.append(tr["vs_bar"][:, tr["alive"][-1]])              # robots that survive the 30 s
    vs = th.cat(parts, dim=1)                                       # (T, n_healthy)
    grid = th.arange(-0.60, 0.0001, 0.005)
    below = vs[:, :, None] < grid[None, None, :]
    c = th.zeros(below.shape[1:])
    trig = th.zeros(below.shape[1:], dtype=th.bool)
    for t in range(below.shape[0]):
        c = th.where(below[t], c + 1, th.zeros_like(c))
        if t >= base.arm_after:
            trig |= c >= base.trigger_k
    false_rate = trig.float().mean(0)
    ok = (false_rate <= OP["walk_false"]).nonzero()
    rec = float(grid[ok.max()]) if ok.numel() else float("nan")
    cur = float(false_rate[(grid - base.trigger_eps).abs().argmin()])
    rows.append(("scenarios.py weight-walk", "trigger_eps", base.trigger_eps, rec,
                 f"false descents per 30 s of healthy walking: current {cur:.1%}; V̄_stand {_pct(vs.flatten())}"))


def leg():
    sc = SC["leg-fault"]
    tr = _run(sc, "task-only", n=64, seed=0)["trace"]
    er, alive = tr["er_bar"], tr["alive_pre"]
    p99 = [_q(er[t][alive[t]], 0.99) for t in range(er.shape[0])]
    med = [_q(er[t][alive[t]], 0.5) for t in range(er.shape[0])]
    healthy = max(p99[100:249])                                     # 2-5 s: walking, leg healthy
    derated_med = sorted(med[500:740])[len(med[500:740]) // 2]      # 10-14.8 s: leg derated to 0.15
    keep = healthy < sc.trigger_eps < max(p99[500:740])
    rows.append(("scenarios.py leg-fault", "trigger_eps", sc.trigger_eps,
                 sc.trigger_eps if keep else (healthy + derated_med) / 2,
                 f"residual healthy p99 {healthy:.4f}, derated median {derated_med:.4f} "
                 f"({'keep' if keep else 'outside the gap — midpoint'})"))
    up = th.cat([_return_population(sc, rep)[0] for rep in range(REPS)])
    rec = _q(up, OP["leg_up"])
    rows.append(("scenarios.py leg-fault", "eps_up", sc.eps_up, rec,
                 f"V̄_up settled REST, leg healed: {_pct(up)}; current passes {_frac(up, sc.eps_up):.0%}"))
    rows.append(("scenarios.py leg-fault", "eps_abort", sc.eps_abort, rec - ABORT_OFFSET["leg"],
                 f"eps_up - {ABORT_OFFSET['leg']}"))


def payload():
    sc = SC["payload-period"]
    up = th.cat([_return_population(sc, rep)[0] for rep in range(REPS)])
    rec = _q(up, OP["payload_up"])
    rows.append(("scenarios.py payload", "eps_up", sc.eps_up, rec,
                 f"V̄_up settled REST, load cleared: {_pct(up)}; current passes {_frac(up, sc.eps_up):.0%}"))
    rows.append(("scenarios.py payload", "eps_abort", sc.eps_abort, rec - ABORT_OFFSET["payload"],
                 f"eps_up - {ABORT_OFFSET['payload']}"))


def demos():
    for ramp, sweep, readout, rule in (("weight-wide", "weight-wide", "stand_wide", "forced-switch window"),
                                       ("weight", "weight", "stand", "forced-switch window"),
                                       ("compound", "compound", "compound_stand", "value-sweep midpoint")):
        f = os.path.join(OUTPUTS, "certificates", f"value_{sweep}.json")
        cur = C.RAMPS[ramp].eps
        if not os.path.exists(f):
            rows.append((f"certificates.py RAMPS[{ramp}]", "eps", cur, None,
                         "no value sweep — run: python scripts/evaluate.py certificates"))
            continue
        q = json.load(open(f))["readouts"][readout]
        flat = q["discrim"] < 0.5                 # the unloaded-leg control reads ~0.0; usable certificates >= ~1
        rec = q["eps"] if rule == "value-sweep midpoint" and not flat else None
        ev = f"value-sweep midpoint {q['eps']:+.3f}, discrimination {q['discrim']:.2f}"
        if flat:
            ev += " — FLAT: no usable certificate, retrain"
        if rule == "forced-switch window":
            ev += "; place eps so the handoff fires inside the low-tip window of certificates/forced_switch.json"
        rows.append((f"certificates.py RAMPS[{ramp}]", "eps", cur, rec, ev))


GROUPS = {"standing": standing, "walk": walk, "leg": leg, "payload": payload, "demos": demos}


def main():
    global REPS
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", choices=list(GROUPS), default=list(GROUPS))
    ap.add_argument("--reps", type=int, default=3, help="seeds pooled per probe (default 3)")
    a = ap.parse_args()
    REPS = a.reps
    for g in a.only:
        print(f"[calibrate] {g} ...", flush=True)
        GROUPS[g]()
    print(f"\n{'where':32s} {'constant':14s} {'current':>8s} {'recommended':>12s}   evidence")
    for where, c, cur, rec, ev in rows:
        cs = "—" if cur is None else f"{cur:+.3f}"
        rs = "—" if rec is None else f"{rec:+.3f}"
        print(f"{where:32s} {c:14s} {cs:>8s} {rs:>12s}   {ev}")
    print("\nTo apply: edit the constant in odd_conditioned/<where>, then rerun the affected evaluate targets.")


if __name__ == "__main__":
    main()
