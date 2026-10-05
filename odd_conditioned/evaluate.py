"""Run the reported experiments and save their results under $ODD_OUTPUTS (see paths.py).

    payload       payload-swap walking: pulse / period / dip x 5 methods            -> payload/
    leg           leg-fault walking x 5 methods                                     -> leg/
    standing      standing automaton: load waves and single excursions x 6 methods  -> standing/
    weight-walk   walking under a 220 N excursion (boundary case) x 3 pushes          -> weight_walk/
    certificates  value sweeps, handoff ramps, forced switch, region grid, compound matrix -> certificates/
    seeds         payload + standing tables at further seeds, then mean ± sd        -> seeds/

Every rollout is seeded (default 0): rerunning a target on the same machine reproduces it (up to rare GPU
floating-point nondeterminism in contact-rich steps, which changes individual robots, not the statistics).
"""
import glob
import json
import os

import numpy as np

from . import certificates as C
from .automaton import label, rollout
from .paths import output_dir
from .scenarios import SCENARIOS

WALK_METHODS = ("task-only", "rest-only", "one-way", "direct", "odd")
STANDING_METHODS = ("odd", "odd-rest-descent", "direct", "one-way", "task-only", "rest-only")
STANDING = {"waves": ("square", "sine"), "single": ("pulse", "period")}
SUMMARY_KEYS = ("S", "SUC", "safe", "success", "afford", "deaths", "early_triggers", "trigger_median",
                "t_goal_median", "scenario", "method", "push", "n", "seed")


def _summary(r):
    return {k: r[k] for k in SUMMARY_KEYS if k in r}


def _dump(obj, *path):
    f = os.path.join(output_dir(*path[:-1]), path[-1])
    with open(f, "w") as fh:
        json.dump(obj, fh)
    print(f"saved -> {f}", flush=True)


def _deaths(r):
    return " ".join(f"{k}:{v}" for k, v in r["deaths"].items() if v)


def _walk_row(m, task, r):
    tg = f"{r['t_goal_median']:.1f}s" if r.get("t_goal_median") else "--"
    tt = f"{r['trigger_median']:.1f}s" if r.get("trigger_median") else "--"
    print(f"{label(m, task):>26} {r['success']:>8.2f} {r['safe']:>6.2f} {tg:>7} {tt:>7} {r['early_triggers']:>6}"
          f"  {_deaths(r)}", flush=True)


def _walk_header():
    print(f"{'method':>26} {'success':>8} {'safe':>6} {'t_goal':>7} {'t_trig':>7} {'early':>6}  deaths by mode",
          flush=True)


def _traj(data):
    return {f"{m}_{k}": v for m, r in data.items() for k, v in r["record"].items() if k != "every"}


def payload(seed=0, n=256, profiles=("pulse", "period", "dip"), record=True, out=("payload",)):
    res = {}
    for prof in profiles:
        sc = SCENARIOS[f"payload-{prof}"]
        print(f"\n===== {sc.title}  (goal {sc.goal:.0f} m, crate on during "
              f"[{sc.params['t0']:.0f}, {sc.params['t1']:.0f}) s, seed {seed}) =====", flush=True)
        _walk_header()
        data = {}
        for m in WALK_METHODS:
            r = rollout(sc, m, n=n, seed=seed, record=record)
            data[m] = r
            res[f"{prof}|{m}"] = _summary(r)
            _walk_row(m, "walk", r)
        if record:
            np.savez_compressed(os.path.join(output_dir(*out), f"traj_{prof}.npz"), **_traj(data))
    _dump(res, *out, f"results_seed{seed}.json" if seed else "results.json")
    return res


def leg(seed=0, n=256, record=True, out=("leg",)):
    sc = SCENARIOS["leg-fault"]
    print(f"\n===== {sc.title}  (goal {sc.goal:.0f} m, seed {seed}) =====", flush=True)
    _walk_header()
    res, data = {}, {}
    for m in WALK_METHODS:
        r = rollout(sc, m, n=n, seed=seed, record=record)
        data[m] = r
        res[f"fault|{m}"] = _summary(r)
        _walk_row(m, "walk", r)
    if record:
        np.savez_compressed(os.path.join(output_dir(*out), "traj.npz"), **_traj(data))
    _dump(res, *out, f"results_seed{seed}.json" if seed else "results.json")
    return res


def standing(seed=0, n=256, tables=("waves", "single"), methods=STANDING_METHODS, out=("standing",)):
    allres = {}
    for table in tables:
        res = {}
        for push in ("benign", "gusty"):
            for prof in STANDING[table]:
                sc = SCENARIOS[f"standing-{prof}"]
                print(f"\n===== {sc.title} / {push} push  (seed {seed}) =====", flush=True)
                print(f"{'method':>32} {'safe':>6} {'stand|alive':>12}  deaths by mode", flush=True)
                for m in methods:
                    r = rollout(sc, m, push=push, n=n, seed=seed)
                    res[f"{push}|{prof}|{m}"] = _summary(r)
                    print(f"{label(m, 'stand'):>32} {r['safe']:>6.2f} {r['afford']:>12.2f}  {_deaths(r)}", flush=True)
        _dump(res, *out, f"{table}_seed{seed}.json" if seed else f"{table}.json")
        allres[table] = res
    return allres


def weight_walk(seed=0, n=256, out=("weight_walk",)):
    res = {}
    for push in ("benign", "medium", "gusty"):
        for prof in ("pulse", "period"):
            sc = SCENARIOS[f"weight-walk-{prof}"]
            print(f"\n===== {sc.title} / {push} push  (seed {seed}) =====", flush=True)
            _walk_header()
            for m in WALK_METHODS:
                r = rollout(sc, m, push=push, n=n, seed=seed)
                res[f"{push}|{prof}|{m}"] = _summary(r)
                _walk_row(m, "walk", r)
    _dump(res, *out, "results.json")
    return res


def certificates(seed=0, out=("certificates",)):
    for key, sw in C.SWEEPS.items():
        r = C.value_sweep(sw, seed=seed)
        C.print_sweep(r)
        _dump(r, *out, f"value_{key}.json")
    for key, rp in C.RAMPS.items():
        r = C.handoff_ramp(rp, seed=seed)
        C.print_ramp(r)
        _dump(r, *out, f"ramp_{key}.json")
    r = C.forced_switch(seed=seed)
    print("forced switch (wide-stand weight ramp): W_switch -> afford / tip / slam")
    for w, row in r["rows"].items():
        print(f"  {w:>5}: {row['afford_pre']:.2f} / {row['tip']:.2f} / {row['slam']:.2f}")
    _dump(r, *out, "forced_switch.json")
    print("region grid (failure fraction, rows = push)")
    _dump(C.region_grid(seed=seed), *out, "region_grid.json")
    print("compound matrix")
    _dump(C.compound_matrix(seed=seed), *out, "compound_matrix.json")


def seeds(extra=(1, 2, 3), n=256):
    """The headline tables at further seeds (seed 0 is the main run), then mean ± sd across all seeds."""
    for s in extra:
        payload(seed=s, n=n, record=False, out=("seeds",))
        standing(seed=s, n=n, out=("seeds",))
    aggregate()


def aggregate():
    from .paths import OUTPUTS
    tables = {"payload": ("payload/results.json", "seeds/results_seed*.json", ("success", "safe")),
              "waves": ("standing/waves.json", "seeds/waves_seed*.json", ("safe",)),
              "single": ("standing/single.json", "seeds/single_seed*.json", ("safe",))}
    summary, lines = {}, ["# Headline tables over seeds — mean ± sd (N = 256 robots per run)\n"]
    for table, (main, pattern, metrics) in tables.items():
        files = [os.path.join(OUTPUTS, main)] + sorted(glob.glob(os.path.join(OUTPUTS, pattern)))
        reps = [json.load(open(f)) for f in files if os.path.exists(f)]
        if not reps:
            continue
        keys = [k for k in reps[0] if all(k in r for r in reps)]
        summary[table] = {}
        lines += [f"\n## {table} ({len(reps)} seeds)\n",
                  "| scenario | method | " + " | ".join(metrics) + " |", "|" + "---|" * (2 + len(metrics))]
        for k in keys:
            vals = {m: [r[k][m] for r in reps] for m in metrics}
            summary[table][k] = {m: {"mean": float(np.mean(v)), "sd": float(np.std(v)), "n": len(v)}
                                 for m, v in vals.items()}
            sc, m = k.rsplit("|", 1)
            task = "walk" if table == "payload" else "stand"
            lines.append(f"| {sc} | {label(m, task)} | "
                         + " | ".join(f"{np.mean(v):.2f} ± {np.std(v):.2f}" for v in vals.values()) + " |")
    _dump(summary, "seeds", "summary.json")
    with open(os.path.join(output_dir("seeds"), "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


TARGETS = {"payload": payload, "leg": leg, "standing": standing, "weight-walk": weight_walk,
           "certificates": certificates, "seeds": seeds, "aggregate": aggregate}
