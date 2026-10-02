"""E094 — seeds pass on the headline tables (3 fresh reps each; existing results.json = rep0).

Tables:
  e092   : payload-swap walking (pulse/period/dip x WALK-ONLY/REST-ONLY/ONE-WAY/V1/V2), metrics success+safe
  waves  : standing waves (square/sine x benign/gusty x V2/V1/ONE-WAY/STAND-ONLY/REST-ONLY), metric safe
  single : standing single excursions (pulse/period x benign/gusty, same arms), metric safe
Run per-table per-rep in SEPARATE processes (the single-excursion module patches E084 globals on import).
  python3 E094_seeds.py --table e092|waves|single --rep K     -> writes seeds_<table>_rep<K>.json
  python3 E094_seeds.py --aggregate                           -> seeds_summary.json + markdown table
"""
import os, sys, json, argparse
from _paths import _ART
sys.path.insert(0, "external/robot-safety-sandbox"); sys.path.insert(0, "experiments")
os.environ.setdefault("MUJOCO_GL", "egl")

ART = os.path.expanduser(_ART)
SEEDDIR = f"{ART}/PAPER-draft/seeds"
ARMS_W = ["WALK-ONLY", "REST-ONLY", "ONE-WAY", "V1", "V2"]
ARMS_S = ["V2", "V1", "ONE-WAY", "STAND-ONLY", "REST-ONLY"]


def run_e092(rep):
    import E092_payload_walk as E92
    out = {}
    for sched in ("pulse", "period", "dip"):
        for arm in ARMS_W:
            r = E92.rollout(sched, arm)
            out[f"{sched}|{arm}"] = {"success": r["success"], "safe": r["safe"]}
            print(f"[e092 rep{rep}] {sched} {arm}: {r['success']:.2f}/{r['safe']:.2f}", flush=True)
    return out


def run_waves(rep):
    import E084_automaton as E
    out = {}
    for cond in ("benign", "gusty"):
        for sched in ("square", "sine"):
            for arm in ARMS_S:
                r = E.rollout(sched, cond, arm)
                out[f"{cond}|{sched}|{arm}"] = {"safe": r["final"]}
                print(f"[waves rep{rep}] {cond}/{sched} {arm}: {r['final']:.2f}", flush=True)
    return out


def run_single(rep):
    import E086_single_pulse  # noqa: F401  (patches E084 globals: 24s schedules, belief gate)
    import E084_automaton as E
    out = {}
    for cond in ("benign", "gusty"):
        for sched in ("pulse", "period"):
            for arm in ARMS_S:
                r = E.rollout(sched, cond, arm)
                out[f"{cond}|{sched}|{arm}"] = {"safe": r["final"]}
                print(f"[single rep{rep}] {cond}/{sched} {arm}: {r['final']:.2f}", flush=True)
    return out


def aggregate():
    import numpy as np
    import glob
    summary = {}
    # rep0 = the existing headline results files
    rep0 = {
        "e092": {k: {"success": v["success"], "safe": v["safe"]}
                 for k, v in json.load(open(f"{ART}/E092-payload-walk/results.json")).items()},
        "waves": {k: {"safe": v["final"] if "final" in v else v["safe"]}
                  for k, v in json.load(open(f"{ART}/E084-automaton/results.json")).items()},
        "single": {k: {"safe": v["final"] if "final" in v else v["safe"]}
                   for k, v in json.load(open(f"{ART}/E084-automaton/results_single.json")).items()},
    }
    lines = ["# Seeds pass — mean ± sd over 4 independent reps (N=256 each; rep0 = headline run)\n"]
    for table in ("e092", "waves", "single"):
        reps = [rep0[table]]
        for f in sorted(glob.glob(f"{SEEDDIR}/seeds_{table}_rep*.json")):
            reps.append(json.load(open(f)))
        keys = [k for k in reps[0] if all(k in r for r in reps)]
        summary[table] = {}
        lines.append(f"\n## {table}  ({len(reps)} reps)\n")
        metrics = ["success", "safe"] if table == "e092" else ["safe"]
        hdr = "| scenario | arm | " + " | ".join(f"{m} (μ±σ)" for m in metrics) + " |"
        lines += [hdr, "|" + "---|" * (2 + len(metrics))]
        for k in keys:
            vals = {m: [r[k][m] for r in reps if m in r[k]] for m in metrics}
            summary[table][k] = {m: {"mean": float(np.mean(v)), "sd": float(np.std(v)), "n": len(v)}
                                 for m, v in vals.items()}
            parts = "|".join(f" {np.mean(v):.2f} ± {np.std(v):.02f} " for v in vals.values())
            sc, arm = k.rsplit("|", 1)
            lines.append(f"| {sc} | {arm} |{parts}|")
    json.dump(summary, open(f"{SEEDDIR}/seeds_summary.json", "w"), indent=1)
    open(f"{SEEDDIR}/seeds_summary.md", "w").write("\n".join(lines))
    print(f"aggregated -> {SEEDDIR}/seeds_summary.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", choices=["e092", "waves", "single"])
    ap.add_argument("--rep", type=int, default=1)
    ap.add_argument("--aggregate", action="store_true")
    a = ap.parse_args()
    os.makedirs(SEEDDIR, exist_ok=True)
    if a.aggregate:
        aggregate(); sys.exit(0)
    out = {"e092": run_e092, "waves": run_waves, "single": run_single}[a.table](a.rep)
    json.dump(out, open(f"{SEEDDIR}/seeds_{a.table}_rep{a.rep}.json", "w"))
    print(f"saved seeds_{a.table}_rep{a.rep}.json")
