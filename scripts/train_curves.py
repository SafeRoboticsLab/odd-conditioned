"""Print training curves from sandbox training logs (the SB3 tables), side by side.

    python scripts/train_curves.py <log> [<log> ...] [--every 5e6] [--keys failure,force,ev]

For each log: one row per `--every` timesteps (the table closest to it) with safety/failure_rate,
game/force_scale_mean and train/explained_variance. Use it to compare two runs of the same recipe
(e.g. a shipped run against a retrain) and see where they diverge.
"""
import argparse
import re

KEYS = {"failure": "safety/failure_rate", "force": "game/force_scale_mean", "ev": "train/explained_variance",
        "std": "train/std", "kl": "train/approx_kl", "vloss": "train/value_loss"}


def tables(path):
    """SB3 log tables: section headers `| game/ |` then indented `|    name | value |` rows."""
    rows, cur, section = [], {}, ""
    for line in open(path, errors="ignore"):
        h = re.match(r"\| (\w+/)\s+\|\s+\|", line)
        if h:
            section = h.group(1)
            continue
        m = re.match(r"\|\s{4}([\w/.]+)\s+\|\s+([-+\d.e]+)\s+\|", line)
        if m:
            try:
                v = float(m.group(2))
            except ValueError:
                continue
            cur[m.group(1)] = v
            cur[section + m.group(1)] = v
        elif line.startswith("---") and cur.get("time/total_timesteps") is not None:
            rows.append(cur)
            cur, section = {}, ""
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--every", type=float, default=5e6)
    ap.add_argument("--keys", default="failure,force,ev")
    a = ap.parse_args()
    keys = [KEYS[k] for k in a.keys.split(",")]
    runs = {p: tables(p) for p in a.logs}
    end = max((r[-1]["time/total_timesteps"] for r in runs.values() if r), default=0)
    marks = [a.every * i for i in range(1, int(end // a.every) + 2)]
    names = [p.split("/")[-1] for p in a.logs]
    print("steps(M) | " + " || ".join(f"{n}: " + "/".join(k.split("/")[-1][:8] for k in keys) for n in names))
    for m in marks:
        cells = []
        for p in a.logs:
            r = runs[p]
            if not r or r[0]["time/total_timesteps"] > m + a.every:
                cells.append("—")
                continue
            t = min(r, key=lambda x: abs(x["time/total_timesteps"] - m))
            cells.append(" ".join(f"{t.get(k, float('nan')):+.3f}" for k in keys))
        print(f"{m / 1e6:7.0f} | " + " || ".join(cells))


if __name__ == "__main__":
    main()
