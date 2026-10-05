"""Run the reported experiments; results land in $ODD_OUTPUTS (default <repo>/outputs).

    python scripts/evaluate.py payload            # payload-swap walking: pulse / period / dip      ~6 min
    python scripts/evaluate.py leg                # leg-fault walking                               ~2 min
    python scripts/evaluate.py standing           # standing automaton: waves + single excursions   ~30 min
    python scripts/evaluate.py weight-walk        # walking under a 220 N excursion (boundary)      ~15 min
    python scripts/evaluate.py certificates       # value sweeps, ramps, region grid, compound      ~25 min
    python scripts/evaluate.py seeds              # payload + standing at seeds 1-3, mean ± sd      ~1.5 h
    python scripts/evaluate.py payload --seed 1 --n 128

Runtimes: one RTX 4070. Run from anywhere after `source activate.sh`.
"""
import argparse
import inspect
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from odd_conditioned.evaluate import TARGETS  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", choices=list(TARGETS))
    ap.add_argument("--seed", type=int, default=None, help="rollout seed (default 0)")
    ap.add_argument("--n", type=int, default=None, help="robots per rollout (default 256)")
    a = ap.parse_args()
    fn = TARGETS[a.target]
    params = inspect.signature(fn).parameters
    kw = {k: v for k, v in (("seed", a.seed), ("n", a.n)) if v is not None and k in params}
    fn(**kw)


if __name__ == "__main__":
    main()
