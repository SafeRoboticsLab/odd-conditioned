"""Draw the figures from saved results -> $ODD_OUTPUTS/figures/.

    python scripts/make_figures.py                      # every figure whose inputs exist
    python scripts/make_figures.py payload certificates # some groups: certificates payload leg standing
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from odd_conditioned.figures import FIGURES, make  # noqa: E402

if __name__ == "__main__":
    groups = sys.argv[1:] or list(FIGURES)
    bad = [g for g in groups if g not in FIGURES]
    if bad:
        raise SystemExit(f"unknown figure group(s) {bad}; choose from {list(FIGURES)}")
    sys.exit(0 if make(groups) else 1)
