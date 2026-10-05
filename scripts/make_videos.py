"""Render the demo videos -> $ODD_OUTPUTS/videos/ (slow; needs EGL).

    python scripts/make_videos.py payload     # needs payload/results.json (evaluate.py payload)   ~15 min
    python scripts/make_videos.py leg                                                            ~3 min
    python scripts/make_videos.py standing                                                       ~6 min
    python scripts/make_videos.py compound    # needs certificates/ramp_compound.json            ~4 min
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from odd_conditioned.videos import VIDEOS  # noqa: E402

if __name__ == "__main__":
    names = sys.argv[1:] or list(VIDEOS)
    for v in names:
        if v not in VIDEOS:
            raise SystemExit(f"unknown video '{v}'; choose from {list(VIDEOS)}")
        VIDEOS[v]()
