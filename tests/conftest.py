"""Tests run against the submodules under external/ (``source activate.sh`` first)."""
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.environ.setdefault("MUJOCO_GL", "egl")


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: needs a CUDA GPU (simulation rollouts)")
    config.addinivalue_line("markers", "weights: needs the trained checkpoints (scripts/fetch_weights.sh)")


def pytest_collection_modifyitems(config, items):
    try:
        import torch
        has_gpu = torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        has_gpu = False
    if has_gpu:
        return
    skip = pytest.mark.skip(reason="no CUDA GPU")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)
