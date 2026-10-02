"""Tests run from the repo root (checkpoint paths and submodule imports are repo-root relative)."""
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
for p in ("experiments", "external/go2_atomic_skills", "external/robot-safety-sandbox",
          "external/safety-stable-baselines", REPO):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("MUJOCO_GL", "egl")


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: needs a CUDA GPU (simulation rollouts)")
    config.addinivalue_line("markers", "weights: needs the trained checkpoints (scripts/fetch_bundles.sh)")


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
