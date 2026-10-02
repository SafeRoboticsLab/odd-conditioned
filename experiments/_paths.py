"""Shared output location for the experiment scripts.

Run every script from the REPO ROOT (``python experiments/E092_payload_walk.py``): checkpoint paths
(``results/...``) and submodule imports (``external/...``) are resolved relative to the working
directory.

``ODD_ARTIFACTS`` is where scripts write their browsable outputs (results.json, figures, videos, one
``E0XX-slug/`` directory per experiment). It defaults to ``~/artifacts/odd-conditioned``, the layout
the reference results were produced in. Point it elsewhere to keep a reproduction from overwriting
the reference outputs:

    ODD_ARTIFACTS=/tmp/odd-repro python experiments/E092_payload_walk.py
"""
import os

_ART = os.environ.get("ODD_ARTIFACTS", "~/artifacts/odd-conditioned").rstrip("/")

if not os.path.isdir("external/robot-safety-sandbox/robot_safety_sandbox"):
    raise SystemExit("run the experiment scripts from the odd-conditioned repo root "
                     "(and run `git submodule update --init` first)")
