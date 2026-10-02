"""Smoke tests for the headline pipeline: environment, checkpoints, and a short rollout of each
switching automaton (standing E084, payload walk E092, leg-fault walk E091).

    pytest -q tests/                 # everything (~3-4 min on an RTX 4070)
    pytest -q tests/ -m "not gpu"    # CPU-only checks (imports, registry, checksums)

These check that the code RUNS and the numbers are SANE at tiny N. They do not reproduce the paper
numbers — docs/REPRODUCE.md does that.
"""
import hashlib
import os

import pytest

CK_AUTOMATON = ("stand", "rest", "getup", "descend")
MANIFEST = "weights/MANIFEST.sha256"


def _have_weights():
    import E084_automaton as E
    return all(os.path.exists(E.CK[k]) for k in CK_AUTOMATON)


needs_weights = pytest.mark.skipif("not _have_weights()",
                                   reason="checkpoints missing — run scripts/fetch_bundles.sh")


def test_submodules_resolve():
    """safety_sb3 / robot_safety_sandbox / go2_atomic_skills must come from external/, not site-packages."""
    import go2_atomic_skills
    import robot_safety_sandbox
    import safety_sb3
    for mod, sub in ((safety_sb3, "external/safety-stable-baselines"),
                     (robot_safety_sandbox, "external/robot-safety-sandbox"),
                     (go2_atomic_skills, "external/go2_atomic_skills")):
        assert sub in mod.__file__, f"{mod.__name__} resolved to {mod.__file__}"
    from importlib.metadata import version
    assert version("safety_sb3") == "0.4.0", version("safety_sb3")


def test_tasks_registered():
    from robot_safety_sandbox import list_tasks
    names = {t if isinstance(t, str) else getattr(t, "task_id", str(t)) for t in list_tasks()}
    for tid in ("go2_weight_stand_hi", "go2_weight_rest_hi", "go2_weight_rest_hi_at_0", "go2_getup",
                "go2_descend", "go2_leg_stand", "go2_leg_rest", "go2_compound_stand", "go2_compound_rest"):
        assert tid in names, tid


@pytest.mark.weights
def test_weights_manifest():
    """Every checkpoint on disk matches the published sha256 (detects partial/mismatched downloads)."""
    if not os.path.exists("results"):
        pytest.skip("no results/ — run scripts/fetch_bundles.sh")
    bad, missing = [], []
    for line in open(MANIFEST):
        digest, rel = line.split()
        if not os.path.exists(rel):
            missing.append(rel)
            continue
        h = hashlib.sha256()
        with open(rel, "rb") as f:
            for blk in iter(lambda: f.read(1 << 20), b""):
                h.update(blk)
        if h.hexdigest() != digest:
            bad.append(rel)
    if len(missing) == sum(1 for _ in open(MANIFEST)):
        pytest.skip("checkpoints not fetched")
    assert not bad, f"checksum mismatch: {bad}"
    assert not missing, f"missing: {missing}"


@pytest.mark.gpu
@pytest.mark.weights
@needs_weights
def test_twins_load():
    from robot_safety_sandbox.eval.policies import load_twin
    import E084_automaton as E
    import torch
    for k in CK_AUTOMATON:
        model, norm = load_twin(E.CK[k], "cuda:0", quiet=True)
        obs = torch.zeros(4, 48, device="cuda:0")
        v = model.policy.predict_values(norm(obs))
        assert v.shape == (4, 1) and torch.isfinite(v).all(), k


@pytest.mark.gpu
def test_walker_loads():
    import E089_goal_walk as G
    w = G.Walker(4)
    assert w.net is not None


def _short(mod, steps):
    saved = mod.STEPS
    mod.STEPS = steps
    return saved


@pytest.mark.gpu
@pytest.mark.weights
@needs_weights
def test_standing_automaton_short():
    """E084: 4 s of the square load wave; REST-ONLY must survive (the anchor safe set)."""
    import E084_automaton as E
    saved = _short(E, 200)
    try:
        for arm in ("V2", "REST-ONLY"):
            r = E.rollout("square", "benign", arm)
            assert len(r["S"]) == 200 and 0.0 <= r["final"] <= 1.0
            if arm == "REST-ONLY":
                assert r["final"] > 0.95, r["final"]
    finally:
        E.STEPS = saved


@pytest.mark.gpu
@pytest.mark.weights
@needs_weights
def test_payload_walk_short():
    """E092: 8 s (through brake -> descend), every switching state reachable, no crash."""
    import E092_payload_walk as E
    saved = _short(E, 400)
    try:
        r = E.rollout("period", "V2", n=16)
        assert len(r["S"]) == 400 and set(r["deaths"]) == set(E.STATES)
        r = E.rollout("period", "REST-ONLY", n=16)
        assert r["safe"] > 0.95
    finally:
        E.STEPS = saved


@pytest.mark.gpu
@pytest.mark.weights
@needs_weights
def test_leg_walk_short():
    """E091: 8 s through the leg derate ramp; the residual detector must fire for the switching arm."""
    import E091_leg_walk as E
    saved = _short(E, 400)
    try:
        r = E.rollout("legonly", "V2-REUSE", n=16)
        assert r["trig_med"] is not None, "torque-saturation residual never fired"
    finally:
        E.STEPS = saved
