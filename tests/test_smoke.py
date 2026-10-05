"""Smoke tests: environment, checkpoints, and short rollouts of every scenario family.

    pytest -q tests/                 # everything (~2 min on an RTX 4070)
    pytest -q tests/ -m "not gpu"    # CPU-only checks

They check that the code RUNS and the numbers are SANE at small N; scripts/reproduce.sh reproduces the results.
"""
import hashlib
import os

import pytest

from odd_conditioned.paths import CHECKPOINTS
from odd_conditioned.policies import AUTOMATON, POLICIES

MANIFEST = "weights/MANIFEST.sha256"
TASKS = ("go2_weight_stand_hi", "go2_weight_rest_hi", "go2_weight_rest_hi_at_0", "go2_weight_unified_hi",
         "go2_weight_unified_disc_hi", "go2_getup", "go2_descend", "go2_leg_stand", "go2_leg_rest",
         "go2_compound_stand", "go2_compound_rest", "go2_compound_rest_at_100")


def _have(names):
    return all(os.path.exists(os.path.join(CHECKPOINTS, n, "model.zip")) for n in names)


needs_weights = pytest.mark.skipif(not _have(AUTOMATON), reason="checkpoints missing — scripts/fetch_weights.sh")


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
    missing = [t for t in TASKS if t not in names]
    assert not missing, missing


def test_schedules():
    from odd_conditioned.scenarios import LEG, PAYLOAD, SCENARIOS
    from odd_conditioned.sim import DT
    for name, sc in SCENARIOS.items():
        Ws = [sc.odd(t)[0] for t in range(sc.steps)]
        assert 0.0 <= min(Ws) and max(Ws) <= 240.0 + 1e-9, name
    assert max(SCENARIOS["payload-period"].odd(t)[0] for t in range(2000)) > 200
    W = [SCENARIOS["payload-dip"].odd(t)[0] for t in range(2000)]
    assert min(W[int(9.7 / DT) - 5:int(9.7 / DT) + 5]) < 130                     # the dip dips below the gate
    th = [SCENARIOS["leg-fault"].odd(t)[2] for t in range(1500)]
    assert th[0] == 1.0 and min(th) == LEG["theta_lo"] and th[-1] == 1.0
    assert SCENARIOS["payload-pulse"].odd(int(PAYLOAD["t0"] / DT))[0] == PAYLOAD["W_hi"]


@pytest.mark.weights
def test_weights_manifest():
    """Installed checkpoints match the published sha256 (skips for a retrained set)."""
    if not os.path.isdir(CHECKPOINTS):
        pytest.skip("no checkpoints/ — scripts/fetch_weights.sh")
    bad, missing, total = [], [], 0
    for line in open(MANIFEST):
        digest, rel = line.split()
        total += 1
        path = os.path.join(CHECKPOINTS, rel.split("/", 1)[1])
        if not os.path.exists(path):
            missing.append(rel)
            continue
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for blk in iter(lambda: f.read(1 << 20), b""):
                h.update(blk)
        if h.hexdigest() != digest:
            bad.append(rel)
    if len(missing) == total:
        pytest.skip("checkpoints not installed")
    if bad and len(bad) + len(missing) == total:
        pytest.skip("no checkpoint matches the published set — a retrained set")
    assert not bad, f"checksum mismatch (corrupted or partial download?): {bad}"
    assert not missing, f"missing: {missing}"


@pytest.mark.gpu
@pytest.mark.weights
@pytest.mark.skipif(not _have(POLICIES), reason="checkpoints missing")
def test_twins_load():
    import torch
    from odd_conditioned.policies import Twin
    for k in POLICIES:
        tw = Twin(k)
        v = tw.model.policy.predict_values(tw.norm(torch.zeros(4, 48, device="cuda:0")))
        assert v.shape == (4, 1) and torch.isfinite(v).all(), k


@pytest.mark.gpu
def test_walker_loads():
    from odd_conditioned.policies import Walker
    assert Walker(4).net is not None


def _short(name, steps):
    from odd_conditioned.scenarios import SCENARIOS
    return SCENARIOS[name].with_(steps=steps)


@pytest.mark.gpu
@pytest.mark.weights
@needs_weights
def test_standing_short():
    """4 s of the square load wave; REST-ONLY must survive (the safe set is real)."""
    from odd_conditioned.automaton import rollout
    for m in ("odd", "rest-only"):
        r = rollout(_short("standing-square", 200), m, n=32)
        assert len(r["S"]) == 200 and 0.0 <= r["safe"] <= 1.0
        if m == "rest-only":
            assert r["safe"] > 0.95, r["safe"]


@pytest.mark.gpu
@pytest.mark.weights
@needs_weights
def test_payload_short():
    """10 s of the payload walk: the crate arrives at 5 s and the filter brakes and descends."""
    from odd_conditioned.automaton import rollout
    r = rollout(_short("payload-pulse", 500), "odd", n=32, record=True)
    assert r["early_triggers"] == 0
    assert r["trigger_median"] is not None and 5.0 <= r["trigger_median"] < 6.0, r["trigger_median"]
    assert r["record"]["traj"].shape == (125, 32, 2)


@pytest.mark.gpu
@pytest.mark.weights
@needs_weights
def test_leg_short():
    """10 s of the leg-fault walk: the residual trigger fires after the fault starts, never before."""
    from odd_conditioned.automaton import rollout
    r = rollout(_short("leg-fault", 500), "odd", n=32)
    assert r["early_triggers"] == 0
    assert r["trigger_median"] is not None and r["trigger_median"] >= 5.0


@pytest.mark.gpu
@pytest.mark.weights
@needs_weights
def test_seeded_rollout_repeats():
    """A seeded rollout repeats on the same machine (rare GPU float drift aside, which a short run avoids)."""
    from odd_conditioned.automaton import rollout
    a = rollout(_short("standing-sine", 100), "odd", n=16, seed=5)
    b = rollout(_short("standing-sine", 100), "odd", n=16, seed=5)
    assert a["S"] == b["S"] and a["deaths"] == b["deaths"]
