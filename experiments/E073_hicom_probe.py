"""E073 (T003) Task-2 VALIDATION PROBE — does the HIGH-CoM (inverted-pendulum) load physics destabilize?

Zero-shot: the OLD stand policy (go2_weight_stand_adv, trained under the pure-force E070 load) on the NEW
physics, driven via the eval channel (env.base_load = [0,0,-W] on the outer bridge + env.mj._weight_h = h on
the inner env, which base.py turns into the torque τ = R(quat)·[0,0,h] × [0,0,-W]).

Protocol: W held from t=0; 10 N ambient lateral pull; a GUST (+25 N lateral, total 35 N) for 0.5 s at t=3 s.
Metric: tip fraction (per-env tilt max(|pg_x|,|pg_y|) > TIP_TILT before the first reset) + mean post-gust tilt.
EXPECT: h=0 mostly survives (E070 "planted"), h=0.25 tips substantially (inverted pendulum). W=0 IDENTICAL
across h (no load ⇒ no torque). Grid: W ∈ {0,60,150} × h ∈ {0,0.25}; escalate to 0.35 only if 0.25 is inert.
"""
import os, sys, io, contextlib
import torch as th
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "external/robot-safety-sandbox")
import numpy as np
from robot_safety_sandbox import make_tensor, spec
from robot_safety_sandbox.eval.policies import load_twin

DEV, N, DT, STEPS = "cuda:0", 256, 0.02, 250
T_GUST0, T_GUST1 = 150, 175                   # gust window: t ∈ [3.0, 3.5] s  (25 steps @ 50 Hz)
TIP_TILT = 0.60                               # tilt (|proj-gravity xy|) beyond which we call it a topple
AMBIENT_SCALE = 0.20                          # 0.2 * 50 N force_max = 10 N ambient lateral pull
GUST_SCALE = 0.70                             # 0.7 * 50 N = 35 N during the gust (10 ambient + 25 gust)
STAND_CK = "results/go2_weight_runs/go2_weight_stand_adv/checkpoints/model_49999872_steps.zip"
PULL = th.tensor([0., 1., 0.])               # lateral pull/gust direction


def probe(W, h):
    # go2_weight_stand_at_{W}: pins _weight_W=W so the 48-dim weight_theta obs the policy trained with is
    # correct at this test load; we additionally set _weight_h each step (the reset event zeroes it).
    task = f"go2_weight_stand_at_{int(W)}"
    with contextlib.redirect_stdout(io.StringIO()):
        env = make_tensor(task, N, DEV, adversary=True)
        model, norm = load_twin(STAND_CK, DEV, quiet=True)
    inner = env.mj
    dstb = PULL.to(DEV)[None].expand(N, spec(task).dstb_dim).contiguous()
    obs = env.reset()
    done_seen = th.zeros(N, dtype=th.bool, device=DEV)
    tipped = th.zeros(N, dtype=th.bool, device=DEV)
    tilt_post = []
    for t in range(STEPS):
        inner._weight_h = th.full((N,), float(h), device=DEV)       # HIGH-CoM lever (set each step: resets zero it)
        env.base_load = th.tensor([0., 0., -float(W)], device=DEV)[None].expand(N, 3)
        env.force_scale = ((GUST_SCALE if T_GUST0 <= t < T_GUST1 else AMBIENT_SCALE)
                           * th.ones(N, device=DEV))
        with th.no_grad():
            a = th.clamp(model.policy._predict(norm(obs), deterministic=True), -1, 1)
        obs, g, dones, touts, l = env.step_tensor(th.cat([a, dstb], dim=1))
        pg = inner.scene["robot"].data.projected_gravity_b
        tilt = th.maximum(pg[:, 0].abs(), pg[:, 1].abs())
        tipped |= (tilt > TIP_TILT) & ~done_seen                     # topple latch (pre first reset)
        done_seen |= (dones & ~touts)
        if t >= T_GUST1:
            tilt_post.append(float(th.where(~done_seen, tilt, th.zeros_like(tilt)).mean()))
    env.close()
    return float(tipped.float().mean()), (float(np.mean(tilt_post)) if tilt_post else float("nan"))


if __name__ == "__main__":
    HS = [0.0, 0.25]
    if len(sys.argv) > 1:
        HS = [float(x) for x in sys.argv[1:]]
    print(f"HIGH-CoM validation: OLD stand policy zero-shot, gust +25N @ t=3s. tip = tilt>{TIP_TILT}.\n")
    print(f"{'W(N)':>5} | " + " ".join(f"h={h:<4} tip  tilt" for h in HS))
    for W in (0, 60, 150):
        cells = []
        for h in HS:
            tip, tilt = probe(W, h)
            cells.append(f"{tip:>10.2f} {tilt:>5.2f}")
        print(f"{W:>5} | " + " ".join(cells))
    print("\nEXPECT: h=0 survives (low tip), h=0.25 tips at W=60/150; W=0 identical across h.")
