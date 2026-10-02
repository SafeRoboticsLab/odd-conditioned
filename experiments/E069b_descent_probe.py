"""E069b (T003, G1 fix) — SCRIPTED-DESCENT physics probe. The soft generalist never descends (its target is
satisfiable standing), so E069's rest column tested nothing. Here: script the legs folding stance→prone over
~1s under load W + the same 10N lateral pull, and measure GENTLENESS at first belly contact:
  v_touch  = |vertical base velocity| when non-foot contact first exceeds 5N  (gentle < ~0.3 m/s)
  tilt_max = max lateral/pitch tilt during the descent (upright-ish descent vs topple)
Termination-proof (measured at first touch, before any done). Sweeps fold depth per W, reports the best-gentle.
GATE: at W where standing fails (>=120N), a fold with small v_touch + small tilt exists => gentle descent is
PHYSICALLY feasible => the handoff window is real.
"""
import os, sys, io, contextlib
import torch as th
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "external/robot-safety-sandbox")
import numpy as np
from robot_safety_sandbox import make_tensor, spec

DEV, N, DT, STEPS = "cuda:0", 128, 0.02, 200
T0 = 30                                     # load + descent start
PULL = th.tensor([0., 1., 0.])
WS = [0, 90, 120, 150, 200, 250]
# fold targets in ACTION space (offsets from default pose / ctrl_gain): [hips(4), thighs(4), calves(4)]
FOLDS = {"shallow": (0.0, 0.15, -0.20), "mid": (0.0, 0.25, -0.30), "deep": (0.0, 0.35, -0.45)}
RAMP = 50                                   # fold ramp-in steps (1s)


def probe(W, fold, ramp=RAMP):
    hip, th_off, ca_off = FOLDS[fold]
    with contextlib.redirect_stdout(io.StringIO()):
        env = make_tensor("go2_weak_leg_blind_soft", N, DEV, adversary=True)   # 80N termination; margins unused
    inner = env.mj
    env.force_scale = 0.2 * th.ones(N, device=DEV)                  # same 10N pull as the stand probe
    dstb = (PULL).to(DEV)[None].expand(N, spec("go2_weak_leg_blind_soft").dstb_dim).contiguous()
    env.reset()
    target = th.zeros(N, 12, device=DEV)
    target[:, 4:8] = th_off; target[:, 8:12] = ca_off; target[:, 0:4] = hip
    touched = th.zeros(N, dtype=th.bool, device=DEV)
    v_touch = th.full((N,), float("nan"), device=DEV)
    tiltmax = th.zeros(N, device=DEV)
    for t in range(STEPS):
        env.base_load = (th.tensor([0., 0., -float(W)], device=DEV)[None].expand(N, 3)
                         if t >= T0 else None)
        frac = min(1.0, max(0.0, (t - T0) / ramp))
        a = target * frac                                            # ramp the fold in
        _o, g, dones, touts, l = env.step_tensor(th.cat([a, dstb], dim=1))
        s = inner.scene["nonfoot_ground_touch"]
        fh = s.data.force_history if s.data.force_history is not None else s.data.force
        force = th.norm(fh, dim=-1).flatten(1).amax(1)
        d = inner.scene["robot"].data
        vz = d.root_link_lin_vel_w[:, 2]
        pg = d.projected_gravity_b
        tilt = th.maximum(pg[:, 0].abs(), pg[:, 1].abs())
        fresh = (force > 5.0) & ~touched
        v_touch = th.where(fresh, vz.abs(), v_touch)
        touched |= fresh
        tiltmax = th.where(~touched, th.maximum(tiltmax, tilt), tiltmax)   # tilt DURING descent (pre-touch)
    env.close()
    tv = v_touch[touched]
    return {"touch_frac": float(touched.float().mean()),
            "v_med": float(tv.median()) if len(tv) else float("nan"),
            "v_p90": float(tv.quantile(0.9)) if len(tv) else float("nan"),
            "tilt_med": float(tiltmax[touched].median()) if bool(touched.any()) else float("nan")}


if __name__ == "__main__":
    print(f"{'W(N)':>5} {'fold':>8} | {'touched':>7} {'v_med':>6} {'v_p90':>6} {'tilt':>5}   (gentle: v<~0.3 m/s, tilt<~0.5)")
    for W in WS:
        best = None
        for fold in FOLDS:
            r = probe(W, fold)
            print(f"{W:>5} {fold:>8} | {r['touch_frac']:>7.2f} {r['v_med']:>6.2f} {r['v_p90']:>6.2f} {r['tilt_med']:>5.2f}")
            if r["touch_frac"] > 0.5 and (best is None or r["v_med"] < best):
                best = r["v_med"]
        print(f"      -> best gentle v_med at W={W}: {best}")
    print("\nGATE: at W>=120 (standing fails), a fold with small v_med + tilt => gentle descent physically feasible.")
