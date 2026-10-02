"""E069 (T003, G1) — WEIGHT-LADDER PHYSICS GATE: does the handoff window exist?

Sweep a constant downward base load W (the new `env.base_load` channel) and probe both modes with EXISTING
checkpoints (zero-shot — policies never saw vertical load; results are a feasibility LOWER bound):

  STAND probe : stance specialist (go2_stabilize_adv) on go2_stabilize, W + a modest 10N lateral pull.
                standing-held = no termination for the full horizon. Also calf torque-saturation fraction
                (physical-infeasibility proxy) if the readout exists.
  REST probe  : soft-rest generalist (go2_weak_leg_blind_soft_adv) on the soft task (80N termination), W on.
                gentleness = per-env PEAK non-foot contact force before first done, vs the static rest force
                (~ mg + W): overhead = peak − static. Gentle descent = low overhead + no tip (fell_over).

GATE: a W range where STAND fails but REST stays gentle => the handoff window is REAL => proceed to G2 training.
Also outputs the static rest force vs W (to design the load-conditioned slam threshold SLAM_N(W)).
"""
import os, sys, io, contextlib
import torch as th
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "external/robot-safety-sandbox")
import numpy as np
from robot_safety_sandbox import make_tensor, spec
from robot_safety_sandbox.eval.policies import load_twin

DEV, N, DT, STEPS = "cuda:0", 128, 0.02, 300
T0 = 50                                      # load switches on at 1s (from W=0)
ROBOT_WEIGHT_N = 150.0                       # Go2 ~15kg, for static-rest reference
WS = [0, 30, 60, 90, 120, 150, 200, 250]
STAND_CK = "results/go2_weak_leg_runs/go2_stabilize_adv/checkpoints/model_49999872_steps.zip"
REST_CK = "results/go2_weak_leg_runs/go2_weak_leg_blind_soft_adv/checkpoints/model_49999872_steps.zip"
PULL = th.tensor([0., 1., 0.])               # modest lateral disturbance for the stand probe


def stand_probe(W):
    with contextlib.redirect_stdout(io.StringIO()):
        env = make_tensor("go2_stabilize", N, DEV, adversary=True)
        model, norm = load_twin(STAND_CK, DEV, quiet=True)
    inner = env.mj
    env.force_scale = 0.2 * th.ones(N, device=DEV)                   # 10N lateral pull
    dstb = (PULL).to(DEV)[None].expand(N, spec("go2_stabilize").dstb_dim).contiguous()
    obs = env.reset(); alive = th.ones(N, dtype=th.bool, device=DEV)
    sat = []
    for t in range(STEPS):
        env.base_load = (th.tensor([0., 0., -float(W)], device=DEV)[None].expand(N, 3)
                         if t >= T0 else None)
        with th.no_grad():
            a = th.clamp(model.policy._predict(norm(obs), deterministic=True), -1, 1)
        obs, g, dones, touts, l = env.step_tensor(th.cat([a, dstb], dim=1))
        alive &= ~(dones & ~touts)
        try:                                                          # calf torque saturation (feasibility proxy)
            f = inner.sim.data.actuator_force
            calf = f[:, [8, 9, 10, 11]].abs()
            sat.append(float((calf > 0.95 * 45.0).float().mean()))
        except Exception:
            pass
    env.close()
    return float(alive.float().mean()), (float(np.mean(sat[T0:])) if sat else float("nan"))


def rest_probe(W):
    with contextlib.redirect_stdout(io.StringIO()):
        env = make_tensor("go2_weak_leg_blind_soft", N, DEV, adversary=True)
        model, norm = load_twin(REST_CK, DEV, quiet=True)
    inner = env.mj
    env.force_scale = 0.0 * th.ones(N, device=DEV)                   # no lateral pull for the descent probe
    dstb = th.zeros(N, spec("go2_weak_leg_blind_soft").dstb_dim, device=DEV)
    obs = env.reset()
    done_seen = th.zeros(N, dtype=th.bool, device=DEV)
    tipped = th.zeros(N, dtype=th.bool, device=DEV)
    peak = th.zeros(N, device=DEV); static_acc = []; height_end = None
    for t in range(STEPS):
        env.base_load = (th.tensor([0., 0., -float(W)], device=DEV)[None].expand(N, 3)
                         if t >= T0 else None)
        with th.no_grad():
            a = th.clamp(model.policy._predict(norm(obs), deterministic=True), -1, 1)
        obs, g, dones, touts, l = env.step_tensor(th.cat([a, dstb], dim=1))
        s = inner.scene["nonfoot_ground_touch"]
        fh = s.data.force_history if s.data.force_history is not None else s.data.force
        force = th.norm(fh, dim=-1).flatten(1).amax(1)
        peak = th.where(~done_seen, th.maximum(peak, force), peak)   # peak before first done only
        tm = inner.termination_manager._term_dones
        z = th.zeros(N, device=DEV)
        tipped |= tm.get("fell_over", z).bool() & ~done_seen
        done_seen |= (dones & ~touts)
        if t >= STEPS - 50:
            static_acc.append(float(force.mean()))                    # late-window ~ static rest force
    d = inner.scene["robot"].data
    height_end = float(d.root_link_pos_w[:, 2].mean())
    env.close()
    static = float(np.mean(static_acc))
    return {"peak": float(peak.mean()), "static": static, "overhead": float(peak.mean()) - static,
            "tipped": float(tipped.float().mean()), "h_end": height_end}


if __name__ == "__main__":
    print(f"{'W(N)':>5} | {'STAND held':>10} {'calf sat':>8} | {'REST peak':>9} {'static':>7} {'overhd':>7} {'tipped':>6} {'h_end':>6}")
    for W in WS:
        sh, sat = stand_probe(W)
        r = rest_probe(W)
        print(f"{W:>5} | {sh:>10.2f} {sat:>8.2f} | {r['peak']:>9.0f} {r['static']:>7.0f} {r['overhead']:>7.0f} "
              f"{r['tipped']:>6.2f} {r['h_end']:>6.2f}")
    print("\nGATE: need a W band where STAND-held collapses but REST stays untipped with low overhead.")
    print(f"(static rest reference ~ {ROBOT_WEIGHT_N:.0f}+W; overhead = impact above static = the real slam signal)")
