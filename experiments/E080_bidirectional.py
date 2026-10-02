"""E080 — BIDIRECTIONAL certified handoff on the weight ladder: W(t) oscillates (square / sine) across the
feasibility boundary (W_c≈130 @10N); the filter should sit when standing is uncertifiable and STAND BACK UP when
it becomes certifiable again. Guard: down = EMA(V_stand)<EPS_DN for K_DN steps (calibrated on standing states);
up = EMA(V_stand)>EPS_UP for K_UP steps (calibrated on prone states, probe E080-A); REFRACT steps after each
switch (anti-chatter). Arms: BIDIR (ours) / ONE-WAY (first descent permanent) / STAND-ONLY / REST-ONLY.
"""
import os, sys, io, contextlib, json, math
import torch as th
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "external/robot-safety-sandbox"); sys.path.insert(0, "experiments")
import numpy as np
from robot_safety_sandbox import make_tensor, spec
from robot_safety_sandbox.eval.policies import load_twin
from _value_util import stand_value

DEV, N, DT, STEPS = "cuda:0", 128, 0.02, 1000
EPS_DN, K_DN = -0.05, 5
EPS_UP, K_UP = 0.15, 25
REFRACT = 50
ALPHA = 0.1
W_C = 130.0                                     # measured feasibility boundary @10N (E079)
CK = "results/go2_weight_runs/{r}/checkpoints/model_49999872_steps.zip"


def W_of(sched, t):
    s = t * DT
    if sched == "square":
        return 40.0 if (int(s // 3.0) % 2 == 0) else 220.0     # 3s@40 / 3s@220
    return 130.0 + 110.0 * math.sin(2 * math.pi * s / 8.0)     # sine, period 8s, range [20,240]


def rollout(sched, arm):
    with contextlib.redirect_stdout(io.StringIO()):
        env = make_tensor("go2_weight_rest_hi_at_0", N, DEV, adversary=True)
        m_s, n_s = load_twin(CK.format(r="E075_recal/go2_weight_stand_hi_adv"), DEV, quiet=True)
        m_r, n_r = load_twin(CK.format(r="go2_weight_rest_hi_adv"), DEV, quiet=True)
    inner = env.mj
    env.force_scale = 0.2 * th.ones(N, device=DEV)
    dstb = th.zeros(N, spec("go2_weight_rest_hi_at_0").dstb_dim, device=DEV); dstb[:, 1] = 1.0
    obs = env.reset()
    in_rest = th.zeros(N, dtype=th.bool, device=DEV)
    if arm == "REST-ONLY": in_rest[:] = True
    vbar = th.zeros(N, device=DEV); below = th.zeros(N, device=DEV); above = th.zeros(N, device=DEV)
    refr = th.zeros(N, device=DEV)
    switches = th.zeros(N, device=DEV)
    tipped = th.zeros(N, dtype=th.bool, device=DEV); slammed = th.zeros(N, dtype=th.bool, device=DEV)
    stand_feas = th.zeros(N, device=DEV); feas_steps = 0
    rest_infeas = th.zeros(N, device=DEV); infeas_steps = 0
    stand_feas_late = th.zeros(N, device=DEV); feas_late = 0    # after first feasible->... recovery window
    traces = {"W": [], "V": [], "h": [], "rest_frac": []}
    for t in range(STEPS):
        W = W_of(sched, t)
        env.base_load = th.tensor([0., 0., -W], device=DEV)[None].expand(N, 3).contiguous()
        inner._weight_W[:] = W; inner._weight_h[:] = 0.25
        V = stand_value(env, m_s, n_s)
        vbar = (1 - ALPHA) * vbar + ALPHA * V
        if arm in ("BIDIR", "ONE-WAY"):
            below = th.where(vbar < EPS_DN, below + 1, th.zeros_like(below))
            above = th.where(vbar > EPS_UP, above + 1, th.zeros_like(above))
            can = refr <= 0
            go_dn = (~in_rest) & (below >= K_DN) & can
            go_up = in_rest & (above >= K_UP) & can
            if arm == "ONE-WAY":
                go_up[:] = False
            in_rest = th.where(go_dn, th.ones_like(in_rest), in_rest)
            in_rest = th.where(go_up, th.zeros_like(in_rest), in_rest)
            sw = go_dn | go_up
            switches += sw.float()
            refr = th.where(sw, th.full_like(refr, REFRACT), refr - 1)
        with th.no_grad():
            a_s = th.clamp(m_s.policy._predict(n_s(obs), deterministic=True), -1, 1)
            a_r = th.clamp(m_r.policy._predict(n_r(obs), deterministic=True), -1, 1)
        a = th.where(in_rest.unsqueeze(-1), a_r, a_s)
        obs, g, dones, touts, l = env.step_tensor(th.cat([a, dstb], dim=1))
        d = inner.scene["robot"].data
        pg = d.projected_gravity_b; tilt = th.maximum(pg[:, 0].abs(), pg[:, 1].abs())
        standing = (d.root_link_pos_w[:, 2] > 0.20) & (tilt < 0.3)
        z = th.zeros(N, device=DEV)
        tipped |= inner.termination_manager._term_dones.get("fell_over", z).bool()
        cap = 80.0 + 1.3 * W
        s = inner.scene["nonfoot_ground_touch"]
        fh = s.data.force_history if s.data.force_history is not None else s.data.force
        slammed |= th.norm(fh, dim=-1).flatten(1).amax(1) > cap
        if W < W_C:
            stand_feas += standing.float(); feas_steps += 1
            if t * DT > 4.0:                                       # feasible windows AFTER the first heavy phase
                stand_feas_late += standing.float(); feas_late += 1
        else:
            rest_infeas += (~standing).float(); infeas_steps += 1
        traces["W"].append(W); traces["V"].append(float(vbar.mean()))
        traces["h"].append(float(d.root_link_pos_w[:, 2].mean()))
        traces["rest_frac"].append(float(in_rest.float().mean()))
    env.close()
    return {
        "mode_track_feas": float((stand_feas / max(1, feas_steps)).mean()),
        "mode_track_infeas": float((rest_infeas / max(1, infeas_steps)).mean()),
        "recovered_avail": float((stand_feas_late / max(1, feas_late)).mean()),
        "tip": float(tipped.float().mean()), "slam": float(slammed.float().mean()),
        "switches": float(switches.mean()),
    }, traces


if __name__ == "__main__":
    out = {}
    for sched in ("square", "sine"):
        print(f"\n===== {sched} wave =====")
        print(f"{'arm':>10} {'std|feas':>8} {'rest|infeas':>11} {'recovered':>9} {'tip':>5} {'slam':>5} {'switches':>8}")
        for arm in ("BIDIR", "ONE-WAY", "STAND-ONLY", "REST-ONLY"):
            r, tr = rollout(sched, arm)
            out[f"{sched}|{arm}"] = {"metrics": r, "traces": tr if arm == "BIDIR" else None}
            print(f"{arm:>10} {r['mode_track_feas']:>8.2f} {r['mode_track_infeas']:>11.2f} "
                  f"{r['recovered_avail']:>9.2f} {r['tip']:>5.2f} {r['slam']:>5.2f} {r['switches']:>8.1f}")
    od = os.path.expanduser("~/artifacts/odd-conditioned/E080-bidirectional")
    os.makedirs(od, exist_ok=True)
    json.dump(out, open(f"{od}/results.json", "w"))
    print(f"\nsaved -> {od}/results.json")
