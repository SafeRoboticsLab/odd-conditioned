"""E084b (V2-EARLY variant) — BATCH-2 evaluation: the certified-transition automaton vs all batch-1 arms, under the E081 survival
protocol (no respawn; first tip∨slam = death). The automaton (per env):
    STAND --(EMA V_stand<EPS_DN, K_DN, refr)--> DESCENDING --(x∈T_rest sustained)--> REST
    REST --(EMA V_up>EPS_UP, K_UP, refr)--> GETTINGUP --(x∈T_stance sustained)--> STAND
    GETTINGUP --(EMA V_up<EPS_ABORT sustained)--> DESCENDING            [certified abort]
Policies: STAND=stand_hi(recal), DESCENDING=descend funnel (or rest_hi in the 'reuse' variant), REST=rest_hi,
GETTINGUP=getup funnel. Values: V_stand (stand twin), V_up (getup twin — the calibrated return certificate).
Arms: V2 (dedicated descend), V2-REUSE (rest_hi as descend), V1 (batch-1 bidir), ONE-WAY, STAND-ONLY, REST-ONLY.
Records: survival S(t), affordance while alive, and DEATH FORENSICS (automaton state at death).
"""
import os, sys, io, contextlib, json, math
import torch as th
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "external/robot-safety-sandbox"); sys.path.insert(0, "experiments")
import numpy as np
from robot_safety_sandbox import make_tensor, spec
from robot_safety_sandbox.eval.policies import load_twin

DEV, N, DT, STEPS = "cuda:0", 256, 0.02, 1000
EPS_DN, K_DN = 0.02, 5          # earlier: descend while V_stand merely DEGRADING
EPS_FORCE, K_FORCE = 0.05, 5    # forced descent when the descend certificate's entry window narrows
EPS_UP, K_UP = 0.02, 15
EPS_ABORT, K_ABORT = -0.02, 10
REFRACT, ALPHA = 50, 0.1
CK = {
    "stand": "results/go2_weight_runs/E075_recal/go2_weight_stand_hi_adv/checkpoints/model_49999872_steps.zip",
    "rest": "results/go2_weight_runs/go2_weight_rest_hi_adv/checkpoints/model_49999872_steps.zip",
    "getup": "results/go2_transition_runs/go2_getup_adv/checkpoints/model_49999872_steps.zip",
    "descend": "results/go2_transition_runs/descend_v4/go2_descend_adv/checkpoints/model_49999872_steps.zip",
}
STATES = ["STAND", "DESCENDING", "REST", "GETTINGUP"]


def W_of(sched, t):
    s = t * DT
    if sched == "square":
        return 40.0 if (int(s // 3.0) % 2 == 0) else 220.0
    return 130.0 + 110.0 * math.sin(2 * math.pi * s / 8.0)


def gust_scale(cond, t):
    if cond == "benign":
        return 0.2
    return 0.7 if ((t * DT) % 2.0) < 0.5 else 0.2


def in_rest_target(inner):
    d = inner.scene["robot"].data
    pg = d.projected_gravity_b
    tilt = th.maximum(pg[:, 0].abs(), pg[:, 1].abs())
    v = th.linalg.norm(d.root_link_lin_vel_b, dim=1)
    w = th.linalg.norm(d.root_link_ang_vel_b, dim=1)
    return (d.root_link_pos_w[:, 2] < 0.15) & (tilt < 0.25) & (v < 0.30) & (w < 0.50)


def in_stance_target(inner):
    d = inner.scene["robot"].data
    pg = d.projected_gravity_b
    tilt = th.maximum(pg[:, 0].abs(), pg[:, 1].abs())
    v = th.linalg.norm(d.root_link_lin_vel_b, dim=1)
    return (d.root_link_pos_w[:, 2] > 0.20) & (tilt < 0.25) & (v < 0.30)


def value_of(env, model, norm):
    obs = env.mj._zoo_last_obs
    with th.no_grad():
        return model.policy.predict_values(norm(obs)).squeeze(-1)


def rollout(sched, cond, arm):
    with contextlib.redirect_stdout(io.StringIO()):
        env = make_tensor("go2_weight_rest_hi_at_0", N, DEV, adversary=True)
        tw = {k: load_twin(v, DEV, quiet=True) for k, v in CK.items() if os.path.exists(v)}
    inner = env.mj
    dstb = th.zeros(N, spec("go2_weight_rest_hi_at_0").dstb_dim, device=DEV); dstb[:, 1] = 1.0
    obs = env.reset()
    # automaton state per env: 0 STAND, 1 DESCENDING, 2 REST, 3 GETTINGUP
    st = th.zeros(N, dtype=th.long, device=DEV)
    if arm == "REST-ONLY": st[:] = 2
    vs_bar = th.zeros(N, device=DEV); vu_bar = th.zeros(N, device=DEV)
    below = th.zeros(N, device=DEV); above = th.zeros(N, device=DEV); abort_c = th.zeros(N, device=DEV)
    settle = th.zeros(N, device=DEV); standok = th.zeros(N, device=DEV)
    refr = th.zeros(N, device=DEV)
    alive = th.ones(N, dtype=th.bool, device=DEV)
    stand_time = th.zeros(N, device=DEV); alive_time = th.zeros(N, device=DEV)
    deaths = {s: 0 for s in STATES}
    S = []
    v1_in_rest = th.zeros(N, dtype=th.bool, device=DEV)  # for V1/ONE-WAY arms
    for t in range(STEPS):
        W = W_of(sched, t)
        env.base_load = th.tensor([0., 0., -W], device=DEV)[None].expand(N, 3).contiguous()
        inner._weight_W[:] = W; inner._weight_h[:] = 0.25
        env.force_scale = gust_scale(cond, t) * th.ones(N, device=DEV)
        m_s, n_s = tw["stand"]; m_r, n_r = tw["rest"]
        Vs = value_of(env, m_s, n_s)
        vs_bar = (1 - ALPHA) * vs_bar + ALPHA * Vs
        if arm in ("V2", "V2-REUSE"):
            m_u, n_u = tw["getup"]
            Vu = value_of(env, m_u, n_u)
            vu_bar = (1 - ALPHA) * vu_bar + ALPHA * Vu
            m_d2, n_d2 = tw["descend"]
            Vd = value_of(env, m_d2, n_d2)
            if not hasattr(rollout, "_vd"): pass
            vd_bar = globals().setdefault("_VDBAR", {}).setdefault(id(env), th.zeros(N, device=DEV))
            vd_bar.mul_(1 - ALPHA).add_(ALPHA * Vd)
            forcec = globals().setdefault("_FORCEC", {}).setdefault(id(env), th.zeros(N, device=DEV))
            forcec.copy_(th.where(vd_bar < EPS_FORCE, forcec + 1, th.zeros_like(forcec)))
            below = th.where(vs_bar < EPS_DN, below + 1, th.zeros_like(below))
            above = th.where(vu_bar > EPS_UP, above + 1, th.zeros_like(above))
            abort_c = th.where(vu_bar < EPS_ABORT, abort_c + 1, th.zeros_like(abort_c))
            settle = th.where(in_rest_target(inner), settle + 1, th.zeros_like(settle))
            standok = th.where(in_stance_target(inner), standok + 1, th.zeros_like(standok))
            can = refr <= 0
            go_desc = (st == 0) & ((below >= K_DN) | (forcec >= K_FORCE)) & can
            go_rest = (st == 1) & (settle >= 5)
            go_up = (st == 2) & (above >= K_UP) & can
            go_stand = (st == 3) & (standok >= 5)
            go_abort = (st == 3) & (abort_c >= K_ABORT)
            st = th.where(go_desc, th.ones_like(st), st)
            st = th.where(go_rest, 2 * th.ones_like(st), st)
            st = th.where(go_up, 3 * th.ones_like(st), st)
            st = th.where(go_stand, th.zeros_like(st), st)
            st = th.where(go_abort, th.ones_like(st), st)
            sw = go_desc | go_up | go_abort
            refr = th.where(sw, th.full_like(refr, REFRACT), refr - 1)
        elif arm in ("V1", "ONE-WAY"):
            below = th.where(vs_bar < EPS_DN, below + 1, th.zeros_like(below))
            above = th.where(vs_bar > 0.15, above + 1, th.zeros_like(above))   # batch-1 prone-V guard
            can = refr <= 0
            go_dn = (~v1_in_rest) & (below >= K_DN) & can
            go_up = v1_in_rest & (above >= 25) & can & th.tensor(arm == "V1", device=DEV)
            v1_in_rest = th.where(go_dn, th.ones_like(v1_in_rest), v1_in_rest)
            v1_in_rest = th.where(go_up, th.zeros_like(v1_in_rest), v1_in_rest)
            refr = th.where(go_dn | go_up, th.full_like(refr, REFRACT), refr - 1)
            st = th.where(v1_in_rest, 2 * th.ones_like(st), th.zeros_like(st))
        # pick action by state
        with th.no_grad():
            a_s = th.clamp(m_s.policy._predict(n_s(obs), deterministic=True), -1, 1)
            a_r = th.clamp(m_r.policy._predict(n_r(obs), deterministic=True), -1, 1)
            a = th.where((st >= 1).unsqueeze(-1) & (st <= 2).unsqueeze(-1), a_r, a_s)
            if arm == "V2" and "descend" in tw:
                m_d, n_d = tw["descend"]
                a_d = th.clamp(m_d.policy._predict(n_d(obs), deterministic=True), -1, 1)
                a = th.where((st == 1).unsqueeze(-1), a_d, a)
            if arm in ("V2", "V2-REUSE"):
                m_u, n_u = tw["getup"]
                a_u = th.clamp(m_u.policy._predict(n_u(obs), deterministic=True), -1, 1)
                a = th.where((st == 3).unsqueeze(-1), a_u, a)
        obs, g, dones, touts, l = env.step_tensor(th.cat([a, dstb], dim=1))
        z = th.zeros(N, device=DEV)
        tip = inner.termination_manager._term_dones.get("fell_over", z).bool()
        cap = 80.0 + 1.3 * W
        sns = inner.scene["nonfoot_ground_touch"]
        fh = sns.data.force_history if sns.data.force_history is not None else sns.data.force
        slam = th.norm(fh, dim=-1).flatten(1).amax(1) > cap
        newdead = (tip | slam) & alive
        for si, sname in enumerate(STATES):
            deaths[sname] += int((newdead & (st == si)).sum())
        alive &= ~(tip | slam)
        d = inner.scene["robot"].data
        pg = d.projected_gravity_b
        standing = (d.root_link_pos_w[:, 2] > 0.18) & (th.maximum(pg[:, 0].abs(), pg[:, 1].abs()) < 0.3)
        stand_time += (standing & alive).float(); alive_time += alive.float()
        S.append(float(alive.float().mean()))
    env.close()
    afford = float((stand_time / alive_time.clamp_min(1)).mean())
    return {"S": S, "final": S[-1], "afford_alive": afford, "deaths": deaths}


if __name__ == "__main__":
    arms = ["V2"]
    out = {}
    for cond in ("benign", "gusty"):
        for sched in ("square", "sine"):
            print(f"\n===== {sched} / {cond} =====")
            print(f"{'arm':>10} {'S(20s)':>7} {'afford|alive':>12}  deaths by state")
            for arm in arms:
                r = rollout(sched, cond, arm)
                out[f"{cond}|{sched}|{arm}"] = r
                dd = " ".join(f"{k}:{v}" for k, v in r["deaths"].items() if v)
                print(f"{arm:>10} {r['final']:>7.2f} {r['afford_alive']:>12.2f}  {dd}")
    od = os.path.expanduser("~/artifacts/odd-conditioned/E084-automaton")
    os.makedirs(od, exist_ok=True)
    json.dump(out, open(f"{od}/results_early.json", "w"))
    print(f"\nsaved -> {od}/results.json")
