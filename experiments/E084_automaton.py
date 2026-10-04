"""E084 — BATCH-2 evaluation: the certified-transition automaton vs all batch-1 arms, under the E081 survival
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
from _paths import _ART
import torch as th
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "external/robot-safety-sandbox"); sys.path.insert(0, "experiments")
import numpy as np
from robot_safety_sandbox import make_tensor, spec
from robot_safety_sandbox.eval.policies import load_twin

DEV, N, DT, STEPS = "cuda:0", 256, 0.02, 1000
EPS_DN, K_DN = -0.05, 5
EPS_UP, K_UP = 0.10, 15   # 0.02 let W=220 prone states fire get-up (E086 diag: 96 mid-pulse attempts, 98 aborts)
EPS_ABORT, K_ABORT = -0.02, 10
REFRACT, ALPHA = 50, 0.1
V1_UP, V1_K_UP = 0.15, 25       # V1 ("direct") return: EMA V_stand read while lying down > V1_UP for V1_K_UP steps
WARMUP = 50                     # no triggers during the spawn transient (E086 diag: EMA-from-0 dips fired descents at W=40)
UP_W_GATE = None                # if set: get-up allowed only while the ODD estimate W < gate (belief-gated return; E086 sets 130)
CK = {
    "stand": "results/go2_weight_runs/E075_recal/go2_weight_stand_hi_adv/checkpoints/model_49999872_steps.zip",
    "rest": "results/go2_weight_runs/go2_weight_rest_hi_adv/checkpoints/model_49999872_steps.zip",
    "getup": "results/go2_transition_runs/getup_v2/go2_getup_adv/checkpoints/model_49999872_steps.zip",
    "descend": "results/go2_transition_runs/descend_v4/go2_descend_adv/checkpoints/model_49999872_steps.zip",
}
STATES = ["STAND", "DESCENDING", "REST", "GETTINGUP"]


def load_twins(ck=None, dev=DEV):
    """Load every twin in ``ck`` (default CK). A missing checkpoint is an ERROR: silently skipping one
    (the old ``if os.path.exists`` pattern) would e.g. turn arm V2 into V2-REUSE without a word."""
    ck = CK if ck is None else ck
    missing = [v for v in ck.values() if not os.path.exists(v)]
    if missing:
        raise SystemExit("missing checkpoints — run `bash scripts/fetch_bundles.sh` from the repo root:\n  "
                         + "\n  ".join(missing))
    return {k: load_twin(v, dev, quiet=True) for k, v in ck.items()}


def W_of(sched, t):
    s = t * DT
    if sched == "square":
        return 40.0 if (int(s // 3.0) % 2 == 0) else 220.0
    return 130.0 + 110.0 * math.sin(2 * math.pi * s / 8.0)


def gust_scale(cond, t):
    """Scripted eval push, as a fraction of force_max (50 N) in +y: benign = constant 10 N; gusty = 35 N for
    0.5 s every 2 s; none = no push (the learned adversary is never used at evaluation)."""
    if cond == "none":
        return 0.0
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


def cal_summary(name, vals, thr, side):
    """Print the distribution of a calibration signal and where the current threshold cuts it.
    side='above': fraction >= thr fires (a return gate); side='below': fraction < thr fires (a descent trigger)."""
    v = th.cat([x.flatten() for x in vals]).float().cpu() if vals else th.zeros(0)
    if v.numel() == 0:
        print(f"{name}: no samples"); return
    q = th.quantile(v, th.tensor([0.05, 0.25, 0.5, 0.75, 0.95]))
    frac = float((v >= thr).float().mean()) if side == "above" else float((v < thr).float().mean())
    print(f"{name}: n={v.numel()}  p5 {q[0]:+.3f}  p25 {q[1]:+.3f}  p50 {q[2]:+.3f}  p75 {q[3]:+.3f}  "
          f"p95 {q[4]:+.3f}  | current threshold {thr:+.3f} fires on {frac:.0%}")


def value_of(env, model, norm):
    obs = env.mj._zoo_last_obs
    with th.no_grad():
        return model.policy.predict_values(norm(obs)).squeeze(-1)


def rollout(sched, cond, arm, cal=None):
    """cal=None: the evaluation. cal="up" (arm V2): return disabled; collects EMA V_up on settled-rest robots
    while the ODD has cleared (the population the return gate sees). cal="dn": collects EMA V_stand on
    robots in STAND (the population the descent trigger sees)."""
    with contextlib.redirect_stdout(io.StringIO()):
        env = make_tensor("go2_weight_rest_hi_at_0", N, DEV, adversary=True)
        tw = load_twins()
    inner = env.mj
    inner.cfg.episode_length_s = 10_000.0   # kill the 20s timeout: no free mid-eval resets (Buzi's catch)
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
    cal_vals, cal_v1 = [], []
    for t in range(STEPS):
        W = W_of(sched, t)
        env.base_load = th.tensor([0., 0., -W], device=DEV)[None].expand(N, 3).contiguous()
        inner._weight_W[:] = W; inner._weight_h[:] = 0.25
        env.force_scale = gust_scale(cond, t) * th.ones(N, device=DEV)
        m_s, n_s = tw["stand"]; m_r, n_r = tw["rest"]
        Vs = value_of(env, m_s, n_s)
        vs_bar = Vs.clone() if t == 0 else (1 - ALPHA) * vs_bar + ALPHA * Vs
        if arm in ("V2", "V2-REUSE"):
            m_u, n_u = tw["getup"]
            Vu = value_of(env, m_u, n_u)
            vu_bar = Vu.clone() if t == 0 else (1 - ALPHA) * vu_bar + ALPHA * Vu
            below = th.where(vs_bar < EPS_DN, below + 1, th.zeros_like(below))
            above = th.where(vu_bar > EPS_UP, above + 1, th.zeros_like(above))
            abort_c = th.where(vu_bar < EPS_ABORT, abort_c + 1, th.zeros_like(abort_c))
            settle = th.where(in_rest_target(inner), settle + 1, th.zeros_like(settle))
            standok = th.where(in_stance_target(inner), standok + 1, th.zeros_like(standok))
            can = (refr <= 0) & (t >= WARMUP)
            go_desc = (st == 0) & (below >= K_DN) & can
            go_rest = (st == 1) & (settle >= 5)
            go_up = (st == 2) & (above >= K_UP) & can
            if UP_W_GATE is not None and W >= UP_W_GATE:
                go_up = th.zeros_like(go_up)
            if cal == "up":
                go_up = th.zeros_like(go_up)
                if W < (UP_W_GATE if UP_W_GATE is not None else 130.0):
                    sel = alive & (st == 2) & (settle >= 5)
                    cal_vals.append(vu_bar[sel].clone())
                    cal_v1.append(vs_bar[sel].clone())
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
            above = th.where(vs_bar > V1_UP, above + 1, th.zeros_like(above))   # batch-1 prone-V guard
            can = (refr <= 0) & (t >= WARMUP)
            go_dn = (~v1_in_rest) & (below >= K_DN) & can
            go_up = v1_in_rest & (above >= V1_K_UP) & can & th.tensor(arm == "V1", device=DEV)
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
        # deaths = the ENV'S OWN failure terminations (fell_over | illegal_contact). Recomputing slam
        # externally missed resets whose sensor buffers were cleared before we read them (E089 post-mortem:
        # 93/94 illegal_contact resets leaked back in as "alive").
        z = th.zeros(N, device=DEV)
        tip = inner.termination_manager._term_dones.get("fell_over", z).bool()
        slam = inner.termination_manager._term_dones.get("illegal_contact", z).bool()
        newdead = (tip | slam) & alive
        for si, sname in enumerate(STATES):
            deaths[sname] += int((newdead & (st == si)).sum())
        alive &= ~(tip | slam)
        if cal == "dn" and t >= WARMUP:
            cal_vals.append(vs_bar[alive & (st == 0)].clone())
        d = inner.scene["robot"].data
        pg = d.projected_gravity_b
        standing = (d.root_link_pos_w[:, 2] > 0.18) & (th.maximum(pg[:, 0].abs(), pg[:, 1].abs()) < 0.3)
        stand_time += (standing & alive).float(); alive_time += alive.float()
        S.append(float(alive.float().mean()))
    env.close()
    afford = float((stand_time / alive_time.clamp_min(1)).mean())
    return {"S": S, "final": S[-1], "afford_alive": afford, "deaths": deaths, "cal": cal_vals, "cal_v1": cal_v1}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cal-dn", action="store_true",
                    help="V_stand on standing robots at constant light / boundary / heavy load (EPS_DN)")
    ap.add_argument("--cal-up", action="store_true",
                    help="V_up on settled-rest robots once the load has cleared, return disabled (EPS_UP)")
    args = ap.parse_args()
    if args.cal_dn:
        import sys as _sys
        me = _sys.modules[__name__]
        saved = me.W_of
        for Wfix in (40.0, 130.0, 220.0):
            me.W_of = lambda sched, t, _w=Wfix: _w
            r = rollout("square", "benign", "STAND-ONLY", cal="dn")
            cal_summary(f"V_stand standing @W={Wfix:.0f}", r["cal"], EPS_DN, "below")
        me.W_of = saved
        raise SystemExit(0)
    if args.cal_up:
        r = rollout("square", "benign", "V2", cal="up")
        cal_summary("V_up on settled rest, load cleared (W=40)", r["cal"], EPS_UP, "above")
        raise SystemExit(0)
    arms = ["V2", "V2-REUSE", "V1", "ONE-WAY", "STAND-ONLY", "REST-ONLY"]
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
    od = os.path.expanduser(_ART + "/E084-automaton")
    os.makedirs(od, exist_ok=True)
    json.dump(out, open(f"{od}/results.json", "w"))
    print(f"\nsaved -> {od}/results.json")
