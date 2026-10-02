"""E081 — SURVIVAL-CURVE evaluation of the wave scenarios (Buzi's critique of E080): start N robots per method,
NO respawn — first failure (tip OR slam) = death, censored thereafter (sim auto-resets but dead robots are
masked out of all statistics). Report S(t) per method and final survival. Two disturbance conditions:
  benign : 10N steady pull only (the E080 setting — the certificate's worst-case never materializes)
  gusty  : 10N + 35N gust pulses (0.5s every 2s) — the threat class the certificate is calibrated against
Methods: STAND-ONLY, REST-ONLY, ONE-WAY handoff, BIDIR handoff. Square & sine W(t) waves as E080.
"""
import os, sys, io, contextlib, json, math
from _paths import _ART
import torch as th
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "external/robot-safety-sandbox"); sys.path.insert(0, "experiments")
import numpy as np
from robot_safety_sandbox import make_tensor, spec
from robot_safety_sandbox.eval.policies import load_twin
from _value_util import stand_value

DEV, N, DT, STEPS = "cuda:0", 256, 0.02, 1000
EPS_DN, K_DN, EPS_UP, K_UP, REFRACT, ALPHA = -0.05, 5, 0.15, 25, 50, 0.1
CK = "results/go2_weight_runs/{r}/checkpoints/model_49999872_steps.zip"


def W_of(sched, t):
    s = t * DT
    if sched == "square":
        return 40.0 if (int(s // 3.0) % 2 == 0) else 220.0
    return 130.0 + 110.0 * math.sin(2 * math.pi * s / 8.0)


def gust_scale(cond, t):
    if cond == "benign":
        return 0.2
    s = t * DT
    return 0.7 if (s % 2.0) < 0.5 else 0.2          # 35N pulses 0.5s every 2s


def rollout(sched, cond, arm):
    with contextlib.redirect_stdout(io.StringIO()):
        env = make_tensor("go2_weight_rest_hi_at_0", N, DEV, adversary=True)
        m_s, n_s = load_twin(CK.format(r="E075_recal/go2_weight_stand_hi_adv"), DEV, quiet=True)
        m_r, n_r = load_twin(CK.format(r="go2_weight_rest_hi_adv"), DEV, quiet=True)
    inner = env.mj
    dstb = th.zeros(N, spec("go2_weight_rest_hi_at_0").dstb_dim, device=DEV); dstb[:, 1] = 1.0
    obs = env.reset()
    in_rest = th.zeros(N, dtype=th.bool, device=DEV)
    if arm == "REST-ONLY": in_rest[:] = True
    vbar = th.zeros(N, device=DEV); below = th.zeros(N, device=DEV); above = th.zeros(N, device=DEV)
    refr = th.zeros(N, device=DEV)
    alive = th.ones(N, dtype=th.bool, device=DEV)
    S = []
    for t in range(STEPS):
        W = W_of(sched, t)
        env.base_load = th.tensor([0., 0., -W], device=DEV)[None].expand(N, 3).contiguous()
        inner._weight_W[:] = W; inner._weight_h[:] = 0.25
        env.force_scale = gust_scale(cond, t) * th.ones(N, device=DEV)
        V = stand_value(env, m_s, n_s)
        vbar = (1 - ALPHA) * vbar + ALPHA * V
        if arm in ("BIDIR", "ONE-WAY"):
            below = th.where(vbar < EPS_DN, below + 1, th.zeros_like(below))
            above = th.where(vbar > EPS_UP, above + 1, th.zeros_like(above))
            can = refr <= 0
            go_dn = (~in_rest) & (below >= K_DN) & can
            go_up = in_rest & (above >= K_UP) & can & (th.tensor(arm == "BIDIR", device=DEV))
            in_rest = th.where(go_dn, th.ones_like(in_rest), in_rest)
            in_rest = th.where(go_up, th.zeros_like(in_rest), in_rest)
            refr = th.where(go_dn | go_up, th.full_like(refr, REFRACT), refr - 1)
        with th.no_grad():
            a_s = th.clamp(m_s.policy._predict(n_s(obs), deterministic=True), -1, 1)
            a_r = th.clamp(m_r.policy._predict(n_r(obs), deterministic=True), -1, 1)
        a = th.where(in_rest.unsqueeze(-1), a_r, a_s)
        obs, g, dones, touts, l = env.step_tensor(th.cat([a, dstb], dim=1))
        z = th.zeros(N, device=DEV)
        tip = inner.termination_manager._term_dones.get("fell_over", z).bool()
        cap = 80.0 + 1.3 * W
        s = inner.scene["nonfoot_ground_touch"]
        fh = s.data.force_history if s.data.force_history is not None else s.data.force
        slam = th.norm(fh, dim=-1).flatten(1).amax(1) > cap
        alive &= ~(tip | slam)                       # first failure = death; no resurrection
        S.append(float(alive.float().mean()))
    env.close()
    return S


if __name__ == "__main__":
    out = {}
    for cond in ("benign", "gusty"):
        for sched in ("square", "sine"):
            print(f"\n===== {sched} / {cond} =====  final survival (of {N}, no respawn):")
            for arm in ("STAND-ONLY", "REST-ONLY", "ONE-WAY", "BIDIR"):
                S = rollout(sched, cond, arm)
                out[f"{cond}|{sched}|{arm}"] = S
                print(f"  {arm:>10}: S(5s)={S[249]:.2f}  S(10s)={S[499]:.2f}  S(20s)={S[-1]:.2f}")
    od = os.path.expanduser(_ART + "/E080-bidirectional")
    json.dump(out, open(f"{od}/survival.json", "w"))
    print(f"\nsaved -> {od}/survival.json")
