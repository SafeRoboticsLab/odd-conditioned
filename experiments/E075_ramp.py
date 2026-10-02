"""E075 (T003) Part A.2 — re-run the E074 gust-ramp with the RECALIBRATED stand twin.

SAME config as E074_ramp (W: 0->250N over t=2..8s, ambient 10N + 25N gusts @t=1.5/5/6.5/8.5s, high-CoM
h=0.25). Only the STAND-side arms change: STAND-ONLY and HANDOFF now use the RECAL stand policy + RECAL
value trigger (trained W~U[0,120] = the comfortably-feasible band). REST-ONLY / UNIFIED / UNIFIED-DISC rows
are COPIED from E074's task3_ramp.json (unchanged — same rest_hi / unified_hi / unified_disc_hi twins).

Question: did the calibrated certificate close the trigger-precision gap? E074 handoff: switch-W 113±79,
tip 0.21, afford 0.46. Oracle safe-switch band: W∈[40,120], tips 0.03-0.05. Report the recal handoff's
switch-W mean±std, tip/slam/affordance vs those.
"""
import os, sys, io, contextlib, json
from _paths import _ART
import torch as th
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "external/robot-safety-sandbox")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from robot_safety_sandbox import make_tensor, spec
from robot_safety_sandbox.eval.policies import load_twin
from _value_util import stand_value

DEV, N, STEPS = "cuda:0", 128, 500
RAMP_START, RAMP_END, W_MAX = 100, 400, 250.0
LOAD_H = 0.25
AMBIENT, GUST_SCALE = 0.20, 0.70
GUST_STEPS = 25
GUST_STARTS = [75, 250, 325, 425]
# RECAL trigger: value net is cleaner than E074's (cert-band ~0, fail-band -0.43). EMA smoothing + settle
# lockout as before. eps=-0.05 lands the switch just past the feasible edge (W~140), inside the safe window.
EPS, HYST, ALPHA, SETTLE = -0.05, 8, 0.10, 130
PULL = th.tensor([0., 1., 0.])
CK = "results/go2_weight_runs/go2_weight_{m}_adv/checkpoints/model_49999872_steps.zip"
RECAL = "results/go2_weight_runs/E075_recal/go2_weight_stand_hi_adv/checkpoints/model_49999872_steps.zip"
TASK = "go2_weight_rest_hi_at_0"
OUT = os.path.expanduser(_ART + "/E075-recal-eval")
E074_JSON = os.path.expanduser(_ART + "/E074-hicom-demo/task3_ramp.json")

# arm -> (stand-phase policy ckpt, rest-phase policy ckpt, rule). RECAL replaces stand_hi on the stand side.
ARMS = {
    "STAND-ONLY(recal)": (RECAL,                       RECAL,                        "none"),
    "HANDOFF(recal)":    (RECAL,                        CK.format(m="rest_hi"),      "V"),
}


def W_of(t):
    if t < RAMP_START: return 0.0
    if t < RAMP_END: return (t - RAMP_START) / (RAMP_END - RAMP_START) * W_MAX
    return W_MAX


def in_gust(t):
    return any(g <= t < g + GUST_STEPS for g in GUST_STARTS)


def rollout(stand_ck, rest_ck, rule):
    with contextlib.redirect_stdout(io.StringIO()):
        env = make_tensor(TASK, N, DEV, adversary=True)
        m_s, n_s = load_twin(stand_ck, DEV, quiet=True)
        m_r, n_r = load_twin(rest_ck, DEV, quiet=True)
    m_s.policy.set_training_mode(False); m_r.policy.set_training_mode(False)
    m_vs, n_vs = m_s, n_s                                # RECAL value = its own stand twin
    inner = env.mj
    dstb = PULL.to(DEV)[None].expand(N, spec(TASK).dstb_dim).contiguous()
    obs = env.reset()

    switched = th.zeros(N, dtype=th.bool, device=DEV)
    below = th.zeros(N, device=DEV); ema = th.zeros(N, device=DEV); ema_init = False
    switch_step = th.full((N,), -1, dtype=th.long, device=DEV)
    switch_W = th.full((N,), -1.0, device=DEV)
    standing_t = th.zeros(N, device=DEV); standing_pre = th.zeros(N, device=DEV)
    tipped = th.zeros(N, dtype=th.bool, device=DEV); slam = th.zeros(N, dtype=th.bool, device=DEV)
    Vs_trace, h_trace, sw_trace = [], [], []

    for t in range(STEPS):
        W = W_of(t)
        inner._weight_h = th.full((N,), LOAD_H, device=DEV)
        inner._weight_W = th.full((N,), W, device=DEV)
        env.base_load = th.tensor([0., 0., -W], device=DEV)[None].expand(N, 3).contiguous()
        env.force_scale = (GUST_SCALE if in_gust(t) else AMBIENT) * th.ones(N, device=DEV)

        Vs = stand_value(env, m_vs, n_vs)
        Vs_trace.append(float(Vs.mean()))
        if not ema_init:
            ema = Vs.clone(); ema_init = True
        else:
            ema = ALPHA * Vs + (1 - ALPHA) * ema
        if rule == "V":
            active = t >= SETTLE
            below = th.where((ema < EPS) & active, below + 1.0, th.zeros_like(below))
            new = (below >= HYST) & ~switched
            if new.any():
                switch_step[new] = t; switch_W[new] = W
            switched = switched | new

        with th.no_grad():
            a_s = th.clamp(m_s.policy._predict(n_s(obs), deterministic=True), -1, 1)
            a_r = th.clamp(m_r.policy._predict(n_r(obs), deterministic=True), -1, 1)
        a = th.where(switched.unsqueeze(-1), a_r, a_s)
        obs, g, dones, touts, l = env.step_tensor(th.cat([a, dstb], dim=1))

        d = inner.scene["robot"].data
        pg = d.projected_gravity_b
        tilt = th.maximum(pg[:, 0].abs(), pg[:, 1].abs())
        is_stand = (d.root_link_pos_w[:, 2] > 0.18) & (tilt < 0.3)
        standing_t += is_stand.float(); standing_pre += (is_stand & ~switched).float()
        s = inner.scene["nonfoot_ground_touch"]
        fh = s.data.force_history if s.data.force_history is not None else s.data.force
        slam |= th.norm(fh, dim=-1).flatten(1).amax(1) > (80.0 + 1.3 * W)
        tipped |= inner.termination_manager._term_dones.get("fell_over", th.zeros(N, device=DEV)).bool()
        h_trace.append(float(d.root_link_pos_w[:, 2].mean())); sw_trace.append(float(switched.float().mean()))

    h_end = float(inner.scene["robot"].data.root_link_pos_w[:, 2].mean())
    env.close()
    res = dict(afford_pre=float(standing_pre.mean()) / STEPS, stand_frac=float(standing_t.mean()) / STEPS,
               tip=float(tipped.float().mean()), slam=float(slam.float().mean()), h_end=h_end,
               switched_frac=float(switched.float().mean()),
               Vs_trace=Vs_trace, h_trace=h_trace, sw_trace=sw_trace)
    did = switch_step >= 0
    if did.any():
        res.update(switchW_mean=float(switch_W[did].mean()), switchW_std=float(switch_W[did].std()),
                   switchT_mean=float(switch_step[did].float().mean()) * 0.02,
                   switch_frac_ev=float(did.float().mean()))
    return res


if __name__ == "__main__":
    e074 = json.load(open(E074_JSON))["arms"]
    results = {"config": dict(N=N, STEPS=STEPS, ramp=(RAMP_START, RAMP_END, W_MAX), eps=EPS, hyst=HYST,
                              ema_alpha=ALPHA, settle=SETTLE), "arms": {}}
    txt = ["E075 PART A.2 — RECAL gust-ramp (STAND-ONLY + HANDOFF re-run w/ recal; REST/UNIFIED copied E074).",
           f"eps={EPS} hyst={HYST}. cols: afford=stand% pre-switch | tip=fell_over | slam=nonfoot>80+1.3W",
           f"{'arm':>18} {'afford':>7} {'stand%':>7} {'tip':>5} {'slam':>5} {'h_end':>6} "
           f"{'swW_mu':>7} {'swW_sd':>7} {'swT_s':>6} {'sw_ev':>6}"]
    print(txt[0]); print(txt[2])
    for arm, (sck, rck, rule) in ARMS.items():
        r = rollout(sck, rck, rule)
        results["arms"][arm] = r
        line = (f"{arm:>18} {r['afford_pre']:>7.2f} {r['stand_frac']:>7.2f} {r['tip']:>5.2f} "
                f"{r['slam']:>5.2f} {r['h_end']:>6.2f} {r.get('switchW_mean', float('nan')):>7.1f} "
                f"{r.get('switchW_std', float('nan')):>7.1f} {r.get('switchT_mean', float('nan')):>6.2f} "
                f"{r.get('switch_frac_ev', float('nan')):>6.2f}")
        print(line); txt.append(line)
    # copied E074 rows for the fixed-ODD baselines (unchanged twins)
    txt.append("  --- copied from E074 (unchanged arms) ---")
    print(txt[-1])
    for arm in ["REST-ONLY", "UNIFIED", "UNIFIED-DISC", "HANDOFF", "STAND-ONLY"]:
        a = e074[arm]
        results["arms"][f"E074:{arm}"] = a
        line = (f"{'E074:'+arm:>18} {a['afford_pre']:>7.2f} {a['stand_frac']:>7.2f} {a['tip']:>5.2f} "
                f"{a['slam']:>5.2f} {a['h_end']:>6.2f} {a.get('switchW_mean', float('nan')):>7.1f} "
                f"{a.get('switchW_std', float('nan')):>7.1f} {a.get('switchT_mean', float('nan')):>6.2f} "
                f"{a.get('switch_frac_ev', float('nan')):>6.2f}")
        print(line); txt.append(line)

    rec = results["arms"]["HANDOFF(recal)"]; e = e074["HANDOFF"]
    verdict = [
        "",
        f"TRIGGER PRECISION: recal switch-W {rec.get('switchW_mean',float('nan')):.0f}±"
        f"{rec.get('switchW_std',float('nan')):.0f} (fire {rec.get('switch_frac_ev',0):.2f})  vs  "
        f"E074 {e.get('switchW_mean',float('nan')):.0f}±{e.get('switchW_std',float('nan')):.0f} "
        f"(fire {e.get('switch_frac_ev',0):.2f}).  oracle safe band W∈[40,120] tips 0.03-0.05.",
        f"SAFETY: recal handoff tip {rec['tip']:.2f} slam {rec['slam']:.2f} afford {rec['afford_pre']:.2f}  vs  "
        f"E074 handoff tip {e['tip']:.2f} slam {e['slam']:.2f} afford {e['afford_pre']:.2f}.",
    ]
    for s in verdict:
        print(s); txt.append(s)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "partA_ramp.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(OUT, "partA_ramp.txt"), "w") as f:
        f.write("\n".join(txt))
    print(f"\nwrote {OUT}/partA_ramp.json + .txt")
