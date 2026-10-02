"""E074 (T003) Task 3 — the 5-arm GUST-RAMP demo (HIGH-CoM inverted-pendulum load).

W ramp: 0 until t=2s, linear to 250N by t=8s, then hold. Ambient 10N lateral pull + 25N GUST pulses (0.5s)
at t ∈ {1.5s (low W — a standing robot should SURVIVE = affordance), 5s, 6.5s, 8.5s (mid/high W — these
should TOPPLE a standing robot as the raised CoM torques it over)}.

Arms (N=128):
  STAND-ONLY   : stand_hi throughout (overconfident fixed-ODD filter).
  REST-ONLY    : rest_hi throughout (worst-case fixed-ODD filter, no affordance).
  HANDOFF      : stand_hi until V_stand<eps (5-step hysteresis) -> permanent switch to rest_hi.
  UNIFIED      : unified_hi (single-net baseline, l=max(l_stand,l_rest)).
  UNIFIED-DISC : unified_disc_hi (rest_discount=0.5, prefers standing while feasible).

Metrics per arm: affordance (stand% pre-switch), tip (fell_over), slam (nonfoot>80+1.3W), h_end,
trigger time/W stats; plus per-step mean V_stand and V_unified traces (the certification-deficit figure).
Success signature: HANDOFF survives the early low-W gust STANDING, switches before the mid-ramp gusts, ~0
tip; STAND-ONLY tips at the mid/high-W gusts; UNIFIED lazy-descends; UNIFIED-DISC hedges.
"""
import os, sys, io, contextlib, json
import torch as th
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "external/robot-safety-sandbox")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from robot_safety_sandbox import make_tensor, spec
from robot_safety_sandbox.eval.policies import load_twin
from _value_util import stand_value

DEV, N, STEPS = "cuda:0", 128, 500
RAMP_START, RAMP_END, W_MAX = 100, 400, 250.0      # t=2s..8s, hold after
LOAD_H = 0.25
AMBIENT, GUST_SCALE = 0.20, 0.70                   # 10N ambient / 35N (10+25 gust)
GUST_STEPS = 25                                    # 0.5 s
GUST_STARTS = [75, 250, 325, 425]                  # t = 1.5, 5.0, 6.5, 8.5 s
# Trigger: EMA-smoothed V_stand (raw per-step V is too noisy under high-CoM — the barely-feasible stand_hi
# value net; Task-2 raw midpoint was -0.107). SETTLE lockout skips the spawn/first-gust transient; eps=-0.04
# lands the switch inside the mechanically-safe window W∈[40,120] found by the forced-switch sweep (E074_tune).
EPS, HYST, ALPHA, SETTLE = -0.04, 8, 0.10, 130
PULL = th.tensor([0., 1., 0.])
CK = "results/go2_weight_runs/go2_weight_{m}_adv/checkpoints/model_49999872_steps.zip"
TASK = "go2_weight_rest_hi_at_0"
OUT = os.path.expanduser("~/artifacts/odd-conditioned/E074-hicom-demo")

# arm -> (stand-phase policy, rest-phase policy, switch rule). "V"=value trigger, "none"=stay stand policy,
# "all"=stay rest policy from t=0.
ARMS = {
    "STAND-ONLY":   ("stand_hi",          "stand_hi",   "none"),
    "REST-ONLY":    ("rest_hi",           "rest_hi",    "all"),
    "HANDOFF":      ("stand_hi",           "rest_hi",   "V"),
    "UNIFIED":      ("unified_hi",         "unified_hi", "none"),
    "UNIFIED-DISC": ("unified_disc_hi",    "unified_disc_hi", "none"),
}


def W_of(t):
    if t < RAMP_START:
        return 0.0
    if t < RAMP_END:
        return (t - RAMP_START) / (RAMP_END - RAMP_START) * W_MAX
    return W_MAX


def in_gust(t):
    return any(g <= t < g + GUST_STEPS for g in GUST_STARTS)


def load_models():
    ms = {}
    with contextlib.redirect_stdout(io.StringIO()):
        for m in ["stand_hi", "rest_hi", "unified_hi", "unified_disc_hi"]:
            ms[m] = load_twin(CK.format(m=m), DEV, quiet=True)
    for mdl, _ in ms.values():
        mdl.policy.set_training_mode(False)
    return ms


def rollout(arm, models):
    p_stand, p_rest, rule = ARMS[arm]
    m_s, n_s = models[p_stand]
    m_r, n_r = models[p_rest]
    m_vs, n_vs = models["stand_hi"]                # V_stand readout twin
    m_vu, n_vu = models["unified_hi"]              # V_unified readout twin
    with contextlib.redirect_stdout(io.StringIO()):
        env = make_tensor(TASK, N, DEV, adversary=True)
    inner = env.mj
    dstb = PULL.to(DEV)[None].expand(N, spec(TASK).dstb_dim).contiguous()
    obs = env.reset()

    switched = th.zeros(N, dtype=th.bool, device=DEV)
    if rule == "all":
        switched[:] = True
    below = th.zeros(N, device=DEV)
    ema = th.zeros(N, device=DEV); ema_init = False
    switch_step = th.full((N,), -1, dtype=th.long, device=DEV)
    switch_W = th.full((N,), -1.0, device=DEV)
    standing_t = th.zeros(N, device=DEV)
    standing_pre = th.zeros(N, device=DEV)
    tipped = th.zeros(N, dtype=th.bool, device=DEV)
    slam = th.zeros(N, dtype=th.bool, device=DEV)
    Vs_trace, Vu_trace, h_trace, W_trace, sw_trace = [], [], [], [], []

    for t in range(STEPS):
        W = W_of(t)
        inner._weight_h = th.full((N,), LOAD_H, device=DEV)
        inner._weight_W = th.full((N,), W, device=DEV)
        env.base_load = th.tensor([0., 0., -W], device=DEV)[None].expand(N, 3).contiguous()
        env.force_scale = (GUST_SCALE if in_gust(t) else AMBIENT) * th.ones(N, device=DEV)

        Vs = stand_value(env, m_vs, n_vs)
        Vu = stand_value(env, m_vu, n_vu)
        Vs_trace.append(float(Vs.mean())); Vu_trace.append(float(Vu.mean()))

        if not ema_init:
            ema = Vs.clone(); ema_init = True
        else:
            ema = ALPHA * Vs + (1 - ALPHA) * ema
        if rule == "V":
            active = t >= SETTLE
            below = th.where((ema < EPS) & active, below + 1.0, th.zeros_like(below))
            new = (below >= HYST) & ~switched
            if new.any():
                switch_step[new] = t
                switch_W[new] = W
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
        standing_t += is_stand.float()
        standing_pre += (is_stand & ~switched).float()
        s = inner.scene["nonfoot_ground_touch"]
        fh = s.data.force_history if s.data.force_history is not None else s.data.force
        slam |= th.norm(fh, dim=-1).flatten(1).amax(1) > (80.0 + 1.3 * W)
        tipped |= inner.termination_manager._term_dones.get(
            "fell_over", th.zeros(N, device=DEV)).bool()
        h_trace.append(float(d.root_link_pos_w[:, 2].mean()))
        W_trace.append(W); sw_trace.append(float(switched.float().mean()))

    h_end = float(inner.scene["robot"].data.root_link_pos_w[:, 2].mean())
    env.close()
    res = dict(afford_pre=float(standing_pre.mean()) / STEPS,
               stand_frac=float(standing_t.mean()) / STEPS,
               tip=float(tipped.float().mean()), slam=float(slam.float().mean()),
               h_end=h_end, switched_frac=float(switched.float().mean()),
               Vs_trace=Vs_trace, Vu_trace=Vu_trace, h_trace=h_trace, sw_trace=sw_trace)
    did = switch_step >= 0
    if did.any():
        res.update(switchW_mean=float(switch_W[did].mean()), switchW_std=float(switch_W[did].std()),
                   switchT_mean=float(switch_step[did].float().mean()) * 0.02,
                   switch_frac_ev=float(did.float().mean()))
    return res


if __name__ == "__main__":
    models = load_models()
    Wv = [W_of(t) for t in range(STEPS)]
    results = {"config": dict(N=N, STEPS=STEPS, ramp=(RAMP_START, RAMP_END, W_MAX), eps=EPS, hyst=HYST,
                              ema_alpha=ALPHA, settle=SETTLE, gust_starts=GUST_STARTS, gust_steps=GUST_STEPS,
                              load_h=LOAD_H, ambient=AMBIENT, gust_scale=GUST_SCALE), "W_trace": Wv, "arms": {}}
    txt = ["E074 TASK 3 — 5-arm GUST-RAMP demo (HIGH-CoM). W: 0->250N over t=2..8s; gusts@t=1.5/5/6.5/8.5s.",
           f"eps={EPS} hyst={HYST}. cols: afford=stand% pre-switch | tip=fell_over | slam=nonfoot>80+1.3W",
           f"{'arm':>13} {'afford':>7} {'stand%':>7} {'tip':>5} {'slam':>5} {'h_end':>6} "
           f"{'swW_mu':>7} {'swW_sd':>7} {'swT_s':>6} {'sw_ev':>6}"]
    print(txt[0]); print(txt[2])
    for arm in ARMS:
        r = rollout(arm, models)
        results["arms"][arm] = r
        line = (f"{arm:>13} {r['afford_pre']:>7.2f} {r['stand_frac']:>7.2f} {r['tip']:>5.2f} "
                f"{r['slam']:>5.2f} {r['h_end']:>6.2f} {r.get('switchW_mean', float('nan')):>7.1f} "
                f"{r.get('switchW_std', float('nan')):>7.1f} {r.get('switchT_mean', float('nan')):>6.2f} "
                f"{r.get('switch_frac_ev', float('nan')):>6.2f}")
        print(line); txt.append(line)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "task3_ramp.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(OUT, "task3_ramp.txt"), "w") as f:
        f.write("\n".join(txt))
    print(f"\nwrote {OUT}/task3_ramp.json + .txt")
