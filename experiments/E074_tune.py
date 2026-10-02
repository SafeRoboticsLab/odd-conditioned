"""E074 trigger tuning (scratch): (A) forced stand->rest switch at fixed W to map the SAFE switching window;
(B) EMA-smoothed V_stand trigger with a settle lockout, sweeping eps. Loads models once. Not a deliverable."""
import os, sys, io, contextlib
import torch as th
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "external/robot-safety-sandbox")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from robot_safety_sandbox import make_tensor, spec
from robot_safety_sandbox.eval.policies import load_twin
from _value_util import stand_value

DEV, N, STEPS = "cuda:0", 128, 500
RAMP_START, RAMP_END, W_MAX = 100, 400, 250.0
LOAD_H, AMBIENT, GUST_SCALE, GUST_STEPS = 0.25, 0.20, 0.70, 25
GUST_STARTS = [75, 250, 325, 425]
PULL = th.tensor([0., 1., 0.])
CK = "results/go2_weight_runs/go2_weight_{m}_adv/checkpoints/model_49999872_steps.zip"
TASK = "go2_weight_rest_hi_at_0"


def W_of(t):
    if t < RAMP_START: return 0.0
    if t < RAMP_END: return (t - RAMP_START) / (RAMP_END - RAMP_START) * W_MAX
    return W_MAX
def in_gust(t): return any(g <= t < g + GUST_STEPS for g in GUST_STARTS)


def models():
    with contextlib.redirect_stdout(io.StringIO()):
        return {m: load_twin(CK.format(m=m), DEV, quiet=True)
                for m in ["stand_hi", "rest_hi", "unified_hi"]}


def run(M, mode, W_sw=None, eps=None, settle=100, alpha=0.12, hyst=8):
    m_s, n_s = M["stand_hi"]; m_r, n_r = M["rest_hi"]; m_vs, n_vs = M["stand_hi"]
    with contextlib.redirect_stdout(io.StringIO()):
        env = make_tensor(TASK, N, DEV, adversary=True)
    inner = env.mj
    dstb = PULL.to(DEV)[None].expand(N, spec(TASK).dstb_dim).contiguous()
    obs = env.reset()
    switched = th.zeros(N, dtype=th.bool, device=DEV)
    below = th.zeros(N, device=DEV); ema = th.zeros(N, device=DEV); ema_init = False
    swW = th.full((N,), -1.0, device=DEV); swT = th.full((N,), -1, dtype=th.long, device=DEV)
    stand_pre = th.zeros(N, device=DEV); tipped = th.zeros(N, dtype=th.bool, device=DEV)
    slam = th.zeros(N, dtype=th.bool, device=DEV)
    for t in range(STEPS):
        W = W_of(t)
        inner._weight_h = th.full((N,), LOAD_H, device=DEV)
        inner._weight_W = th.full((N,), W, device=DEV)
        env.base_load = th.tensor([0., 0., -W], device=DEV)[None].expand(N, 3).contiguous()
        env.force_scale = (GUST_SCALE if in_gust(t) else AMBIENT) * th.ones(N, device=DEV)
        Vs = stand_value(env, m_vs, n_vs)
        if not ema_init: ema = Vs.clone(); ema_init = True
        else: ema = alpha * Vs + (1 - alpha) * ema
        if mode == "V":
            active = t >= settle
            below = th.where((ema < eps) & active, below + 1.0, th.zeros_like(below))
            new = (below >= hyst) & ~switched
        else:  # forced W
            new = (th.full((N,), W, device=DEV) >= W_sw) & ~switched & (t >= settle)
        if new.any(): swW[new] = W; swT[new] = t
        switched |= new
        with th.no_grad():
            a_s = th.clamp(m_s.policy._predict(n_s(obs), deterministic=True), -1, 1)
            a_r = th.clamp(m_r.policy._predict(n_r(obs), deterministic=True), -1, 1)
        a = th.where(switched.unsqueeze(-1), a_r, a_s)
        obs, g, dones, touts, l = env.step_tensor(th.cat([a, dstb], dim=1))
        d = inner.scene["robot"].data
        pg = d.projected_gravity_b; tilt = th.maximum(pg[:, 0].abs(), pg[:, 1].abs())
        stand_pre += (((d.root_link_pos_w[:, 2] > 0.18) & (tilt < 0.3)) & ~switched).float()
        s = inner.scene["nonfoot_ground_touch"]
        fh = s.data.force_history if s.data.force_history is not None else s.data.force
        slam |= th.norm(fh, dim=-1).flatten(1).amax(1) > (80.0 + 1.3 * W)
        tipped |= inner.termination_manager._term_dones.get("fell_over", th.zeros(N, device=DEV)).bool()
    h_end = float(inner.scene["robot"].data.root_link_pos_w[:, 2].mean())
    env.close()
    did = swT >= 0
    return dict(afford=float(stand_pre.mean()) / STEPS, tip=float(tipped.float().mean()),
                slam=float(slam.float().mean()), h_end=h_end,
                swW=float(swW[did].mean()) if did.any() else float("nan"),
                swWsd=float(swW[did].std()) if did.any() else float("nan"),
                ev=float(did.float().mean()))


if __name__ == "__main__":
    M = models()
    print("=== (A) FORCED switch at fixed W (settle=100): the safe switching window ===")
    print(f"{'W_sw':>6} {'afford':>7} {'tip':>5} {'slam':>5} {'h_end':>6} {'swW':>6} {'ev':>5}")
    for w in [1, 40, 80, 120, 160, 200, 240, 9999]:
        r = run(M, "W", W_sw=float(w))
        print(f"{w:>6} {r['afford']:>7.2f} {r['tip']:>5.2f} {r['slam']:>5.2f} {r['h_end']:>6.2f} {r['swW']:>6.1f} {r['ev']:>5.2f}")
    print("\n=== (B) EMA V_stand trigger: settle x eps x alpha sweep ===")
    print(f"{'settle':>6} {'eps':>6} {'alpha':>5} {'afford':>7} {'tip':>5} {'slam':>5} {'swW':>6} {'swWsd':>6} {'ev':>5}")
    for settle in [130, 155, 180]:
        for alpha in [0.06, 0.10]:
            for eps in [-0.04, -0.07, -0.10]:
                r = run(M, "V", eps=eps, settle=settle, alpha=alpha, hyst=8)
                print(f"{settle:>6} {eps:>6.2f} {alpha:>5.2f} {r['afford']:>7.2f} {r['tip']:>5.2f} "
                      f"{r['slam']:>5.2f} {r['swW']:>6.1f} {r['swWsd']:>6.1f} {r['ev']:>5.2f}")
