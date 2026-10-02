"""E076 (T004) Part B.2 — V_leg_stand contraction vs θ (the leg-ODD certificate).

Under the leg_stand policy at 10N pull, warm ~150 steps at each fixed θ, then read V_leg_stand (the
leg_stand twin's own reach-avoid value at the state; >=0 == stance still certifiable). θ driven per step
via _fr_torque_frac + actuator_forcerange (E060 machinery). θ ∈ {1.0,0.8,0.6,0.5,0.4,0.2,0.1,0.0}.

CRITICAL GATE: does V_leg_stand DISCRIMINATE θ (contracts as the leg dies) or is it FLAT? If flat, the
handoff has no trigger signal — report and skip the handoff. eps = midpoint of {mean V over the certifiable
band θ≥0.5} and {mean V over the failed band θ≤0.1}. (leg_stand had 0.37 train failure → expect V noise.)
"""
import os, sys, io, contextlib, json
from _paths import _ART
import torch as th
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "external/robot-safety-sandbox")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from robot_safety_sandbox import make_tensor, spec
from robot_safety_sandbox.eval.policies import load_twin
from robot_safety_sandbox.envs.go2_broken_leg.env_cfg import _ensure_fr_cache
from _value_util import stand_value

DEV, N, WARM = "cuda:0", 128, 150
THETAS = [1.0, 0.8, 0.6, 0.5, 0.4, 0.2, 0.1, 0.0]
CERT_BAND = [1.0, 0.8, 0.6, 0.5]
FAIL_BAND = [0.1, 0.0]
PULL_SCALE = 0.20
LEG_STAND = "results/go2_leg_family_runs/go2_leg_stand_adv/checkpoints/model_49999872_steps.zip"
PULL = th.tensor([0., 1., 0.])
TASK = "go2_leg_rest"                              # common surface (θ driven per step); actor obs identical
OUT = os.path.expanduser(_ART + "/E076-leg-demo")


def curve():
    with contextlib.redirect_stdout(io.StringIO()):
        env = make_tensor(TASK, N, DEV, adversary=True)
        m, n = load_twin(LEG_STAND, DEV, quiet=True)
    inner = env.mj
    _ensure_fr_cache(inner)
    ids, nominal = inner._fr_act_ids, inner._fr_nominal_forcerange
    dstb = (PULL / PULL.norm()).to(DEV)[None].expand(N, spec(TASK).dstb_dim).contiguous()

    def set_theta(th_):
        inner.sim.model.actuator_forcerange[:, ids, :] = nominal * float(th_)
        inner._fr_torque_frac[:] = float(th_)

    rows = {}
    for theta in THETAS:
        obs = env.reset()
        st = th.zeros(N, device=DEV)
        for _ in range(WARM):
            set_theta(theta)
            env.force_scale = PULL_SCALE * th.ones(N, device=DEV)
            with th.no_grad():
                a = th.clamp(m.policy._predict(n(obs), deterministic=True), -1, 1)
            obs, g, dones, touts, l = env.step_tensor(th.cat([a, dstb], dim=1))
            d = inner.scene["robot"].data
            pg = d.projected_gravity_b
            tilt = th.maximum(pg[:, 0].abs(), pg[:, 1].abs())
            st += ((d.root_link_pos_w[:, 2] > 0.18) & (tilt < 0.3)).float()
        V = stand_value(env, m, n)
        rows[theta] = dict(V=float(V.mean()), V_std=float(V.std()), stand_frac=float(st.mean()) / WARM)
    env.close()
    return rows


if __name__ == "__main__":
    rows = curve()
    cert = sum(rows[t]["V"] for t in CERT_BAND) / len(CERT_BAND)
    fail = sum(rows[t]["V"] for t in FAIL_BAND) / len(FAIL_BAND)
    eps = 0.5 * (cert + fail)
    mstd = sum(rows[t]["V_std"] for t in THETAS) / len(THETAS)
    discrim = abs(cert - fail) / max(mstd, 1e-6)

    txt = ["E076 PART B.2 — V_leg_stand contraction vs θ (leg_stand policy, 10N pull, warm 150).",
           f"{'θ':>5} {'V_stand':>9} {'±std':>7} {'stand%':>7}"]
    print(txt[0]); print(txt[1])
    for t in THETAS:
        r = rows[t]
        line = f"{t:>5.1f} {r['V']:>9.4f} {r['V_std']:>7.4f} {r['stand_frac']:>7.2f}"
        print(line); txt.append(line)
    summary = [
        "",
        f"cert-band (θ≥0.5) mean={cert:+.4f}  fail-band (θ≤0.1) mean={fail:+.4f}  Δ={cert-fail:+.4f}",
        f"mean_std={mstd:.4f}  DISCRIMINATION=|Δ|/std={discrim:.2f}  => EPS={eps:+.4f}",
        f"GATE: {'DISCRIMINATES (contracts as leg dies) — handoff viable' if discrim > 0.5 and cert > fail else 'FLAT / NON-MONOTONE — handoff trigger NOT reliable'}.",
    ]
    for s in summary:
        print(s); txt.append(s)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "partB_value.json"), "w") as f:
        json.dump(dict(rows=rows, eps=eps, cert_mean=cert, fail_mean=fail, discrim=discrim,
                       cert_band=CERT_BAND, fail_band=FAIL_BAND, mean_std=mstd), f, indent=2)
    with open(os.path.join(OUT, "partB_value.txt"), "w") as f:
        f.write("\n".join(txt))
    print(f"\nwrote {OUT}/partB_value.json + .txt  | EPS={eps:+.4f} discrim={discrim:.2f}")
