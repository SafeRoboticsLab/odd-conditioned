"""E078 (T004, DEMO 2) Task 2 — THE DECISIVE TEST: V_compound_stand(θ) contraction.

Under the compound_stand policy at 10N pull, carrying the constant high-CoM load W=80@h=0.25, warm ~150
steps at each fixed θ, then read V_compound_stand (the stand twin's own reach-avoid value; >=0 == stance
still certifiable). θ driven per step via _fr_torque_frac + actuator_forcerange; W driven per step via
base_load + _weight_W + _weight_h (the compound driving). θ ∈ {1.0,0.8,0.6,0.5,0.4,0.3,0.2,0.1,0.0}.

THE CRUX (T004 gate): unlike the UNLOADED leg-at-stance certificate (E076: discrim 0.18, FLAT — reactive
absorb), does the loaded certificate CONTRACT as the leg dies? Discrimination = |Δ|/σ between the FEASIBLE
band (θ≥0.5) and the FAILED band (θ≤0.2). Compare to flat leg (0.18) and weight (1.16). If discrim ≥ ~0.8
and monotone-ish → contracts → pick eps and PROCEED to the handoff. If flat → STOP, report loudly.
"""
import os, sys, io, contextlib, json
import torch as th
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "external/robot-safety-sandbox")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from robot_safety_sandbox import make_tensor, spec
from robot_safety_sandbox.eval.policies import load_twin
from robot_safety_sandbox.envs.go2_broken_leg.env_cfg import _ensure_fr_cache
from _value_util import stand_value

DEV, N, WARM = "cuda:0", 128, 150
THETAS = [1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
CERT_BAND = [1.0, 0.8, 0.6, 0.5]     # feasible band θ≥0.5
FAIL_BAND = [0.2, 0.1, 0.0]          # failed band θ≤0.2
PULL_SCALE = 0.20
W, LOAD_H = 80.0, 0.25
STAND = "results/go2_compound_runs/go2_compound_stand_adv/checkpoints/model_49999872_steps.zip"
PULL = th.tensor([0., 1., 0.])
TASK = "go2_compound_rest"                          # common surface (θ+W driven per step); actor obs identical
OUT = os.path.expanduser("~/artifacts/odd-conditioned/E078-compound-demo")
REF = dict(flat_leg=0.18, weight=1.16)             # E076 (unloaded leg, FLAT) / E075 (weight, contracts)


def curve():
    with contextlib.redirect_stdout(io.StringIO()):
        env = make_tensor(TASK, N, DEV, adversary=True)
        m, n = load_twin(STAND, DEV, quiet=True)
    inner = env.mj
    _ensure_fr_cache(inner)
    ids, nominal = inner._fr_act_ids, inner._fr_nominal_forcerange
    dstb = (PULL / PULL.norm()).to(DEV)[None].expand(N, spec(TASK).dstb_dim).contiguous()

    def drive(th_):
        inner.sim.model.actuator_forcerange[:, ids, :] = nominal * float(th_)
        inner._fr_torque_frac[:] = float(th_)
        inner._weight_W = th.full((N,), W, device=DEV)
        inner._weight_h = th.full((N,), LOAD_H, device=DEV)
        env.base_load = th.tensor([0., 0., -W], device=DEV)[None].expand(N, 3).contiguous()

    rows = {}
    for theta in THETAS:
        obs = env.reset()
        st = th.zeros(N, device=DEV)
        for _ in range(WARM):
            drive(theta)
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

    txt = ["E078 TASK 2 — V_compound_stand contraction vs θ (compound_stand policy, W=80@h=0.25, 10N pull, warm 150).",
           f"{'θ':>5} {'V_stand':>9} {'±std':>7} {'stand%':>7}"]
    print(txt[0]); print(txt[1])
    for t in THETAS:
        r = rows[t]
        line = f"{t:>5.1f} {r['V']:>9.4f} {r['V_std']:>7.4f} {r['stand_frac']:>7.2f}"
        print(line); txt.append(line)
    contracts = discrim >= 0.8 and cert > fail
    summary = [
        "",
        f"cert-band (θ≥0.5) mean={cert:+.4f}  fail-band (θ≤0.2) mean={fail:+.4f}  Δ={cert-fail:+.4f}",
        f"mean_std={mstd:.4f}  DISCRIMINATION=|Δ|/std={discrim:.2f}  => EPS={eps:+.4f}",
        f"COMPARE: compound {discrim:.2f}  vs  flat-leg {REF['flat_leg']} (negative control)  vs  weight {REF['weight']}.",
        f"VERDICT: {'CONTRACTS (discrim>=0.8, cert>fail) — handoff viable, PROCEED to task 3.' if contracts else 'FLAT — training absorbed the loaded case; STOP and report loudly.'}",
    ]
    for s in summary:
        print(s); txt.append(s)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "task2_value.json"), "w") as f:
        json.dump(dict(rows=rows, eps=eps, cert_mean=cert, fail_mean=fail, discrim=discrim,
                       cert_band=CERT_BAND, fail_band=FAIL_BAND, mean_std=mstd, ref=REF,
                       contracts=contracts), f, indent=2)
    with open(os.path.join(OUT, "task2_value.txt"), "w") as f:
        f.write("\n".join(txt))
    print(f"\nwrote {OUT}/task2_value.json + .txt  | EPS={eps:+.4f} discrim={discrim:.2f}")
