"""E074 (T003) Task 2 — V_stand_hi contraction + eps recalibration + CERTIFICATION-DEFICIT check.

Under the STAND-hi policy at pull 10N, warm ~150 steps at each fixed W, then read:
  * V_stand   = stand_hi twin's own reach-avoid value at the state (>=0 == standing still certifiable).
  * V_unified = unified_hi twin's value at the SAME state (its cert signal along the stand trajectory).
W ∈ {0,30,60,90,120,150,200,250}, load driven per step (base_load + _weight_h + _weight_W).

eps = midpoint of {mean V_stand over the CERTIFIABLE band} and {mean V_stand over the FAILED band}, where
the bands are read from Task 1 (feasible standing W<=150, failed W>=200). The certification-deficit read:
does V_unified stay >=0 / flat across W (no handoff signal) while V_stand crosses 0 (the single-net baseline
cannot emit the switch the spec-family's certificate does)?
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

DEV, N, WARM = "cuda:0", 128, 150
WS = [0, 30, 60, 90, 120, 150, 200, 250]
LOAD_H, PULL_SCALE = 0.25, 0.20
CERT_BAND = [0, 30, 60, 90, 120, 150]      # feasible standing (Task 1)
FAIL_BAND = [200, 250]                      # standing fails (Task 1)
PULL = th.tensor([0., 1., 0.])
CK = "results/go2_weight_runs/go2_weight_{m}_adv/checkpoints/model_49999872_steps.zip"
TASK = "go2_weight_rest_hi_at_0"
OUT = os.path.expanduser(_ART + "/E074-hicom-demo")


def curve():
    with contextlib.redirect_stdout(io.StringIO()):
        env = make_tensor(TASK, N, DEV, adversary=True)
        m_stand, n_stand = load_twin(CK.format(m="stand_hi"), DEV, quiet=True)
        m_uni, n_uni = load_twin(CK.format(m="unified_hi"), DEV, quiet=True)
    inner = env.mj
    dstb = PULL.to(DEV)[None].expand(N, spec(TASK).dstb_dim).contiguous()
    rows = {}
    for W in WS:
        obs = env.reset()
        st = th.zeros(N, device=DEV)
        for _ in range(WARM):
            inner._weight_h = th.full((N,), LOAD_H, device=DEV)
            inner._weight_W = th.full((N,), float(W), device=DEV)
            env.base_load = th.tensor([0., 0., -float(W)], device=DEV)[None].expand(N, 3).contiguous()
            env.force_scale = PULL_SCALE * th.ones(N, device=DEV)
            with th.no_grad():
                a = th.clamp(m_stand.policy._predict(n_stand(obs), deterministic=True), -1, 1)
            obs, g, dones, touts, l = env.step_tensor(th.cat([a, dstb], dim=1))
            d = inner.scene["robot"].data
            pg = d.projected_gravity_b
            tilt = th.maximum(pg[:, 0].abs(), pg[:, 1].abs())
            st += ((d.root_link_pos_w[:, 2] > 0.18) & (tilt < 0.3)).float()
        # both values at the SAME (stand-policy-generated) state
        Vs = stand_value(env, m_stand, n_stand)
        Vu = stand_value(env, m_uni, n_uni)
        rows[W] = dict(V_stand=float(Vs.mean()), V_stand_std=float(Vs.std()),
                       V_unified=float(Vu.mean()), V_unified_std=float(Vu.std()),
                       stand_frac=float(st.mean()) / WARM)
    env.close()
    return rows


if __name__ == "__main__":
    rows = curve()
    cert = sum(rows[W]["V_stand"] for W in CERT_BAND) / len(CERT_BAND)
    fail = sum(rows[W]["V_stand"] for W in FAIL_BAND) / len(FAIL_BAND)
    eps = 0.5 * (cert + fail)
    ucert = sum(rows[W]["V_unified"] for W in CERT_BAND) / len(CERT_BAND)
    ufail = sum(rows[W]["V_unified"] for W in FAIL_BAND) / len(FAIL_BAND)

    txt = ["E074 TASK 2 — V_stand_hi contraction + V_unified deficit (STAND-hi policy, 10N pull, warm 150)",
           f"{'W(N)':>5} {'V_stand':>9} {'±std':>7} {'V_unified':>10} {'±std':>7} {'stand%':>7}"]
    print(txt[0]); print(txt[1])
    for W in WS:
        r = rows[W]
        line = (f"{W:>5} {r['V_stand']:>9.4f} {r['V_stand_std']:>7.4f} "
                f"{r['V_unified']:>10.4f} {r['V_unified_std']:>7.4f} {r['stand_frac']:>7.2f}")
        print(line); txt.append(line)
    summary = [
        "",
        f"V_stand  : certifiable-band mean={cert:.4f}  failed-band mean={fail:.4f}  => EPS={eps:.4f}",
        f"V_unified: certifiable-band mean={ucert:.4f}  failed-band mean={ufail:.4f}  "
        f"(deficit: Δ={ufail-ucert:+.4f} vs V_stand Δ={fail-cert:+.4f})",
        f"CERTIFICATION DEFICIT: V_stand crosses (cert {cert:+.4f} -> fail {fail:+.4f}); "
        f"V_unified {'stays flat/ >=0' if ufail > -0.005 else 'also moves'} ({ucert:+.4f} -> {ufail:+.4f}).",
    ]
    for s in summary:
        print(s); txt.append(s)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "task2_value.json"), "w") as f:
        json.dump(dict(rows=rows, eps=eps, cert_mean=cert, fail_mean=fail,
                       u_cert_mean=ucert, u_fail_mean=ufail,
                       cert_band=CERT_BAND, fail_band=FAIL_BAND), f, indent=2)
    with open(os.path.join(OUT, "task2_value.txt"), "w") as f:
        f.write("\n".join(txt))
    print(f"\nwrote {OUT}/task2_value.json + .txt  | EPS={eps:.4f}")
