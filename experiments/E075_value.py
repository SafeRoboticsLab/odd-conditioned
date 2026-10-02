"""E075 (T003) Part A.1 — V_stand contraction for the RECALIBRATED stand_hi twin (trained W~U[0,120]).

E074's stand_hi was trained W~U[0,150] (mostly infeasible under the high-CoM pendulum, 0.78 train failure)
→ a NOISY value net → imprecise trigger (E074 handoff switch-W std ~79, tip 0.21 ≫ oracle 0.03-0.05).
E075 retrained on the comfortably-feasible band W~U[0,120] (0.52 final failure). This script measures whether
the recalibrated certificate DISCRIMINATES the feasible/failed W bands more cleanly (lower variance, sharper
crossing) than the E074 noisy stand_hi (cert-band V ≈ -0.014, fail-band V ≈ -0.200, high var).

Under the RECAL stand policy at pull 10N, warm ~150 steps at each fixed W, then read V_recal (its own
reach-avoid value at the state). W ∈ {0,30,60,90,120,150,200}. Same high-CoM physics as E074.
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
WS = [0, 30, 60, 90, 120, 150, 200]
LOAD_H, PULL_SCALE = 0.25, 0.20
CERT_BAND = [0, 30, 60, 90, 120]           # feasible standing (recal trained here)
FAIL_BAND = [200]                          # standing fails (OOD-heavy for recal)
PULL = th.tensor([0., 1., 0.])
RECAL = "results/go2_weight_runs/E075_recal/go2_weight_stand_hi_adv/checkpoints/model_49999872_steps.zip"
# E074's noisy stand_hi, for a same-state discrimination comparison:
E074_STAND = "results/go2_weight_runs/go2_weight_stand_hi_adv/checkpoints/model_49999872_steps.zip"
TASK = "go2_weight_rest_hi_at_0"
OUT = os.path.expanduser(_ART + "/E075-recal-eval")


def curve():
    with contextlib.redirect_stdout(io.StringIO()):
        env = make_tensor(TASK, N, DEV, adversary=True)
        m_recal, n_recal = load_twin(RECAL, DEV, quiet=True)
        m_e074, n_e074 = load_twin(E074_STAND, DEV, quiet=True)
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
                a = th.clamp(m_recal.policy._predict(n_recal(obs), deterministic=True), -1, 1)
            obs, g, dones, touts, l = env.step_tensor(th.cat([a, dstb], dim=1))
            d = inner.scene["robot"].data
            pg = d.projected_gravity_b
            tilt = th.maximum(pg[:, 0].abs(), pg[:, 1].abs())
            st += ((d.root_link_pos_w[:, 2] > 0.18) & (tilt < 0.3)).float()
        Vr = stand_value(env, m_recal, n_recal)      # recal twin's own value
        Ve = stand_value(env, m_e074, n_e074)        # E074 noisy stand_hi value at the SAME state
        rows[W] = dict(V_recal=float(Vr.mean()), V_recal_std=float(Vr.std()),
                       V_e074=float(Ve.mean()), V_e074_std=float(Ve.std()),
                       stand_frac=float(st.mean()) / WARM)
    env.close()
    return rows


if __name__ == "__main__":
    rows = curve()
    cert = sum(rows[W]["V_recal"] for W in CERT_BAND) / len(CERT_BAND)
    fail = sum(rows[W]["V_recal"] for W in FAIL_BAND) / len(FAIL_BAND)
    eps = 0.5 * (cert + fail)
    # discrimination = |cert-fail| / mean_std over the whole grid (higher = cleaner separation)
    mstd_r = sum(rows[W]["V_recal_std"] for W in WS) / len(WS)
    mstd_e = sum(rows[W]["V_e074_std"] for W in WS) / len(WS)
    ce = sum(rows[W]["V_e074"] for W in CERT_BAND) / len(CERT_BAND)
    fe = sum(rows[W]["V_e074"] for W in FAIL_BAND) / len(FAIL_BAND)

    txt = ["E075 PART A.1 — V_stand contraction: RECAL (W~U[0,120]) vs E074 noisy stand_hi (W~U[0,150]).",
           "Under RECAL stand policy, 10N pull, warm 150. V read from each twin at the SAME state.",
           f"{'W(N)':>5} {'V_recal':>9} {'±std':>7} {'V_e074':>9} {'±std':>7} {'stand%':>7}"]
    print(txt[0]); print(txt[2])
    for W in WS:
        r = rows[W]
        line = (f"{W:>5} {r['V_recal']:>9.4f} {r['V_recal_std']:>7.4f} "
                f"{r['V_e074']:>9.4f} {r['V_e074_std']:>7.4f} {r['stand_frac']:>7.2f}")
        print(line); txt.append(line)
    summary = [
        "",
        f"RECAL : cert-band mean={cert:+.4f}  fail-band mean={fail:+.4f}  Δ={fail-cert:+.4f}  "
        f"mean_std={mstd_r:.4f}  discrim=|Δ|/std={abs(fail-cert)/max(mstd_r,1e-6):.2f}  => EPS={eps:+.4f}",
        f"E074  : cert-band mean={ce:+.4f}  fail-band mean={fe:+.4f}  Δ={fe-ce:+.4f}  "
        f"mean_std={mstd_e:.4f}  discrim=|Δ|/std={abs(fe-ce)/max(mstd_e,1e-6):.2f}",
        f"VERDICT: recal discrimination {abs(fail-cert)/max(mstd_r,1e-6):.2f} vs E074 "
        f"{abs(fe-ce)/max(mstd_e,1e-6):.2f} (higher = cleaner certificate).",
    ]
    for s in summary:
        print(s); txt.append(s)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "partA_value.json"), "w") as f:
        json.dump(dict(rows=rows, eps=eps, cert_mean=cert, fail_mean=fail,
                       cert_band=CERT_BAND, fail_band=FAIL_BAND,
                       discrim_recal=abs(fail-cert)/max(mstd_r,1e-6),
                       discrim_e074=abs(fe-ce)/max(mstd_e,1e-6),
                       mean_std_recal=mstd_r, mean_std_e074=mstd_e), f, indent=2)
    with open(os.path.join(OUT, "partA_value.txt"), "w") as f:
        f.write("\n".join(txt))
    print(f"\nwrote {OUT}/partA_value.json + .txt  | EPS={eps:+.4f}")
