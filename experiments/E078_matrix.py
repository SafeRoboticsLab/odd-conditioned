"""E078 (T004, DEMO 2) Task 1 — per-θ BEHAVIORAL MATRIX for both COMPOUND arms.

Compound ODD = FR-leg torque fraction θ WHILE carrying a constant high-CoM load W=80N @ h=0.25 (the
inverted pendulum). Both policies ∈ {compound_stand, compound_rest} × θ ∈ {1.0,0.8,0.6,0.5,0.4,0.3,0.2,
0.1,0.0}, 300 steps, on the permissive compound_rest cfg (illegal_contact 500N — cross-policy common
surface). Per step BOTH channels are driven:
  * WEIGHT: env.base_load=[0,0,-80]; inner._weight_W[:]=80 (SLAM_CAP obs/margin); inner._weight_h[:]=0.25
  * θ:      inner._fr_torque_frac[:]=θ AND actuator_forcerange[:, fr_ids, :] = nominal*θ (_ensure_fr_cache).
Pull = lateral force_scale: 10N ambient + a 25N gust (→35N, 0.5s) at t=3s (steps 150-175).

Per cell: stand%=(z>0.18 & tilt<0.3) | tip=fell_over | slam=nonfoot>184N (SLAM_CAP(80)) | h_end.
Expect: compound_stand good θ≥0.4, collapsing below (the trained θ_c≈0.3); compound_rest flat-safe incl. θ=0.
"""
import os, sys, io, contextlib, json
from _paths import _ART
import torch as th
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "external/robot-safety-sandbox")
from robot_safety_sandbox import make_tensor, spec
from robot_safety_sandbox.eval.policies import load_twin
from robot_safety_sandbox.envs.go2_broken_leg.env_cfg import _ensure_fr_cache

DEV, N, STEPS = "cuda:0", 128, 300
THETAS = [1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
POLICIES = {"compound_stand": "results/go2_compound_runs/go2_compound_stand_adv/checkpoints/model_49999872_steps.zip",
            "compound_rest":  "results/go2_compound_runs/go2_compound_rest_adv/checkpoints/model_49999872_steps.zip"}
T_GUST0, T_GUST1 = 150, 175                        # gust window t∈[3.0,3.5]s
AMBIENT, GUST = 0.20, 0.70                          # 10N ambient / 35N (10+25 gust) at force_max 50
W, LOAD_H = 80.0, 0.25
SLAM_N = 184.0                                     # SLAM_CAP(80)=80+1.3*80
PULL = th.tensor([0., 1., 0.])
TASK = "go2_compound_rest"                          # permissive common cfg (500N contact); θ+W driven per step
OUT = os.path.expanduser(_ART + "/E078-compound-demo")


def cell(ck, theta):
    with contextlib.redirect_stdout(io.StringIO()):
        env = make_tensor(TASK, N, DEV, adversary=True)
        model, norm = load_twin(ck, DEV, quiet=True)
    inner = env.mj
    _ensure_fr_cache(inner)
    ids, nominal = inner._fr_act_ids, inner._fr_nominal_forcerange
    dstb = (PULL / PULL.norm()).to(DEV)[None].expand(N, spec(TASK).dstb_dim).contiguous()
    obs = env.reset()

    def drive(th_):
        inner.sim.model.actuator_forcerange[:, ids, :] = nominal * float(th_)
        inner._fr_torque_frac[:] = float(th_)
        inner._weight_W = th.full((N,), W, device=DEV)
        inner._weight_h = th.full((N,), LOAD_H, device=DEV)
        env.base_load = th.tensor([0., 0., -W], device=DEV)[None].expand(N, 3).contiguous()
    drive(theta)
    standing_t = th.zeros(N, device=DEV)
    tipped = th.zeros(N, dtype=th.bool, device=DEV)
    slam = th.zeros(N, dtype=th.bool, device=DEV)
    for t in range(STEPS):
        drive(theta)
        env.force_scale = (GUST if T_GUST0 <= t < T_GUST1 else AMBIENT) * th.ones(N, device=DEV)
        with th.no_grad():
            a = th.clamp(model.policy._predict(norm(obs), deterministic=True), -1, 1)
        obs, g, dones, touts, l = env.step_tensor(th.cat([a, dstb], dim=1))
        d = inner.scene["robot"].data
        pg = d.projected_gravity_b
        tilt = th.maximum(pg[:, 0].abs(), pg[:, 1].abs())
        standing_t += ((d.root_link_pos_w[:, 2] > 0.18) & (tilt < 0.3)).float()
        s = inner.scene["nonfoot_ground_touch"]
        fh = s.data.force_history if s.data.force_history is not None else s.data.force
        slam |= th.norm(fh, dim=-1).flatten(1).amax(1) > SLAM_N
        tipped |= inner.termination_manager._term_dones.get("fell_over", th.zeros(N, device=DEV)).bool()
    h_end = float(inner.scene["robot"].data.root_link_pos_w[:, 2].mean())
    env.close()
    return dict(stand=float(standing_t.mean()) / STEPS, tip=float(tipped.float().mean()),
                slam=float(slam.float().mean()), h_end=h_end)


if __name__ == "__main__":
    results = {"config": dict(N=N, STEPS=STEPS, thetas=THETAS, gust=(T_GUST0, T_GUST1),
                              ambient=AMBIENT, gust_scale=GUST, slam_n=SLAM_N, W=W, load_h=LOAD_H), "matrix": {}}
    txt = ["E078 TASK 1 — per-θ behavioral matrix (COMPOUND: W=80@h=0.25 + FR θ, 10N ambient + 25N gust @t=3s, N=128 x 300)",
           "cols: stand%=(z>0.18 & tilt<0.3) frac | tip=fell_over | slam=nonfoot>184N (SLAM_CAP(80)) | h_end", ""]
    for name, ck in POLICIES.items():
        results["matrix"][name] = {}
        hdr = f"=== policy: go2_{name} ==="
        cols = f"{'θ':>5} {'stand%':>7} {'tip':>5} {'slam':>5} {'h_end':>6}"
        print("\n" + hdr); print(cols)
        txt.append(hdr); txt.append(cols)
        for theta in THETAS:
            r = cell(ck, theta)
            results["matrix"][name][theta] = r
            line = f"{theta:>5.1f} {r['stand']:>7.2f} {r['tip']:>5.2f} {r['slam']:>5.2f} {r['h_end']:>6.2f}"
            print(line); txt.append(line)
        txt.append("")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "task1_matrix.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(OUT, "task1_matrix.txt"), "w") as f:
        f.write("\n".join(txt))
    print(f"\nwrote {OUT}/task1_matrix.json + .txt")
