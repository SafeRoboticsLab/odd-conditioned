"""E076 (T004) Part B.1 — per-θ BEHAVIORAL MATRIX for both leg-family arms (leg-degradation ODD).

Both leg policies ∈ {leg_stand, leg_rest} × θ ∈ {1.0,0.8,0.6,0.5,0.4,0.2,0.1,0.0}, 300 steps, on the
permissive leg_rest cfg (illegal_contact raised to 500 N — cross-policy common surface, belly rest allowed).
θ driven per step: env.mj._fr_torque_frac[:] = θ AND actuator_forcerange[:, fr_ids, :] = nominal * θ
(the E060 machinery, via _ensure_fr_cache). Pull = lateral force_scale: 10N ambient + a 25N gust (→35N,
0.5s) at t=3s (steps 150-175).

Per cell: stand%=(z>0.18 & tilt<0.3) | tip=fell_over | slam=nonfoot>80N | h_end.
Expect: leg_stand good θ≥0.5, degrading below; leg_rest safe EVERYWHERE incl. θ=0 (the E061-era all-collapse
extreme) — that θ=0 cell for leg_rest is the highlight (a fully-dead leg, lie down, no slam/tip).
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
THETAS = [1.0, 0.8, 0.6, 0.5, 0.4, 0.2, 0.1, 0.0]
POLICIES = {"leg_stand": "results/go2_leg_family_runs/go2_leg_stand_adv/checkpoints/model_49999872_steps.zip",
            "leg_rest":  "results/go2_leg_family_runs/go2_leg_rest_adv/checkpoints/model_49999872_steps.zip"}
T_GUST0, T_GUST1 = 150, 175                        # gust window t∈[3.0,3.5]s
AMBIENT, GUST = 0.20, 0.70                          # 10N ambient / 35N (10+25 gust)
SLAM_N = 80.0
PULL = th.tensor([0., 1., 0.])
TASK = "go2_leg_rest"                              # permissive common cfg (500N contact term); θ driven per step
OUT = os.path.expanduser(_ART + "/E076-leg-demo")


def cell(ck, theta):
    with contextlib.redirect_stdout(io.StringIO()):
        env = make_tensor(TASK, N, DEV, adversary=True)
        model, norm = load_twin(ck, DEV, quiet=True)
    inner = env.mj
    _ensure_fr_cache(inner)
    ids, nominal = inner._fr_act_ids, inner._fr_nominal_forcerange
    dstb = (PULL / PULL.norm()).to(DEV)[None].expand(N, spec(TASK).dstb_dim).contiguous()
    obs = env.reset()

    def set_theta(th_):
        inner.sim.model.actuator_forcerange[:, ids, :] = nominal * float(th_)
        inner._fr_torque_frac[:] = float(th_)
    set_theta(theta)
    standing_t = th.zeros(N, device=DEV)
    tipped = th.zeros(N, dtype=th.bool, device=DEV)
    slam = th.zeros(N, dtype=th.bool, device=DEV)
    for t in range(STEPS):
        set_theta(theta)
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
                              ambient=AMBIENT, gust_scale=GUST, slam_n=SLAM_N), "matrix": {}}
    txt = ["E076 PART B.1 — per-θ behavioral matrix (leg-ODD, 10N ambient + 25N gust @ t=3s, N=128 x 300 steps)",
           "cols: stand%=(z>0.18 & tilt<0.3) frac | tip=fell_over | slam=nonfoot>80N | h_end", ""]
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
    # highlight the θ=0 leg_rest cell (the E061-era all-collapse extreme)
    z = results["matrix"]["leg_rest"][0.0]
    hl = (f"HIGHLIGHT — leg_rest @ θ=0 (FULLY DEAD FR leg): stand%={z['stand']:.2f} tip={z['tip']:.2f} "
          f"slam={z['slam']:.2f} h_end={z['h_end']:.2f}  (E061-era stance objectives ALL-COLLAPSED here)")
    print("\n" + hl); txt.append(hl)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "partB_matrix.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(OUT, "partB_matrix.txt"), "w") as f:
        f.write("\n".join(txt))
    print(f"\nwrote {OUT}/partB_matrix.json + .txt")
