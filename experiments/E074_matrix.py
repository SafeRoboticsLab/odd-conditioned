"""E074 (T003) Task 1 — per-W BEHAVIORAL MATRIX for all 4 HIGH-CoM arms, under gusts.

Each policy ∈ {stand_hi, rest_hi, unified_hi, unified_disc_hi} × W ∈ {0,30,60,90,120,150,200}, 300 steps,
on the permissive REST-hi cfg (contact term raised, fell_over active — cross-policy common surface, like
E070_matrix). Load driven per step (base_load = physics force + torque magnitude, _weight_h = HIGH-CoM lever,
_weight_W = obs conditioning). Ambient 10N lateral pull PLUS a 25N GUST pulse (0.5 s at t=3 s) = the E073
probe's gust (total 35N over steps 150-175).

Per cell:
  stand% = time-fraction (base_z>0.18 & tilt<0.3)             [affordance / stand-spec]
  tip    = fraction fell_over                                 [both specs' failure]
  slam   = fraction nonfoot force > SLAM_CAP(W)=80+1.3W       [rest-spec failure]
  h_end  = mean final base height

Reads: (a) stand_hi's TRUE feasible band under gusts; (b) does unified_hi ALWAYS descend (stand%~0 at all W =
the lexicographic finding); (c) unified_disc_hi's hedge profile (stands where? tips doing so?).
"""
import os, sys, io, contextlib, json
import torch as th
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "external/robot-safety-sandbox")
from robot_safety_sandbox import make_tensor, spec
from robot_safety_sandbox.eval.policies import load_twin

DEV, N, STEPS = "cuda:0", 128, 300
WS = [0, 30, 60, 90, 120, 150, 200]
POLICIES = ["stand_hi", "rest_hi", "unified_hi", "unified_disc_hi"]
LOAD_H = 0.25
T_GUST0, T_GUST1 = 150, 175                         # gust window t∈[3.0,3.5]s
AMBIENT, GUST = 0.20, 0.70                          # 10N ambient / 35N (10+25 gust)
PULL = th.tensor([0., 1., 0.])
CK = "results/go2_weight_runs/go2_weight_{m}_adv/checkpoints/model_49999872_steps.zip"
TASK = "go2_weight_rest_hi_at_0"                    # permissive common cfg; W driven per step
OUT = os.path.expanduser("~/artifacts/odd-conditioned/E074-hicom-demo")


def cell(policy_m, W):
    with contextlib.redirect_stdout(io.StringIO()):
        env = make_tensor(TASK, N, DEV, adversary=True)
        model, norm = load_twin(CK.format(m=policy_m), DEV, quiet=True)
    inner = env.mj
    dstb = PULL.to(DEV)[None].expand(N, spec(TASK).dstb_dim).contiguous()
    obs = env.reset()
    standing_t = th.zeros(N, device=DEV)
    tipped = th.zeros(N, dtype=th.bool, device=DEV)
    slam = th.zeros(N, dtype=th.bool, device=DEV)
    cap = 80.0 + 1.3 * W
    for t in range(STEPS):
        inner._weight_h = th.full((N,), LOAD_H, device=DEV)               # HIGH-CoM lever (reset zeroes it)
        inner._weight_W = th.full((N,), float(W), device=DEV)             # obs conditioning
        env.base_load = th.tensor([0., 0., -float(W)], device=DEV)[None].expand(N, 3).contiguous()
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
        slam |= th.norm(fh, dim=-1).flatten(1).amax(1) > cap
        tipped |= inner.termination_manager._term_dones.get(
            "fell_over", th.zeros(N, device=DEV)).bool()
    h_end = float(inner.scene["robot"].data.root_link_pos_w[:, 2].mean())
    env.close()
    return dict(stand=float(standing_t.mean()) / STEPS, tip=float(tipped.float().mean()),
                slam=float(slam.float().mean()), h_end=h_end)


if __name__ == "__main__":
    results = {"config": dict(N=N, STEPS=STEPS, WS=WS, gust=(T_GUST0, T_GUST1),
                              ambient=AMBIENT, gust_scale=GUST, load_h=LOAD_H), "matrix": {}}
    txt = ["E074 TASK 1 — per-W behavioral matrix (HIGH-CoM, 10N ambient + 25N gust @ t=3s, N=128 x 300 steps)",
           "cols: stand%=(z>0.18 & tilt<0.3) frac | tip=fell_over | slam=nonfoot>80+1.3W | h_end", ""]
    for m in POLICIES:
        results["matrix"][m] = {}
        hdr = f"=== policy: go2_weight_{m} ==="
        cols = f"{'W(N)':>5} {'stand%':>7} {'tip':>5} {'slam':>5} {'h_end':>6}"
        print("\n" + hdr); print(cols)
        txt.append(hdr); txt.append(cols)
        for W in WS:
            r = cell(m, W)
            results["matrix"][m][W] = r
            line = f"{W:>5} {r['stand']:>7.2f} {r['tip']:>5.2f} {r['slam']:>5.2f} {r['h_end']:>6.2f}"
            print(line); txt.append(line)
        txt.append("")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "task1_matrix.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(OUT, "task1_matrix.txt"), "w") as f:
        f.write("\n".join(txt))
    print(f"\nwrote {OUT}/task1_matrix.json + .txt")
