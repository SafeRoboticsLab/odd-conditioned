"""E070 swap matrix — both mode policies × fixed W, common physics, dual scoring (each mode's spec).
Eval on the REST task cfg (contact termination at 500N ⇒ no premature end on belly contact; fell_over active),
10N lateral pull, W pinned per column. Per cell:
  stand%  = time-fraction genuinely standing (base_z>0.18 & tilt<0.3)     [affordance / stand-spec]
  tip     = fraction fell_over                                            [both specs' failure]
  slam    = fraction with nonfoot contact > SLAM_CAP(W)=80+1.3W           [rest-spec failure]
  h_end   = mean final base height
Expected spec-family signature: stand-policy = high stand% at low W, tips/collapses at high W (its spec is
INFEASIBLE there); rest-policy = ~0 stand% (it descends) but SAFE at every W. No single policy row is both
safe everywhere AND standing where possible — that is the fixed-ODD filter's dilemma the handoff resolves.
"""
import os, sys, io, contextlib
import torch as th
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "external/robot-safety-sandbox")
from robot_safety_sandbox import make_tensor, spec
from robot_safety_sandbox.eval.policies import load_twin

DEV, N, STEPS = "cuda:0", 128, 300
WS = [0, 60, 90, 120, 150, 200, 250]
PULL = th.tensor([0., 1., 0.])
CK = "results/go2_weight_runs/go2_weight_{m}_adv/checkpoints/model_49999872_steps.zip"


def cell(policy_m, W, FR=0.2):
    task = f"go2_weight_rest_at_{W}"                      # rest cfg = permissive terminations, W pinned
    with contextlib.redirect_stdout(io.StringIO()):
        env = make_tensor(task, N, DEV, adversary=True)
        model, norm = load_twin(CK.format(m=policy_m), DEV, quiet=True)
    inner = env.mj
    env.force_scale = FR * th.ones(N, device=DEV)
    dstb = (PULL).to(DEV)[None].expand(N, spec(task).dstb_dim).contiguous()
    obs = env.reset()
    standing_t = th.zeros(N, device=DEV); tipped = th.zeros(N, dtype=th.bool, device=DEV)
    slam = th.zeros(N, dtype=th.bool, device=DEV); cap = 80.0 + 1.3 * W
    for t in range(STEPS):
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
        z = th.zeros(N, device=DEV)
        tipped |= inner.termination_manager._term_dones.get("fell_over", z).bool()
    h_end = float(inner.scene["robot"].data.root_link_pos_w[:, 2].mean())
    env.close()
    return (float(standing_t.mean()) / STEPS, float(tipped.float().mean()),
            float(slam.float().mean()), h_end)


if __name__ == "__main__":
    import sys as _s
    FR = float(_s.argv[1]) if len(_s.argv) > 1 else 0.2
    for m in ("stand", "rest"):
        print(f"\n=== policy: go2_weight_{m} @ pull {FR*50:.0f}N ===")
        print(f"{'W(N)':>5} {'stand%':>7} {'tip':>5} {'slam':>5} {'h_end':>6}")
        for W in WS:
            s, t, sl, h = cell(m, W, FR)
            print(f"{W:>5} {s:>7.2f} {t:>5.2f} {sl:>5.2f} {h:>6.2f}")
