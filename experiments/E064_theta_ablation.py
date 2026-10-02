"""E064 θ-ablation — is the runtime θ INPUT load-bearing for the conditioned POLICY, or vestigial?
Take ONE conditioned network and feed it different θ inputs (physics = TRUE θ throughout). If survival is
invariant to the θ input, the benefit of conditioning is a TRAINING effect (privileged info → better net,
RMA-phase-1 style), NOT a runtime-information effect — which would mean the deploy-time belief B̂ is not needed
for the POLICY (only, at most, for the certificate/VALUE). This is the crux for the contribution.

Two policies: DYNAMIC-trained conditioned (E064 s0) and STATIC-trained conditioned (E058). θ-input modes on the
leg-death ramp (θ_phys 1.0->0.2 @ t=2s): oracle (=true), frozen1.0 (wrong after death), frozen0.2 (wrong before),
inverted (1.0 after / 0.2 before — maximally wrong). Also a FIXED-θ=0.2 episode with input 0.2 vs 1.0.
"""
import os, sys, io, contextlib
import torch as th
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "external/robot-safety-sandbox")
from robot_safety_sandbox import make_tensor, spec
from robot_safety_sandbox.eval.policies import load_twin
from robot_safety_sandbox.envs.go2_broken_leg.env_cfg import _ensure_fr_cache

DEV, N, DT = "cuda:0", 128, 0.02
TASK = "go2_weak_leg_conditioned_soft"
PULL = th.tensor([0., 1., 0.])
POLICIES = {
    "dyn-cond(s0)":  "results/go2_weak_leg_runs/E064_s0/go2_weak_leg_conditioned_soft_dyn_adv/checkpoints/model_49999872_steps.zip",
    "static-cond":   "results/go2_weak_leg_runs/go2_weak_leg_conditioned_soft_adv/checkpoints/model_49999872_steps.zip",
}


def ramp(ck, fr, mode, tsw=100, steps=250):
    with contextlib.redirect_stdout(io.StringIO()):
        env = make_tensor(TASK, N, DEV, adversary=True); model, norm = load_twin(ck, DEV, quiet=True)
    inner = env.mj; _ensure_fr_cache(inner); ids, nom = inner._fr_act_ids, inner._fr_nominal_forcerange
    env.force_scale = fr * th.ones(N, device=DEV)
    dstb = (PULL / PULL.norm()).to(DEV)[None].expand(N, spec(TASK).dstb_dim).contiguous()
    obs = env.reset(); alive = th.ones(N, dtype=th.bool, device=DEV)
    for t in range(steps):
        theta_phys = 1.0 if t < tsw else 0.2                       # PHYSICS always the true leg death
        inner.sim.model.actuator_forcerange[:, ids, :] = nom * theta_phys
        if mode == "oracle":     theta_in = theta_phys
        elif mode == "frozen1":  theta_in = 1.0
        elif mode == "frozen0.2":theta_in = 0.2
        elif mode == "inverted": theta_in = 0.2 if t < tsw else 1.0   # maximally wrong
        inner._fr_torque_frac[:] = float(theta_in)
        with th.no_grad():
            a = th.clamp(model.policy._predict(norm(obs), deterministic=True), -1, 1)
        obs, g, dones, touts, l = env.step_tensor(th.cat([a, dstb], dim=1))
        alive &= ~(dones & ~touts)
    env.close()
    return float(alive.float().mean())


if __name__ == "__main__":
    MODES = ["oracle", "frozen1", "frozen0.2", "inverted"]
    for name, ck in POLICIES.items():
        if not os.path.exists(ck):
            print(f"{name}: checkpoint missing, skip"); continue
        print(f"\n=== {name} — survival on leg-death ramp, θ-INPUT modes (physics=true θ) ===")
        print(f"{'force':>6} | " + " | ".join(f"{m:>10}" for m in MODES))
        for fr in (0.3, 0.5):
            row = [f"{ramp(ck, fr, m):.2f}" for m in MODES]
            print(f"{int(fr*50):>4}N | " + " | ".join(f"{c:>10}" for c in row))
    print("\nIf oracle≈frozen≈inverted -> θ INPUT is VESTIGIAL (benefit was privileged TRAINING, not runtime info).")
    print("If inverted << oracle -> the runtime θ signal IS load-bearing (belief B̂ matters for the policy).")
