"""E063 — break the FL (front-LEFT) leg instead of FR, on the FR-trained policies. The policies were trained with
the FR leg degrading and condition on a SINGLE scalar θ = "the monitored leg's torque". What happens when a
DIFFERENT leg (FL) fails?

We break a chosen leg mid-episode (θ 1.0→0.2 @ t=2s, per-step forcerange write) under a 15N pull, soft-rest scoring,
converged 50M policies. leg=FR reproduces E060 (sanity). leg=FL is the probe. For the conditioned arm we test BOTH:
  cond-told   : θ input = the live degradation (the policy is told "a monitored leg is at θ" — but it learned FR,
                so for an FL break this MIS-ATTRIBUTES the failure to FR)
  cond-unaware: θ input = 1.0 (the FR-monitoring belief sees FR fine; the FL failure is OUTSIDE the modeled ODD)
Compares survival/slam per (leg, arm) — does the FR-learned compensation transfer to FL, or is it leg-specific?
"""
import os, sys, io, contextlib, mujoco
import torch as th
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "external/robot-safety-sandbox")
from robot_safety_sandbox import make_tensor, spec
from robot_safety_sandbox.eval.policies import load_twin
from robot_safety_sandbox.envs.go2_broken_leg.env_cfg import _ensure_fr_cache

DEV, N, DT = "cuda:0", 256, 0.02
STEPS, T_SWITCH, HI, LO = 250, 100, 1.0, 0.2
FR_SCALE = 0.3                                  # 15 N pull
PULL = th.tensor([0., 1., 0.])
CK = "results/go2_weak_leg_runs/{run}/checkpoints/model_49999872_steps.zip"
RUN = {"blind": "go2_weak_leg_blind_soft_adv", "history": "go2_weak_leg_history_soft_adv",
       "conditioned": "go2_weak_leg_conditioned_soft_adv"}
TASK = {"blind": "go2_weak_leg_blind_soft", "history": "go2_weak_leg_history_soft",
        "conditioned": "go2_weak_leg_conditioned_soft"}


def leg_actuator_ids(m, leg):
    names = [f"robot/{leg}_{j}_joint" for j in ("hip", "thigh", "calf")]
    ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in names]
    assert all(i >= 0 for i in ids), (leg, ids)
    return th.tensor(ids, device=DEV, dtype=th.long)


def run(arm, leg, theta_mode):
    task, run_dir = TASK[arm], RUN[arm]
    with contextlib.redirect_stdout(io.StringIO()):
        env = make_tensor(task, N, DEV, adversary=True)
        model, norm = load_twin(CK.format(run=run_dir), DEV, quiet=True)
    inner = env.mj; _ensure_fr_cache(inner)
    m = inner.sim.mj_model
    fr_ids, fr_nom = inner._fr_act_ids, inner._fr_nominal_forcerange   # TRUE FR nominal (pre-randomization)
    env.force_scale = FR_SCALE * th.ones(N, device=DEV)
    dstb = (PULL / PULL.norm()).to(DEV)[None].expand(N, spec(task).dstb_dim).contiguous()
    obs = env.reset()
    if leg == "FR":
        brk_ids, brk_nom = fr_ids, fr_nom
    else:
        brk_ids = leg_actuator_ids(m, leg)
        brk_nom = inner.sim.model.actuator_forcerange[:, brk_ids, :].clone()  # FL not randomized by reset -> nominal
    alive = th.ones(N, dtype=th.bool, device=DEV); slam = th.zeros(N, dtype=th.bool, device=DEV)
    for t in range(STEPS):
        theta = HI if t < T_SWITCH else LO
        # keep FR HEALTHY (override the reset randomization) unless FR is the leg we're breaking
        inner.sim.model.actuator_forcerange[:, fr_ids, :] = fr_nom
        inner.sim.model.actuator_forcerange[:, brk_ids, :] = brk_nom * float(theta)   # break the CHOSEN leg
        # conditioning signal the policy reads (env._fr_torque_frac = "FR torque" belief):
        inner._fr_torque_frac[:] = float(theta) if theta_mode == "told" else 1.0
        with th.no_grad():
            a = th.clamp(model.policy._predict(norm(obs), deterministic=True), -1, 1)
        obs, g, dones, touts, l = env.step_tensor(th.cat([a, dstb], dim=1))
        ic = inner.termination_manager._term_dones.get("illegal_contact")
        nf = dones & ~touts & alive
        if ic is not None: slam |= nf & ic
        alive &= ~(dones & ~touts)
    env.close()
    return float(alive.float().mean()), float(slam.float().mean())


if __name__ == "__main__":
    print(f"{'condition':28s} {'survival':>9} {'slam':>6}")
    rows = [("blind", "FR", "told"), ("history", "FR", "told"), ("conditioned", "FR", "told"),
            ("blind", "FL", "told"), ("history", "FL", "told"),
            ("conditioned", "FL", "told"), ("conditioned", "FL", "unaware")]
    for arm, leg, mode in rows:
        s, sl = run(arm, leg, mode)
        label = f"{leg} break | {arm}" + ("" if arm != "conditioned" else f" ({mode})")
        print(f"{label:28s} {s:9.2f} {sl:6.2f}")
