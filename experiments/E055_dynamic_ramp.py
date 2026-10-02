"""E055 — DYNAMIC mid-episode leg degradation (the COMMITMENT-WINDOW test; the "gradual 4→3 leg" experiment).

Every eval so far applied the disturbance from t=0, giving the policy the whole episode to feel the weak leg and
adapt — maximally favourable to reactive/blind control, so it could NOT show conditioning's value. This one
degrades the FR leg SUDDENLY MID-EPISODE while the robot is already resisting a pull: θ (FR allowable torque) is
held at 1.0, then dropped to θ_lo at t_switch. A policy briefly COMMITTED to a leg that just weakened is exactly
the regime where knowing θ (conditioned) or inferring it fast (history) could beat blind — IF a commitment window
exists on this testbed.

Mechanism (the runtime-override analog of the payload E021 live `jnt_stiffness` write): each step we write the FR
actuators' `actuator_forcerange = nominal * θ_t` directly on the model, and set `_fr_torque_frac = θ_t` so the
conditioned policy sees the LIVE θ. θ_t follows a step or ramp schedule.

Read-out: survival(t) aligned to the switch. If a COMMITMENT WINDOW exists, blind (and the θ=1 normal specialist)
drop at the switch while conditioned/history hold → conditioning wins dynamically. If all drop together (or none),
the leg ODD is absorbed reactively even under a sudden change → same negative conclusion as the static evals.
"""
import os, sys, io, contextlib, math, json
import torch as th
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "external/robot-safety-sandbox")
from robot_safety_sandbox import make_tensor, spec
from robot_safety_sandbox.eval.policies import load_twin

DEV, N, DT = "cuda:0", 256, 0.02
STEPS, T_SWITCH = 250, 100                 # 5 s episode; leg degrades at t=2 s
THETA_HI, THETA_LO = 1.0, 0.2
PULL = th.tensor([0., 1., 0.])             # constant lateral pull throughout
CK = "results/go2_weak_leg_runs/{run}/checkpoints/model_49999872_steps.zip"
# policy -> (eval task giving its obs surface, run dir). All eval tasks carry cache_fr_nominal (per-world
# forcerange + nominal snapshot) so the per-step override works; the reset randomization is overwritten each step.
POLICIES = {
    "normal-spec(θ1)": ("go2_weak_leg_blind",       "go2_stabilize_adv"),      # 47-dim; θ=1-only specialist
    "blind":           ("go2_weak_leg_blind",       "go2_weak_leg_blind_adv"),
    "conditioned":     ("go2_weak_leg_conditioned", "go2_weak_leg_conditioned_adv"),
    "history":         ("go2_weak_leg_history",     "go2_weak_leg_history_adv"),
}


def theta_at(t, ramp_steps):
    if t < T_SWITCH:
        return THETA_HI
    f = min(1.0, (t - T_SWITCH) / max(1, ramp_steps))
    return THETA_HI + (THETA_LO - THETA_HI) * f


def set_theta(inner, ids, nominal, theta):
    inner.sim.model.actuator_forcerange[:, ids, :] = nominal * float(theta)
    inner._fr_torque_frac[:] = float(theta)


def run(task, ck, fr, ramp_steps):
    with contextlib.redirect_stdout(io.StringIO()):
        env = make_tensor(task, N, DEV, adversary=True)
        model, norm = load_twin(ck, DEV, quiet=True)
    inner = env.mj
    dd = spec(task).dstb_dim
    env.force_scale = fr * th.ones(N, device=DEV)
    dstb = (PULL / PULL.norm()).to(DEV)[None].expand(N, dd).contiguous()
    obs = env.reset()
    # ensure the FR cache exists, then pin θ=HI before the first action
    from robot_safety_sandbox.envs.go2_broken_leg.env_cfg import _ensure_fr_cache
    _ensure_fr_cache(inner)
    ids, nominal = inner._fr_act_ids, inner._fr_nominal_forcerange
    set_theta(inner, ids, nominal, THETA_HI)
    alive = th.ones(N, dtype=th.bool, device=DEV)
    surv = []
    for t in range(STEPS):
        set_theta(inner, ids, nominal, theta_at(t, ramp_steps))   # live θ for THIS physics step
        with th.no_grad():
            a = th.clamp(model.policy._predict(norm(obs), deterministic=True), -1, 1)
        obs, g, dones, touts, l = env.step_tensor(th.cat([a, dstb], dim=1))
        alive &= ~(dones & ~touts)                                 # once fallen, stays fallen
        surv.append(float(alive.float().mean()))
    env.close()
    return surv


if __name__ == "__main__":
    FR = 0.3            # 15 N pull — hard enough to stress the switch, soft enough to see a transient
    RAMP = {"step": 1, "ramp0.5s": 25}
    twins = {name: load_twin(CK.format(run=run_dir), DEV, quiet=True) for name, (_t, run_dir) in POLICIES.items()}
    print(f"E055 dynamic ramp: θ {THETA_HI}→{THETA_LO} at t={T_SWITCH*DT:.1f}s, pull {int(FR*50)}N, N={N}")
    out = {}
    for shape, rs in RAMP.items():
        print(f"\n=== ramp shape: {shape} ===")
        for name, (task, run_dir) in POLICIES.items():
            model, norm = twins[name]
            # rebuild-per-run via run(): reload fresh env each time
            surv = run(task, CK.format(run=run_dir), FR, rs)
            out[f"{shape}|{name}"] = surv
            pre = surv[T_SWITCH - 1]
            trans = surv[min(STEPS - 1, T_SWITCH + 25)]
            end = surv[-1]
            print(f"  {name:16s} surv@switch={pre:.2f}  +0.5s={trans:.2f}  end={end:.2f}  drop={pre-trans:+.2f}")
    od = os.path.expanduser("~/artifacts/odd-conditioned/E055-dynamic-ramp")
    os.makedirs(od, exist_ok=True)
    json.dump({"steps": STEPS, "t_switch": T_SWITCH, "dt": DT, "theta_hi": THETA_HI, "theta_lo": THETA_LO,
               "force_N": FR * 50, "surv": out}, open(f"{od}/results.json", "w"), indent=2)
    print(f"\nsaved -> {od}/results.json")
