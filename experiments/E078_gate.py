"""E078 (T004) — COMPOUND-ODD PHYSICS GATE: FR-leg death WHILE carrying a high-CoM load.

Pre-registered kill gate (see experiments.md E078). Fixed carried load W=80 N at h=0.25 (inverted pendulum),
sweep the FR-leg allowable-torque fraction θ. Two probes, both on the permissive common surface
``go2_weight_rest_hi_at_0`` (contact term 500 N, fell_over active, weight_theta obs present). The load is
driven per step (base_load force + _weight_W obs + _weight_h pendulum lever, exactly as E074_matrix) AND the
FR leg is degraded per step (actuator_forcerange = nominal*θ + _fr_torque_frac, exactly as E076_matrix).

(a) STAND probe: the E075 RECAL weight-stand policy (knows W via obs 48's weight_theta=W/150≈0.53; has NO θ
    input — the leg death is unobserved physics). 10 N ambient pull + a 25 N gust (→35 N) at t=3 s.
    Metrics per θ: stand%(z>0.18 & tilt<0.3) | tip=fell_over | slam=nonfoot>SLAM_CAP(80)=184 N | h_end.
(b) DESCENT probe: scripted DEEP fold (E069b) at the same (W=80, θ). Metrics: v_touch (|vz| at first belly
    contact) | tilt at first contact | settled static force (late-window nonfoot force — informs the rest cap).

GATE: STAND fails (tip+collapse) below some θ_c ≥ ~0.2 (a real contraction boundary ABOVE zero — unlike the
UNLOADED leg-at-stance case E076 where stand held to θ=0.1) AND descent stays gentle there  → PASS, build.
If the LOADED stand STILL holds to θ≤0.1 → KILL (the compound axis is reactively absorbable too).
"""
import os, sys, io, contextlib, json
from _paths import _ART
import torch as th
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "external/robot-safety-sandbox")
import numpy as np
from robot_safety_sandbox import make_tensor, spec
from robot_safety_sandbox.eval.policies import load_twin
from robot_safety_sandbox.envs.go2_broken_leg.env_cfg import _ensure_fr_cache

DEV, N, DT, STEPS = "cuda:0", 128, 0.02, 300
W, LOAD_H = 80.0, 0.25
SLAM_CAP = 80.0 + 1.3 * W                          # load-conditioned no-slam cap at W=80 → 184 N
THETAS = [1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
T_GUST0, T_GUST1 = 150, 175                        # gust window t∈[3.0,3.5]s
AMBIENT, GUST = 0.20, 0.70                          # 10 N ambient pull / 35 N (10+25 gust); force_max=50 N
PULL = th.tensor([0., 1., 0.])
TASK = "go2_weight_rest_hi_at_0"                    # permissive common cfg; W + θ driven per step
STAND_CK = "results/go2_weight_runs/E075_recal/go2_weight_stand_hi_adv/checkpoints/model_49999872_steps.zip"
OUT = os.path.expanduser(_ART + "/E078-compound-gate")

# scripted DEEP fold (E069b) in action space: [hips(4), thighs(4), calves(4)] offsets from default pose
FOLD_HIP, FOLD_TH, FOLD_CA = 0.0, 0.35, -0.45
T0_DESC, RAMP = 30, 50                              # descent: load+fold start @0.6s, ramp in over 1s


def _drivers(inner):
    """Return a closure that (re)asserts W=80 @ h=0.25 + FR degraded to θ every step (env auto-reset zeroes
    _weight_W/_weight_h and the STAND policy's obs; forcerange survives reset but re-applied for safety)."""
    _ensure_fr_cache(inner)
    ids, nominal = inner._fr_act_ids, inner._fr_nominal_forcerange

    def drive(env, theta):
        inner._weight_h = th.full((N,), LOAD_H, device=DEV)
        inner._weight_W = th.full((N,), W, device=DEV)
        env.base_load = th.tensor([0., 0., -W], device=DEV)[None].expand(N, 3).contiguous()
        inner.sim.model.actuator_forcerange[:, ids, :] = nominal * float(theta)
        inner._fr_torque_frac[:] = float(theta)
    return drive


def stand_probe(theta):
    with contextlib.redirect_stdout(io.StringIO()):
        env = make_tensor(TASK, N, DEV, adversary=True)
        model, norm = load_twin(STAND_CK, DEV, quiet=True)
    inner = env.mj
    drive = _drivers(inner)
    dstb = PULL.to(DEV)[None].expand(N, spec(TASK).dstb_dim).contiguous()
    obs = env.reset()
    standing_t = th.zeros(N, device=DEV)
    tipped = th.zeros(N, dtype=th.bool, device=DEV)
    slam = th.zeros(N, dtype=th.bool, device=DEV)
    for t in range(STEPS):
        drive(env, theta)
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
        slam |= th.norm(fh, dim=-1).flatten(1).amax(1) > SLAM_CAP
        tipped |= inner.termination_manager._term_dones.get("fell_over", th.zeros(N, device=DEV)).bool()
    h_end = float(inner.scene["robot"].data.root_link_pos_w[:, 2].mean())
    env.close()
    return dict(stand=float(standing_t.mean()) / STEPS, tip=float(tipped.float().mean()),
                slam=float(slam.float().mean()), h_end=h_end)


def descent_probe(theta):
    with contextlib.redirect_stdout(io.StringIO()):
        env = make_tensor(TASK, N, DEV, adversary=True)
    inner = env.mj
    drive = _drivers(inner)
    dstb = PULL.to(DEV)[None].expand(N, spec(TASK).dstb_dim).contiguous()
    env.reset()
    target = th.zeros(N, 12, device=DEV)
    target[:, 0:4] = FOLD_HIP; target[:, 4:8] = FOLD_TH; target[:, 8:12] = FOLD_CA
    touched = th.zeros(N, dtype=th.bool, device=DEV)
    v_touch = th.full((N,), float("nan"), device=DEV)
    tilt_touch = th.full((N,), float("nan"), device=DEV)
    static_acc = []
    for t in range(STEPS):
        drive(env, theta)
        env.force_scale = AMBIENT * th.ones(N, device=DEV)          # 10 N pull, no gust for the descent probe
        frac = min(1.0, max(0.0, (t - T0_DESC) / RAMP))
        a = target * frac
        _o, g, dones, touts, l = env.step_tensor(th.cat([a, dstb], dim=1))
        s = inner.scene["nonfoot_ground_touch"]
        fh = s.data.force_history if s.data.force_history is not None else s.data.force
        force = th.norm(fh, dim=-1).flatten(1).amax(1)
        d = inner.scene["robot"].data
        vz = d.root_link_lin_vel_w[:, 2]
        pg = d.projected_gravity_b
        tilt = th.maximum(pg[:, 0].abs(), pg[:, 1].abs())
        fresh = (force > 5.0) & ~touched
        v_touch = th.where(fresh, vz.abs(), v_touch)
        tilt_touch = th.where(fresh, tilt, tilt_touch)
        touched |= fresh
        if t >= STEPS - 50:
            static_acc.append(float(force.mean()))                  # late-window ~ settled static rest force
    tv = v_touch[touched]; tt = tilt_touch[touched]
    static = float(np.mean(static_acc)) if static_acc else float("nan")
    env.close()
    return dict(touch_frac=float(touched.float().mean()),
                v_med=float(tv.median()) if len(tv) else float("nan"),
                v_p90=float(tv.quantile(0.9)) if len(tv) else float("nan"),
                tilt_med=float(tt.median()) if len(tt) else float("nan"),
                static=static)


if __name__ == "__main__":
    cfg = dict(N=N, STEPS=STEPS, W=W, load_h=LOAD_H, slam_cap=SLAM_CAP, thetas=THETAS,
               gust=(T_GUST0, T_GUST1), ambient=AMBIENT, gust_scale=GUST, fold=(FOLD_HIP, FOLD_TH, FOLD_CA))
    res = {"config": cfg, "stand": {}, "descent": {}}
    txt = [f"E078 COMPOUND-ODD GATE — W={W:.0f}N @ h={LOAD_H}, FR-torque θ sweep, N={N} x {STEPS} steps",
           f"SLAM_CAP(W=80)={SLAM_CAP:.0f}N. STAND: E075 recal weight-stand policy, 10N ambient + 25N gust @t=3s.", ""]

    hdr = "=== (a) STAND probe (E075 recal, loaded) ==="
    cols = f"{'θ':>5} {'stand%':>7} {'tip':>5} {'slam':>5} {'h_end':>6}"
    print("\n" + hdr); print(cols); txt += [hdr, cols]
    for theta in THETAS:
        r = stand_probe(theta); res["stand"][theta] = r
        line = f"{theta:>5.1f} {r['stand']:>7.2f} {r['tip']:>5.2f} {r['slam']:>5.2f} {r['h_end']:>6.2f}"
        print(line); txt.append(line)

    hdr = "\n=== (b) DESCENT probe (scripted deep fold, loaded) ==="
    cols = (f"{'θ':>5} {'touched':>7} {'v_med':>6} {'v_p90':>6} {'tilt':>5} {'static':>7}"
            "   (gentle: v<~0.3 m/s, tilt<~0.5; static vs cap=184)")
    print(hdr); print(cols); txt += [hdr, cols]
    for theta in THETAS:
        r = descent_probe(theta); res["descent"][theta] = r
        line = (f"{theta:>5.1f} {r['touch_frac']:>7.2f} {r['v_med']:>6.2f} {r['v_p90']:>6.2f} "
                f"{r['tilt_med']:>5.2f} {r['static']:>7.0f}")
        print(line); txt.append(line)

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "gate.json"), "w") as f:
        json.dump(res, f, indent=2)
    with open(os.path.join(OUT, "gate.txt"), "w") as f:
        f.write("\n".join(txt))
    print(f"\nwrote {OUT}/gate.json + .txt")
    print("\nGATE: PASS iff STAND collapses (tip↑ / stand%↓) below some θ_c≥~0.2 AND descent stays gentle there.")
    print("      KILL iff loaded STAND still holds to θ≤0.1 (compound axis reactively absorbable).")
