"""E091 — LEG DEGRADATION AS A SAFETY-FILTERING PROBLEM (redesigned walking experiment).

Scenario ("thermal"): the robot walks toward a goal 11 m ahead; at t=5 s one leg's motors derate
(θ: 1.0 -> 0.15 linearly over 3 s — thermal protection), stay dead until t=15 s, then recover (cooled).
Horizon 30 s. The goal is NOT reachable before the fault (≤ ~6 m) — ONLY strategies that survive the fault
AND resume walking can succeed. Conditions: legonly (W=0) and compound (constant 60 N high-CoM payload).

Detection (the honest channel): V̄_stand is BLIND to actuator faults (E091 probe: hobbling robots read
V̄=+0.07) — unmodeled θ changes don't manifest in the value's modeled state. The descent trigger is therefore
a BELIEF residual: per-leg joint tracking error (|q - q_target| with q_target = default + 0.75·a), the
E018 set-membership idea made model-free. V̄<-0.10 remains as an OR-term.

Arms (all switching arms share the SAME descent trigger; they differ ONLY in the return —
the discriminator this experiment is built around):
  WALK-ONLY  : nominal walker, no filter (structural failure case)
  REST-ONLY  : safety expert (safe rate anchor; success 0)
  ONE-WAY    : descend on trigger, never return
  V1         : descend on trigger; BLIND return (prone V̄_stand>0.15 — no θ̂ access; fires mid-fault)
  V2-REUSE   : descend on trigger; CERTIFIED return (θ̂ healthy sustained + V̄_up gate + getup funnel + abort)
Deaths = env spec (fell_over | illegal_contact). N=256, no respawn, timeout disabled.
Modes: --cal (residual detector calibration), main (stats+figures), --videos.
"""
import os, sys, io, math, json, contextlib, argparse
import torch as th
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "external/robot-safety-sandbox"); sys.path.insert(0, "experiments")
sys.path.insert(0, "/home/buzi/Desktop/RESEARCH/SAFE/DEVELOPMENT/go2_atomic_skills")
import numpy as np
from robot_safety_sandbox import make_tensor, spec
from robot_safety_sandbox.eval.policies import load_twin
from robot_safety_sandbox.envs.go2_broken_leg.env_cfg import _ensure_fr_cache
from E084_automaton import (CK, value_of, in_rest_target, in_stance_target,
                            EPS_UP, K_UP, EPS_ABORT, K_ABORT, REFRACT, ALPHA, K_DN, WARMUP)
import E089_goal_walk as G

DEV, N, DT, STEPS = "cuda:0", 256, 0.02, 1500
GOAL_D, GOAL_R = 11.0, 1.0
T_FAULT0, T_FAULT1, T_HEAL = 5.0, 8.0, 15.0
THETA_LO = 0.15
EPS_DN_WALK = -0.10
ERR_DEG, K_DEG = 0.03, 10       # higher bar + 0.2s sustain: rejects load bursts WITHOUT the 1s latency that made braking lethal (trig 8.2s = post-ramp, 139 brake deaths)
HEAL_SUST = 25                  # θ̂ healthy sustained (0.5 s) before return gate opens
EPS_UP_LEG, EPS_ABORT_LEG = -0.35, -0.45   # V_up recalibrated on THIS regime's rest poses (W=0 post-brake:
                                           # population median -0.27; certificates are ordinal off-distribution,
                                           # thresholds are per-regime calibration — belief gate carries the ODD decision
W_COMPOUND = 30.0               # heavy enough to matter, light enough that a HEALTHY loaded walker is viable
STATES = ["WALK", "DESCENDING", "REST", "GETTINGUP", "BRAKE"]


def theta_of(t):
    s = t * DT
    if s < T_FAULT0: return 1.0
    if s < T_FAULT1: return 1.0 + (THETA_LO - 1.0) * (s - T_FAULT0) / (T_FAULT1 - T_FAULT0)
    if s < T_HEAL: return THETA_LO
    return 1.0


def make_residual(inner):
    """Torque-saturation residual: demanded PD torque vs achieved motor torque, per leg.
    Demand f = kp*(ctrl - q) - kd*qd (the robot knows its own PD command); achieved = motor current sensing
    (sim: post-clamp actuator_force). A derated motor shows a persistent positive gap. Model-free re theta."""
    import mujoco as _mj
    mm = inner.sim.mj_model
    jn = list(inner.scene["robot"].joint_names)
    amap = [jn.index(_mj.mj_id2name(mm, _mj.mjtObj.mjOBJ_JOINT, mm.actuator_trnid[a, 0]).split("/")[-1])
            for a in range(mm.nu)]
    amap_t = th.tensor(amap, device=DEV, dtype=th.long)
    kp = th.tensor(mm.actuator_gainprm[:, 0], device=DEV, dtype=th.float32)
    kd = th.tensor(-mm.actuator_biasprm[:, 2], device=DEV, dtype=th.float32)
    fnom = th.tensor(np.abs(mm.actuator_forcerange[:, 1]), device=DEV, dtype=th.float32).clamp_min(1.0)
    legs = th.tensor([[0, 4, 8], [1, 5, 9], [2, 6, 10], [3, 7, 11]], device=DEV)   # FL FR RL RR (type-grouped)
    def residual(d, data):
        q = d.joint_pos[:, amap_t]; qd = d.joint_vel[:, amap_t]
        f_dem = kp[None] * (data.ctrl - q) - kd[None] * qd
        gap = (f_dem.abs() - data.actuator_force.abs()).clamp(min=0) / fnom[None]
        return gap[:, legs].mean(-1).max(dim=1).values          # worst-leg mean saturation gap
    return residual


def rollout(cond, arm, n=N, cal=False, record=False, render=False):
    with contextlib.redirect_stdout(io.StringIO()):
        env = make_tensor("go2_weight_rest_hi_at_0", n, DEV, adversary=True,
                          **({"render_mode": "rgb_array"} if render else {}))
        if render:
            try: env.mj.cfg.viewer.max_extra_envs = n - 1
            except Exception: pass
        tw = {k: load_twin(v, DEV, quiet=True) for k, v in CK.items() if os.path.exists(v)}
    inner = env.mj
    inner.cfg.episode_length_s = 10_000.0
    _ensure_fr_cache(inner)
    ids, nom = inner._fr_act_ids, inner._fr_nominal_forcerange
    residual = make_residual(inner)
    walker = G.Walker(n)
    dstb = th.zeros(n, spec("go2_weight_rest_hi_at_0").dstb_dim, device=DEV); dstb[:, 1] = 1.0
    obs = env.reset()
    d = inner.scene["robot"].data
    W = W_COMPOUND if cond == "compound" else 0.0
    yaw0 = G.yaw_of(d)
    goal = d.root_link_pos_w[:, :2].clone()
    goal[:, 0] += GOAL_D * th.cos(yaw0); goal[:, 1] += GOAL_D * th.sin(yaw0)
    st = th.zeros(n, dtype=th.long, device=DEV)
    if arm == "REST-ONLY": st[:] = 2
    vs_bar = th.zeros(n, device=DEV); vu_bar = th.zeros(n, device=DEV); er_bar = th.zeros(n, device=DEV)
    deg_c = th.zeros(n, device=DEV); ok_c = th.zeros(n, device=DEV)
    below = th.zeros(n, device=DEV); above = th.zeros(n, device=DEV); abort_c = th.zeros(n, device=DEV)
    settle = th.zeros(n, device=DEV); standok = th.zeros(n, device=DEV); refr = th.zeros(n, device=DEV)
    v1_above = th.zeros(n, device=DEV); brake_c = th.zeros(n, device=DEV)
    alive = th.ones(n, dtype=th.bool, device=DEV)
    reached = th.zeros(n, dtype=th.bool, device=DEV)
    t_goal = th.full((n,), float("nan"), device=DEV)
    trig_t = th.full((n,), float("nan"), device=DEV)
    deaths = {s: 0 for s in STATES}
    S, cal_tr = [], []
    traj, sttr, alv_tr, frames = [], [], [], []
    if render:
        import mujoco as _mj
        mm = inner.sim.mj_model
        base_id = _mj.mj_name2id(mm, _mj.mjtObj.mjOBJ_BODY, "robot/base_link")
        trunk = [gg for gg in range(mm.ngeom) if mm.geom_bodyid[gg] == base_id]
        GRAY = np.array([0.5, 0.5, 0.5, 1.0]); ORNG = np.array([0.95, 0.55, 0.1, 1.0])
        ST_COL = {1: np.array([0.90, 0.49, 0.13, 1.0]), 2: np.array([0.15, 0.35, 0.90, 1.0]),
                  3: np.array([0.12, 0.52, 0.29, 1.0])}
    for t in range(STEPS):
        theta = theta_of(t)
        inner.sim.model.actuator_forcerange[:, ids, :] = nom * float(theta)
        env.base_load = th.tensor([0., 0., -W], device=DEV)[None].expand(n, 3).contiguous()
        inner._weight_W[:] = W; inner._weight_h[:] = 0.25
        env.force_scale = 0.2 * th.ones(n, device=DEV)
        m_s, n_s = tw["stand"]; m_r, n_r = tw["rest"]
        Vs = value_of(env, m_s, n_s)
        vs_bar = Vs.clone() if t == 0 else (1 - ALPHA) * vs_bar + ALPHA * Vs
        er = residual(d, inner.sim.data)
        er_bar = er.clone() if t == 0 else (1 - ALPHA) * er_bar + ALPHA * er
        if cal:
            if alive.any():
                cal_tr.append((theta, float(er_bar[alive].median()), float(er_bar[alive].quantile(0.99))))
            else:
                cal_tr.append((theta, float('nan'), float('nan')))
        walking_now = (st == 0)
        deg_sig = (er_bar > ERR_DEG) & walking_now            # residual only meaningful while walking
        deg_c = th.where(deg_sig, deg_c + 1, th.zeros_like(deg_c))
        healthy = th.tensor(theta >= 0.99, device=DEV)        # thermal sensor (belief) for the return gate
        ok_c = th.where(healthy.expand(n), ok_c + 1, th.zeros_like(ok_c))
        below = th.where(vs_bar < EPS_DN_WALK, below + 1, th.zeros_like(below))
        can = (refr <= 0) & (t >= 100)   # 2s arming: spawn + load-settling transient spikes the residual
        go_brake = th.zeros(n, dtype=th.bool, device=DEV)
        if arm in ("ONE-WAY", "V1", "V2-REUSE"):
            go_brake = (st == 0) & (deg_c >= K_DEG) & can   # residual-only: the value channel is blind to actuator faults (the finding)
            trig_t = th.where(go_brake & th.isnan(trig_t), th.full_like(trig_t, t * DT), trig_t)
        brake_c = th.where(st == 4, brake_c + 1, th.zeros_like(brake_c))
        vslow = th.linalg.norm(d.root_link_lin_vel_b, dim=1) < 0.35
        go_desc = (st == 4) & (vslow | (brake_c >= 40))       # descend once stopped (or 0.8s cap)
        go_rest = (st == 1) & (th.where(in_rest_target(inner), settle + 1, th.zeros_like(settle)) >= 5)
        settle = th.where(in_rest_target(inner), settle + 1, th.zeros_like(settle))
        standok = th.where(in_stance_target(inner), standok + 1, th.zeros_like(standok))
        go_up = th.zeros(n, dtype=th.bool, device=DEV)
        go_abort = th.zeros(n, dtype=th.bool, device=DEV)
        if arm == "V2-REUSE":
            m_u, n_u = tw["getup"]
            Vu = value_of(env, m_u, n_u)
            vu_bar = Vu.clone() if t == 0 else (1 - ALPHA) * vu_bar + ALPHA * Vu
            above = th.where(vu_bar > EPS_UP_LEG, above + 1, th.zeros_like(above))
            abort_c = th.where(vu_bar < EPS_ABORT_LEG, abort_c + 1, th.zeros_like(abort_c))
            go_up = (st == 2) & (above >= K_UP) & (ok_c >= HEAL_SUST) & can
            go_abort = (st == 3) & (abort_c >= K_ABORT)
        elif arm == "V1":
            v1_above = th.where(vs_bar > 0.15, v1_above + 1, th.zeros_like(v1_above))
            go_up = (st == 2) & (v1_above >= 25) & can       # BLIND: no θ̂, no getup certificate
        go_stand = (st == 3) & (standok >= 5)
        if arm == "V1":
            go_stand = (st == 3)                              # V1 has no funnel: swap straight to walking
        st = th.where(go_brake, 4 * th.ones_like(st), st)
        st = th.where(go_desc, th.ones_like(st), st)
        st = th.where(go_rest, 2 * th.ones_like(st), st)
        st = th.where(go_up, 3 * th.ones_like(st), st)
        st = th.where(go_stand, th.zeros_like(st), st)
        st = th.where(go_abort, th.ones_like(st), st)
        walker.reset(go_stand)
        refr = th.where(go_brake | go_up | go_abort, th.full_like(refr, REFRACT), refr - 1)
        walking = (st == 0) & th.tensor(arm != "REST-ONLY", device=DEV)
        cmd = G.steer_cmd(d, goal, reached, walking)     # braking robots (st=4) get cmd 0 -> walker stops
        a_w = walker.act(inner, cmd)
        with th.no_grad():
            a_r = th.clamp(m_r.policy._predict(n_r(obs), deterministic=True), -1, 1)
            a = th.where((walking | (st == 4)).unsqueeze(-1), a_w, a_r)
            if arm == "V2-REUSE":
                m_u, n_u = tw["getup"]
                a_u = th.clamp(m_u.policy._predict(n_u(obs), deterministic=True), -1, 1)
                a = th.where((st == 3).unsqueeze(-1), a_u, a)
        obs, g, dones, touts, l = env.step_tensor(th.cat([a, dstb], dim=1))
        z = th.zeros(n, device=DEV)
        tip = inner.termination_manager._term_dones.get("fell_over", z).bool()
        slam = inner.termination_manager._term_dones.get("illegal_contact", z).bool()
        newdead = (tip | slam) & alive
        for si, sname in enumerate(STATES):
            deaths[sname] += int((newdead & (st == si)).sum())
        alive &= ~(tip | slam)
        dist = th.linalg.norm(d.root_link_pos_w[:, :2] - goal, dim=1)
        newreach = (dist < GOAL_R) & alive & ~reached
        t_goal = th.where(newreach, th.full_like(t_goal, t * DT), t_goal)
        reached |= newreach
        S.append(float(alive.float().mean()))
        if record and t % 4 == 0:
            traj.append(d.root_link_pos_w[:, :2].clone())
            sttr.append(st.clone()); alv_tr.append(alive.clone())
        if render:
            maj = int(th.mode(st[alive]).values) if alive.any() else 0
            base = GRAY + (ORNG - GRAY) * (1.0 - theta)       # walking tint: gray -> orange as the leg dies
            tint = base if maj == 0 else ST_COL[maj]
            for gg in trunk: mm.geom_rgba[gg] = tint
            if t % 2 == 0:
                frames.append(np.asarray(env.render()))
    env.close()
    tg = t_goal[reached]
    tt = trig_t[~th.isnan(trig_t)]
    out = {"S": S, "safe": S[-1], "success": float(reached.float().mean()),
           "t_goal_med": float(tg.median()) if tg.numel() else None,
           "trig_med": float(tt.median()) if tt.numel() else None,
           "deaths": deaths, "cal": cal_tr if cal else None}
    if record:
        out["traj"] = th.stack(traj).cpu().numpy(); out["sttr"] = th.stack(sttr).cpu().numpy()
        out["alv_tr"] = th.stack(alv_tr).cpu().numpy()
        out["spawn"] = traj[0].cpu().numpy(); out["yaw0"] = yaw0.cpu().numpy(); out["goal"] = goal.cpu().numpy()
        out["reached_mask"] = reached.cpu().numpy(); out["alive_mask"] = alive.cpu().numpy()
    if render:
        out["frames"] = frames
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cal", action="store_true")
    ap.add_argument("--videos", action="store_true")
    ap.add_argument("--cond", default="legonly")
    args = ap.parse_args()
    OD = os.path.expanduser("~/artifacts/odd-conditioned/E091-leg-walk")
    os.makedirs(OD, exist_ok=True)
    if args.cal:
        r = rollout(args.cond, "WALK-ONLY", n=64, cal=True)
        c = r["cal"]
        h = [x for x in c[100:249]]; dg = [x for x in c[500:740]]
        print(f"healthy walk: med {np.median([x[1] for x in h]):.4f}  p99 {np.median([x[2] for x in h]):.4f}")
        print(f"degraded (0.15): med {np.median([x[1] for x in dg]):.4f}  p99 {np.median([x[2] for x in dg]):.4f}")
        print(f"WALK-ONLY under fault: success {r['success']:.2f} safe {r['safe']:.2f}")
        sys.exit(0)
    if args.videos:
        import imageio.v2 as imageio
        VARMS = ["WALK-ONLY", "V1", "V2-REUSE"]
        for cond in ("legonly",):
            R = {arm: rollout(cond, arm, n=9, record=True, render=True) for arm in VARMS}
            for arm in VARMS:
                print(f"video {cond} {arm}: succ {R[arm]['success']:.2f} safe {R[arm]['safe']:.2f}", flush=True)
            nfr = min(len(R[a]["frames"]) for a in VARMS)
            meds = {}
            for arm in VARMS:
                gt = G.to_goal_frame if hasattr(G, "to_goal_frame") else None
                yaw0 = R[arm]["yaw0"]; spawn = R[arm]["spawn"]
                cth, sth = np.cos(-yaw0), np.sin(-yaw0)
                rel = R[arm]["traj"] - spawn[None]
                gx = rel[..., 0] * cth[None] - rel[..., 1] * sth[None]
                gy = rel[..., 0] * sth[None] + rel[..., 1] * cth[None]
                dist = np.sqrt((gx - GOAL_D) ** 2 + gy ** 2)
                alv = R[arm]["alv_tr"].astype(bool)
                m = np.array([np.median(dist[k][alv[k]]) if alv[k].any() else np.nan for k in range(dist.shape[0])])
                meds[arm] = np.interp(np.arange(nfr) * 2, np.arange(dist.shape[0]) * 4, m)
            import matplotlib; matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            COL = {"WALK-ONLY": "#c0392b", "V1": "#8e44ad", "V2-REUSE": "#1a5276"}
            STN = {0: "WALK", 1: "DESCENDING", 2: "REST", 3: "GETTING-UP"}
            def gframe(i, W_, H_):
                tt_ = np.arange(nfr) * 2 * DT
                thv = np.array([theta_of(k * 2) for k in range(nfr)])
                fig = plt.figure(figsize=(W_ / 100, H_ / 100), dpi=100)
                ax = fig.add_subplot(111); ax2 = ax.twinx()
                ax2.fill_between(tt_, thv, color="#e67e22", alpha=0.12)
                ax2.plot(tt_[:i + 1], thv[:i + 1], color="#e67e22", lw=1.6, label="leg θ(t)")
                ax2.set_ylim(0, 3.2); ax2.set_yticks([0.15, 1.0]); ax2.set_ylabel("θ", color="#e67e22")
                for arm in VARMS:
                    ax.plot(tt_, meds[arm], color=COL[arm], lw=1.0, alpha=0.25)
                    ax.plot(tt_[:i + 1], meds[arm][:i + 1], color=COL[arm], lw=2.2, label=arm)
                ax.axhline(GOAL_R, color="#1a5276", ls=":", lw=1)
                ax.axvline(tt_[i], color="#555", lw=1.5)
                ax.set_xlim(0, tt_[-1]); ax.set_ylim(0, 13)
                ax.set_xlabel("time (s)"); ax.set_ylabel("median dist to goal (live)")
                ax.legend(loc="upper right", fontsize=7.5); ax2.legend(loc="upper left", fontsize=7.5)
                fig.tight_layout(pad=0.5); fig.canvas.draw()
                buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy(); plt.close(fig)
                return buf
            comb = []
            for i in range(nfr):
                row = []
                for arm in VARMS:
                    r = R[arm]
                    k4 = min(i // 2, r["alv_tr"].shape[0] - 1)
                    nal = int(r["alv_tr"][k4].sum())
                    stmaj = int(np.bincount(r["sttr"][k4][r["alv_tr"][k4].astype(bool)]).argmax()) if nal else 0
                    txt = f"{arm}" + (f" [{STN[stmaj]}]" if arm != "WALK-ONLY" else "") + f"   alive {nal}/9"
                    row.append(G.__dict__.get("tag", None) or None)
                    from E090_walk_viz import tag as _tag
                    row[-1] = _tag(r["frames"][i], txt)
                top = np.hstack(row)
                TW = top.shape[1]; GH = 300
                comb.append(np.vstack([top, gframe(i, TW, GH)[:GH, :TW]]))
            vp = f"{OD}/legwalk_{cond}.mp4"
            imageio.mimsave(vp, comb, fps=25, macro_block_size=1)
            print("video ->", vp, flush=True)
        sys.exit(0)
    # ── main: stats + figures ──
    out = {}
    ARMS = ["WALK-ONLY", "REST-ONLY", "ONE-WAY", "V1", "V2-REUSE"]
    for cond in ("legonly",):
        print(f"\n===== {cond} =====  (goal {GOAL_D}m; fault {T_FAULT0}-{T_HEAL}s θ={THETA_LO})", flush=True)
        print(f"{'arm':>10} {'success':>8} {'safe':>6} {'t_goal':>7} {'t_trig':>7}  deaths by state", flush=True)
        data = {}
        for arm in ARMS:
            r = rollout(cond, arm, record=True)
            data[arm] = r; out[f"{cond}|{arm}"] = {k: r[k] for k in ("S", "safe", "success", "t_goal_med", "trig_med", "deaths")}
            dd = " ".join(f"{k}:{v}" for k, v in r["deaths"].items() if v)
            tgs = f"{r['t_goal_med']:.1f}s" if r["t_goal_med"] else "--"
            tts = f"{r['trig_med']:.1f}s" if r["trig_med"] else "--"
            print(f"{arm:>10} {r['success']:>8.2f} {r['safe']:>6.2f} {tgs:>7} {tts:>7}  {dd}", flush=True)
        np.savez_compressed(f"{OD}/traj_{cond}.npz",
                            **{f"{arm}_{k}": data[arm][k] for arm in ARMS
                               for k in ("traj", "sttr", "alv_tr", "yaw0", "spawn", "goal",
                                         "reached_mask", "alive_mask")})
        # top-down + timeline figures (goal frame)
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(ARMS), figsize=(4 * len(ARMS), 4.4), sharex=True, sharey=True)
        for ax, arm in zip(axes, ARMS):
            r = data[arm]
            yaw0 = r["yaw0"]; spawn = r["spawn"]
            cth, sth = np.cos(-yaw0), np.sin(-yaw0)
            rel = r["traj"] - spawn[None]
            gx = rel[..., 0] * cth[None] - rel[..., 1] * sth[None]
            gy = rel[..., 0] * sth[None] + rel[..., 1] * cth[None]
            alv = r["alv_tr"]
            for i in range(gx.shape[1]):
                last = max(int(alv[:, i].sum()), 1)
                if r["reached_mask"][i]:
                    ax.plot(gx[:last, i], gy[:last, i], color="#1e8449", lw=0.7, alpha=0.35)
                elif not r["alive_mask"][i]:
                    ax.plot(gx[:last, i], gy[:last, i], color="#c0392b", lw=0.7, alpha=0.30)
                    ax.plot(gx[last - 1, i], gy[last - 1, i], "x", color="#c0392b", ms=4, alpha=0.7)
                else:
                    ax.plot(gx[:last, i], gy[:last, i], color="#777", lw=0.6, alpha=0.30)
            ax.add_patch(plt.Circle((GOAL_D, 0), GOAL_R, fill=False, color="#1a5276", lw=2))
            ax.plot(0, 0, "k^", ms=8)
            ax.set_title(f"{arm}\nsucc {r['success']:.2f} / safe {r['safe']:.2f}", fontsize=10)
            ax.set_xlim(-2, 13.5); ax.set_ylim(-5, 5); ax.set_aspect("equal"); ax.grid(alpha=0.2)
            ax.set_xlabel("progress toward goal (m)")
        axes[0].set_ylabel("lateral (m)")
        fig.suptitle(f"E091 leg-fault walking — {cond} (green=reached, red=died, gray=alive)", y=1.04)
        fig.tight_layout(); fig.savefig(f"{OD}/topdown_{cond}.png", dpi=140, bbox_inches="tight"); plt.close(fig)
    json.dump(out, open(f"{OD}/results.json", "w"))
    print(f"\nsaved -> {OD}/results.json", flush=True)
