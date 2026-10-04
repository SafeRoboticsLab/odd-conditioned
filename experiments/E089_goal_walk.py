"""E089 — WALKING WITH THE SAFETY FILTER (Buzi's final experiment set).

Task: walk from spawn to a goal GOAL_D = 12 m ahead (success = within GOAL_R while alive) on a 30 s horizon,
while the ODD parameter W follows a single excursion starting/ending at ZERO weight (E089v2 protocol):
    pulse : W = 0, jumps to 220 N for t in [5,13) s, back to 0
    period: W = 220 * sin^2(pi*(t-5)/8) for t in [5,13) s, else 0
Pre-excursion reach is ~4.2 m, so only post-excursion RESUMPTION can score. Conditions benign / medium /
gusty. RESULT (boundary finding): every walking arm ends safe <= ~0.05 — the load-naive walker cannot carry
220 N at h=0.25, and settling under peak load from a gait flips robots. The earlier 9 m-goal protocol looked
positive only because robots finished before the load arrived. The walking demonstration that works is
E092 (payload swap with a W=0 light phase, belief trigger, brake handoff).
The nominal policy is the weight-NAIVE joystick walker from safe_mjlab_zoo (go2_walker_flat, via the
go2_atomic_skills extracted actor+norm; 47-d obs rebuilt here batched). The safety filter owns the weight:
value-based filtering — walker runs while V̄_stand is healthy; collapse => certified descent -> rest;
belief gate (W<130) + V̄_up => certified get-up -> walking resumes toward the goal.

Arms: WALK-ONLY (task expert) | REST-ONLY (safety expert) | ONE-WAY | V1 (batch-1 switching) |
V2-REUSE (certified automaton, walker as the STAND-state policy).
Metrics: success rate (goal reached), safe rate (no tip/slam; N=256, no respawn, timeout disabled),
median time-to-goal, deaths by automaton state.

Modes: --cal (calibrate EPS_DN on walking states: V̄_stand distributions at W=0/120/220),
       --smoke (16-env walker-only sanity: displacement at W=0).
"""
import os, sys, io, math, json, contextlib, argparse
from _paths import _ART
import torch as th
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "external/robot-safety-sandbox"); sys.path.insert(0, "experiments")
sys.path.insert(0, "external/go2_atomic_skills")
import numpy as np
from robot_safety_sandbox import make_tensor, spec
from robot_safety_sandbox.eval.policies import load_twin
import E084_automaton as E
from E084_automaton import (CK, load_twins, value_of, in_rest_target, in_stance_target,
                            EPS_UP, K_UP, EPS_ABORT, K_ABORT, REFRACT, ALPHA, K_DN, WARMUP, V1_UP, V1_K_UP)
from go2_atomic_skills.nets import WalkerNorm, load_actor
from go2_atomic_skills.obs import DEFAULT_JOINT_POS, CTRL_GAIN, WALK_PHASE_PERIOD, PHASE_STAND_EPS

DEV, N, DT, STEPS = "cuda:0", 256, 0.02, 1500
GOAL_D, GOAL_R = 12.0, 1.0
UP_W_GATE = 130.0
EPS_DN_WALK = -0.15        # ROC-calibrated (E089 sweep): 1.6% false sits/30s, det 0.57s@220N
K_DN_WALK = 10             # sustain half: pairs with the deeper threshold
LOOSE_TERM = False         # True: death = REAL flip-over only (|tilt|>80 deg); contact term disabled (Buzi)
STATES = ["STAND", "DESCENDING", "REST", "GETTINGUP"]
DJP = th.tensor(DEFAULT_JOINT_POS, device=DEV)


def W_of(sched, t):
    s = t * DT
    # excursion window [5,13)s + goal 12m (Buzi): pre-excursion reach caps ~4.2m so ONLY resumption scores
    if sched == "pulse":
        return 220.0 if 5.0 <= s < 13.0 else 0.0
    return 220.0 * math.sin(math.pi * (s - 5.0) / 8.0) ** 2 if 5.0 <= s < 13.0 else 0.0


def gust_scale(cond, t):
    if cond == "benign":
        return 0.2                                    # 10 N ambient
    hi = 0.4 if cond == "medium" else 0.7             # gusts: 20 N (medium) / 35 N (gusty)
    return hi if ((t * DT) % 2.0) < 0.5 else 0.2


class Walker:
    """Batched port of go2_atomic_skills.WalkSkill (actor + VecNormalize stats + 47-d obs builder)."""
    def __init__(self, n):
        self.net = load_actor("walker_actor.pt", 47, DEV)
        self.norm = WalkerNorm(DEV)
        self.n = n
        self.reset(th.ones(n, dtype=th.bool, device=DEV))

    def reset(self, mask):
        if not hasattr(self, "last_ctrl"):
            self.last_ctrl = th.zeros(self.n, 12, device=DEV)
            self.stepc = th.zeros(self.n, device=DEV)
        self.last_ctrl[mask] = 0.0
        self.stepc[mask] = 0.0

    def act(self, inner, cmd):
        d = inner.scene["robot"].data
        p = (self.stepc * DT) % WALK_PHASE_PERIOD / WALK_PHASE_PERIOD
        phase = th.stack([th.sin(p * 2 * math.pi), th.cos(p * 2 * math.pi)], dim=1)
        phase = th.where((th.linalg.norm(cmd, dim=1, keepdim=True) < PHASE_STAND_EPS), th.zeros_like(phase), phase)
        obs = th.cat([d.root_link_ang_vel_b, d.projected_gravity_b, cmd, phase,
                      d.joint_pos - DJP, d.joint_vel, self.last_ctrl], dim=1)
        with th.no_grad():
            a = th.clamp(self.net(self.norm(obs)), -1.0, 1.0)
        self.last_ctrl = CTRL_GAIN * a
        self.stepc += 1
        return a


def yaw_of(d):
    q = d.root_link_quat_w    # (N,4) w,x,y,z
    return th.atan2(2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                    1.0 - 2.0 * (q[:, 2] ** 2 + q[:, 3] ** 2))


def steer_cmd(d, goal, reached, walking):
    """Joystick command toward the goal: vx=1.0, wz = P(heading error), zero once reached / not walking."""
    pos = d.root_link_pos_w[:, :2]
    gvec = goal - pos
    hd_err = th.atan2(gvec[:, 1], gvec[:, 0]) - yaw_of(d)
    hd_err = th.atan2(th.sin(hd_err), th.cos(hd_err))
    cmd = th.zeros(pos.shape[0], 3, device=DEV)
    cmd[:, 0] = 1.0
    cmd[:, 2] = th.clamp(1.5 * hd_err, -0.5, 0.5)
    off = reached | ~walking
    cmd[off] = 0.0
    return cmd


def rollout(sched, cond, arm, n=N, cal=False, record=False, render=False):
    with contextlib.redirect_stdout(io.StringIO()):
        env = make_tensor("go2_weight_rest_hi_at_0", n, DEV, adversary=True,
                          **({"render_mode": "rgb_array"} if render else {}))
        if render:
            try: env.mj.cfg.viewer.max_extra_envs = n - 1
            except Exception: pass
        tw = load_twins()
    inner = env.mj
    inner.cfg.episode_length_s = 10_000.0      # no free timeout resets
    if LOOSE_TERM:
        tmn = inner.termination_manager
        names = getattr(tmn, "_term_names", None) or tmn.active_terms
        for nm, c in zip(names, tmn._term_cfgs):
            if nm == "fell_over": c.params["limit_angle"] = 1.3962634       # 80 deg
            if nm == "illegal_contact": c.params["force_threshold"] = 1e9   # disabled
    walker = Walker(n)
    dstb = th.zeros(n, spec("go2_weight_rest_hi_at_0").dstb_dim, device=DEV); dstb[:, 1] = 1.0
    obs = env.reset()
    d = inner.scene["robot"].data
    yaw0 = yaw_of(d)                            # goal lies GOAL_D m along each robot's INITIAL heading
    goal = d.root_link_pos_w[:, :2].clone()
    goal[:, 0] += GOAL_D * th.cos(yaw0)
    goal[:, 1] += GOAL_D * th.sin(yaw0)
    st = th.zeros(n, dtype=th.long, device=DEV)
    if arm == "REST-ONLY": st[:] = 2
    vs_bar = th.zeros(n, device=DEV); vu_bar = th.zeros(n, device=DEV)
    below = th.zeros(n, device=DEV); above = th.zeros(n, device=DEV); abort_c = th.zeros(n, device=DEV)
    settle = th.zeros(n, device=DEV); standok = th.zeros(n, device=DEV); refr = th.zeros(n, device=DEV)
    alive = th.ones(n, dtype=th.bool, device=DEV)
    reached = th.zeros(n, dtype=th.bool, device=DEV)
    t_goal = th.full((n,), float("nan"), device=DEV)
    v1_in_rest = th.zeros(n, dtype=th.bool, device=DEV)
    early_desc = 0
    deaths = {s: 0 for s in STATES}
    S, cal_tr = [], {"vs": [], "W": []}
    traj, sttr, alv_tr, frames = [], [], [], []
    death_xy, death_t, death_st = [], [], []
    if render:
        import mujoco as _mj
        mm = inner.sim.mj_model
        base_id = _mj.mj_name2id(mm, _mj.mjtObj.mjOBJ_BODY, "robot/base_link")
        trunk = [gg for gg in range(mm.ngeom) if mm.geom_bodyid[gg] == base_id]
        GRAY = np.array([0.5, 0.5, 0.5, 1.0]); RED = np.array([0.85, 0.1, 0.1, 1.0])
        ST_COL = {1: np.array([0.90, 0.49, 0.13, 1.0]), 2: np.array([0.15, 0.35, 0.90, 1.0]),
                  3: np.array([0.12, 0.52, 0.29, 1.0])}
    for t in range(STEPS):
        W = W_of(sched, t)
        env.base_load = th.tensor([0., 0., -W], device=DEV)[None].expand(n, 3).contiguous()
        inner._weight_W[:] = W; inner._weight_h[:] = 0.25
        env.force_scale = gust_scale(cond, t) * th.ones(n, device=DEV)
        m_s, n_s = tw["stand"]; m_r, n_r = tw["rest"]
        Vs = value_of(env, m_s, n_s)
        vs_bar = Vs.clone() if t == 0 else (1 - ALPHA) * vs_bar + ALPHA * Vs
        if cal:
            cal_tr["vs"].append(vs_bar.clone()); cal_tr["W"].append(W)
            cal_tr.setdefault("alive", []).append(alive.clone())
        if arm == "V2-REUSE":
            m_u, n_u = tw["getup"]
            Vu = value_of(env, m_u, n_u)
            vu_bar = Vu.clone() if t == 0 else (1 - ALPHA) * vu_bar + ALPHA * Vu
            below = th.where(vs_bar < EPS_DN_WALK, below + 1, th.zeros_like(below))
            above = th.where(vu_bar > EPS_UP, above + 1, th.zeros_like(above))
            abort_c = th.where(vu_bar < EPS_ABORT, abort_c + 1, th.zeros_like(abort_c))
            settle = th.where(in_rest_target(inner), settle + 1, th.zeros_like(settle))
            standok = th.where(in_stance_target(inner), standok + 1, th.zeros_like(standok))
            can = (refr <= 0) & (t >= WARMUP)
            go_desc = (st == 0) & (below >= K_DN_WALK) & can
            if t * DT < 5.0:
                early_desc += int((go_desc & alive).sum())
            go_rest = (st == 1) & (settle >= 5)
            go_up = (st == 2) & (above >= K_UP) & can
            if W >= UP_W_GATE:
                go_up = th.zeros_like(go_up)
            go_stand = (st == 3) & (standok >= 5)
            go_abort = (st == 3) & (abort_c >= K_ABORT)
            st = th.where(go_desc, th.ones_like(st), st)
            st = th.where(go_rest, 2 * th.ones_like(st), st)
            st = th.where(go_up, 3 * th.ones_like(st), st)
            st = th.where(go_stand, th.zeros_like(st), st)
            st = th.where(go_abort, th.ones_like(st), st)
            walker.reset(go_stand)                      # fresh gait clock when walking resumes
            refr = th.where(go_desc | go_up | go_abort, th.full_like(refr, REFRACT), refr - 1)
        elif arm in ("V1", "ONE-WAY"):
            below = th.where(vs_bar < EPS_DN_WALK, below + 1, th.zeros_like(below))
            above = th.where(vs_bar > V1_UP, above + 1, th.zeros_like(above))
            can = (refr <= 0) & (t >= WARMUP)
            go_dn = (~v1_in_rest) & (below >= K_DN_WALK) & can
            if t * DT < 5.0:
                early_desc += int((go_dn & alive).sum())
            go_up = v1_in_rest & (above >= V1_K_UP) & can & th.tensor(arm == "V1", device=DEV)
            v1_in_rest = th.where(go_dn, th.ones_like(v1_in_rest), v1_in_rest)
            v1_in_rest = th.where(go_up, th.zeros_like(v1_in_rest), v1_in_rest)
            walker.reset(go_up)
            refr = th.where(go_dn | go_up, th.full_like(refr, REFRACT), refr - 1)
            st = th.where(v1_in_rest, 2 * th.ones_like(st), th.zeros_like(st))
        walking = (st == 0) & (arm != "REST-ONLY")
        cmd = steer_cmd(d, goal, reached, walking)
        a_w = walker.act(inner, cmd)
        with th.no_grad():
            a_r = th.clamp(m_r.policy._predict(n_r(obs), deterministic=True), -1, 1)
            a = th.where(walking.unsqueeze(-1), a_w, a_r)
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
        if record and newdead.any():
            death_xy.append(d.root_link_pos_w[newdead, :2].clone())
            death_t += [t * DT] * int(newdead.sum())
            death_st += [int(x) for x in st[newdead]]
        alive &= ~(tip | slam)
        if record and t % 4 == 0:
            traj.append(d.root_link_pos_w[:, :2].clone())
            sttr.append(st.clone()); alv_tr.append(alive.clone())
        if render:
            maj = int(th.mode(st[alive]).values) if alive.any() else 0
            tint = (GRAY + (RED - GRAY) * min(1.0, W / 250.0)) if maj == 0 else ST_COL[maj]
            for gg in trunk: mm.geom_rgba[gg] = tint
            if t % 2 == 0:
                frames.append(np.asarray(env.render()))
        dist = th.linalg.norm(d.root_link_pos_w[:, :2] - goal, dim=1)
        newreach = (dist < GOAL_R) & alive & ~reached
        t_goal = th.where(newreach, th.full_like(t_goal, t * DT), t_goal)
        reached |= newreach
        S.append(float(alive.float().mean()))
    env.close()
    tg = t_goal[reached]
    out = {"S": S, "safe": S[-1], "success": float(reached.float().mean()), "early_desc": early_desc,
           "t_goal_med": float(tg.median()) if tg.numel() else None,
           "deaths": deaths, "cal": cal_tr if cal else None}
    if record:
        out["traj"] = th.stack(traj).cpu().numpy()          # (T/4, n, 2)
        out["sttr"] = th.stack(sttr).cpu().numpy()
        out["alv_tr"] = th.stack(alv_tr).cpu().numpy()
        out["spawn"] = (traj[0]).cpu().numpy()
        out["yaw0"] = yaw0.cpu().numpy(); out["goal"] = goal.cpu().numpy()
        out["reached_mask"] = reached.cpu().numpy(); out["alive_mask"] = alive.cpu().numpy()
        out["death_xy"] = (th.cat(death_xy).cpu().numpy() if death_xy else np.zeros((0, 2)))
        out["death_t"] = np.array(death_t); out["death_st"] = np.array(death_st)
    if render:
        out["frames"] = frames
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cal", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--eps_dn", type=float, default=None)
    args = ap.parse_args()
    if args.eps_dn is not None:
        EPS_DN_WALK = args.eps_dn
    if args.smoke:
        r = rollout("pulse", "benign", "WALK-ONLY", n=16)
        print(f"SMOKE walker in weight env: success {r['success']:.2f} safe {r['safe']:.2f} "
              f"t_goal {r['t_goal_med']} deaths {r['deaths']}")
        sys.exit(0)
    if args.cal:
        # V̄_stand while WALKING at constant W: 0 / 120 / 220 (benign) — pick the threshold between bands
        import types
        for Wfix in (0.0, 120.0, 220.0):
            E089 = sys.modules[__name__]
            saved = E089.W_of
            E089.W_of = lambda sched, t, _w=Wfix: _w
            r = rollout("pulse", "benign", "WALK-ONLY", n=64, cal=True)
            E089.W_of = saved
            vs = th.stack(r["cal"]["vs"][100:400])     # t in [2,8] s
            q = th.quantile(vs.flatten(), th.tensor([0.01, 0.05, 0.5, 0.95], device=vs.device))
            print(f"W={Wfix:5.0f}: V̄_stand p1={q[0]:+.3f} p5={q[1]:+.3f} p50={q[2]:+.3f} p95={q[3]:+.3f} "
                  f"(safe {r['safe']:.2f}, success {r['success']:.2f})")
        sys.exit(0)
    out = {}
    for cond in ("benign", "medium", "gusty"):
        for sched in ("pulse", "period"):
            print(f"\n===== {sched} / {cond} =====  (goal {GOAL_D}m, eps_dn={EPS_DN_WALK})", flush=True)
            print(f"{'arm':>10} {'success':>8} {'safe':>6} {'t_goal':>7}  deaths by state", flush=True)
            for arm in ("WALK-ONLY", "REST-ONLY", "ONE-WAY", "V1", "V2-REUSE"):
                r = rollout(sched, cond, arm)
                out[f"{cond}|{sched}|{arm}"] = r
                dd = " ".join(f"{k}:{v}" for k, v in r["deaths"].items() if v)
                tgs = f"{r['t_goal_med']:.1f}s" if r["t_goal_med"] else "--"
                print(f"{arm:>10} {r['success']:>8.2f} {r['safe']:>6.2f} {tgs:>7}  {dd}", flush=True)
    od = os.path.expanduser(_ART + "/E089-goal-walk")
    os.makedirs(od, exist_ok=True)
    json.dump(out, open(f"{od}/results.json", "w"))
    print(f"\nsaved -> {od}/results.json", flush=True)
