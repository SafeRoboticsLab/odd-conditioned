"""E092 — THE PAYLOAD-SWAP WALKING EVALUATION (Buzi's two claims, weight-ladder setup).

ODD = the carried payload, jointly (W, h): light = UNLOADED (W 0, h 0.25) — the walker's comfort zone
(it was trained at W=0; even 40 N taxed every arm ~45% over 40 s, so iteration-3 dropped it); heavy =
(220 N, h 0.40) — a TALL heavy object: fatal to ANYTHING upright (walker 0.00, stand expert 0.00, stopping
0.36-and-useless; probes 2026-08-24) while settled rest reads 1.00. Schedules: pulse (step, t in [5,13)s)
and period (sin^2 bump, same window; h interpolates with W). Horizon 40 s, goal 12 m (pre-pulse reach <=
~4.2 m: only post-pulse resumption scores). Benign disturbance. LOOSE terminations: death = genuine
flip-over (|tilt| > 80 deg); contact term off. N=256, no respawn.

The two claims this is built to test:
  1. SAFETY: V2 tunnels between ODDs through the certified rest set -> survival curves.
  2. PERMISSIVENESS: V2 withdraws only when needed and RESUMES when safe -> success-CDF over time
     (cumulative fraction of the ORIGINAL fleet that has reached the goal — counts, not survivor-medians).

Arms: WALK-ONLY | REST-ONLY | ONE-WAY | V1 (blind return) | V2 (brake -> descend FUNNEL -> rest ->
belief-gated certified get-up -> walk). All switching arms share the trigger (V̄_stand ROC point -0.15/K10)
and the brake; they differ only in the return.
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
from E084_automaton import (CK, load_twins, cal_summary, value_of, in_rest_target, in_stance_target,
                            EPS_UP, K_UP, EPS_ABORT, K_ABORT, REFRACT, ALPHA)
import E089_goal_walk as G

DEV, N, DT, STEPS = "cuda:0", 256, 0.02, 2000
GOAL_D, GOAL_R = 12.0, 1.0
T0, T1 = 5.0, 13.0                     # heavy window
W_LO, W_HI, H_LO, H_HI = 0.0, 220.0, 0.25, 0.35   # light = UNLOADED (walker trained at W=0; even 40N is marginal over long horizons)
W_TRIG = 60.0                          # belief trigger at the walking-ODD edge
BRAKE_CAP = 40                         # max brake steps before the funnel takes over regardless
UP_SUST = 100                          # belief DEBOUNCE: W must stay low 2s before the return gate opens
                                       # (a 1.2s false dip must NOT bait the certified return)
EPS_UP_92, EPS_ABORT_92 = -0.30, -0.45 # return certificate recalibrated on THIS regime's rest poses (ordinal use)
EPS_DN_WALK, K_DN_WALK = -0.15, 10     # ROC-calibrated walking trigger
UP_W_GATE = 130.0
PUSH_SCALE = 0.2                       # scripted constant +y push = PUSH_SCALE x 50 N (10 N); 0 = no push
STATES = ["WALK", "DESCENDING", "REST", "GETTINGUP", "BRAKE"]
ARMS = ["WALK-ONLY", "REST-ONLY", "ONE-WAY", "V1", "V2"]


def Wh_of(sched, t):
    s = t * DT
    if not (T0 <= s < T1):
        return W_LO, H_LO
    if sched == "pulse":
        return W_HI, H_HI
    a = math.sin(math.pi * (s - T0) / (T1 - T0)) ** 2
    W = W_LO + (W_HI - W_LO) * a
    if sched == "dip":                 # FALSE DIP: load briefly lightens mid-window (bait for blind returns)
        W = W * (1.0 - 0.75 * math.exp(-(((s - 9.7) / 1.0) ** 2)))   # min ~55N; below-gate ~1.6s (< 2s debounce, > 0.5s blind guard)
    h = H_LO + (H_HI - H_LO) * (W - W_LO) / (W_HI - W_LO)
    return W, h


def rollout(sched, arm, n=N, record=False, render=False, cal_up=False):
    """cal_up=True (arm V2): return disabled; collects EMA V_up on settled-rest robots once the debounced belief
    says the load has cleared — the population the return gate sees (calibrates EPS_UP_92)."""
    with contextlib.redirect_stdout(io.StringIO()):
        env = make_tensor("go2_weight_rest_hi_at_0", n, DEV, adversary=True,
                          **({"render_mode": "rgb_array"} if render else {}))
        if render:
            try: env.mj.cfg.viewer.max_extra_envs = n - 1
            except Exception: pass
        tw = load_twins()
    inner = env.mj
    inner.cfg.episode_length_s = 10_000.0
    tmn = inner.termination_manager
    names = getattr(tmn, "_term_names", None) or tmn.active_terms
    for nm, c in zip(names, tmn._term_cfgs):
        if nm == "fell_over": c.params["limit_angle"] = 1.3962634          # 80 deg: REAL flip-overs only
        if nm == "illegal_contact": c.params["force_threshold"] = 1e9      # off (loose criterion)
    walker = G.Walker(n)
    dstb = th.zeros(n, spec("go2_weight_rest_hi_at_0").dstb_dim, device=DEV); dstb[:, 1] = 1.0
    obs = env.reset()
    d = inner.scene["robot"].data
    yaw0 = G.yaw_of(d)
    goal = d.root_link_pos_w[:, :2].clone()
    goal[:, 0] += GOAL_D * th.cos(yaw0); goal[:, 1] += GOAL_D * th.sin(yaw0)
    st = th.zeros(n, dtype=th.long, device=DEV)
    if arm == "REST-ONLY": st[:] = 2
    vs_bar = th.zeros(n, device=DEV); vu_bar = th.zeros(n, device=DEV)
    below = th.zeros(n, device=DEV); above = th.zeros(n, device=DEV); abort_c = th.zeros(n, device=DEV)
    settle = th.zeros(n, device=DEV); standok = th.zeros(n, device=DEV); refr = th.zeros(n, device=DEV)
    v1_above = th.zeros(n, device=DEV); brake_c = th.zeros(n, device=DEV)
    wcnt = 0; wlow = 0
    alive = th.ones(n, dtype=th.bool, device=DEV)
    reached = th.zeros(n, dtype=th.bool, device=DEV)
    hold = th.randint(0, 30, (n,), device=DEV)      # staggered walk start: desyncs gait phase from the
                                                    # schedule (identical clocks made handoff mortality
                                                    # deterministic — it is GAIT-PHASE-dependent)
    t_goal = th.full((n,), float("nan"), device=DEV)
    early_desc = 0
    deaths = {s: 0 for s in STATES}
    S, SUC = [], []
    cal_vals = []
    traj, sttr, alv_tr, frames = [], [], [], []
    if render:
        import mujoco as _mj
        mm = inner.sim.mj_model
        base_id = _mj.mj_name2id(mm, _mj.mjtObj.mjOBJ_BODY, "robot/base_link")
        trunk = [gg for gg in range(mm.ngeom) if mm.geom_bodyid[gg] == base_id]
        GRAY = np.array([0.5, 0.5, 0.5, 1.0]); RED = np.array([0.85, 0.1, 0.1, 1.0])
        ST_COL = {1: np.array([0.90, 0.49, 0.13, 1.0]), 2: np.array([0.15, 0.35, 0.90, 1.0]),
                  3: np.array([0.12, 0.52, 0.29, 1.0]), 4: np.array([0.95, 0.85, 0.2, 1.0])}
    for t in range(STEPS):
        W, h = Wh_of(sched, t)
        env.base_load = th.tensor([0., 0., -W], device=DEV)[None].expand(n, 3).contiguous()
        inner._weight_W[:] = W; inner._weight_h[:] = h
        env.force_scale = PUSH_SCALE * th.ones(n, device=DEV)
        m_s, n_s = tw["stand"]; m_r, n_r = tw["rest"]
        Vs = value_of(env, m_s, n_s)
        vs_bar = Vs.clone() if t == 0 else (1 - ALPHA) * vs_bar + ALPHA * Vs
        below = th.where(vs_bar < EPS_DN_WALK, below + 1, th.zeros_like(below))
        can = (refr <= 0) & (t >= 100)
        wcnt = wcnt + 1 if W >= W_TRIG else 0
        wlow = wlow + 1 if W < UP_W_GATE else 0
        go_brake = th.zeros(n, dtype=th.bool, device=DEV)
        if arm in ("ONE-WAY", "V1", "V2"):
            # BELIEF trigger (W observed / E018-estimated) OR value collapse as backup — the same
            # belief-primary pattern as the leg-fault residual: the walker's gait is OOD for V̄_stand,
            # so the value alone is a weak weight detector (E092 run-1: 25% false sits, late ramp firing)
            go_brake = (st == 0) & th.tensor(wcnt >= 3, device=DEV) & can   # belief-ONLY: the value backup false-fired ~20-25% on gaits
            if t * DT < T0:
                early_desc += int((go_brake & alive).sum())
        brake_c = th.where(st == 4, brake_c + 1, th.zeros_like(brake_c))
        vslow = th.linalg.norm(d.root_link_lin_vel_b, dim=1) < 0.35
        go_desc = (st == 4) & (vslow | (brake_c >= BRAKE_CAP))
        settle = th.where(in_rest_target(inner), settle + 1, th.zeros_like(settle))
        standok = th.where(in_stance_target(inner), standok + 1, th.zeros_like(standok))
        go_rest = (st == 1) & (settle >= 5)
        go_up = th.zeros(n, dtype=th.bool, device=DEV)
        go_abort = th.zeros(n, dtype=th.bool, device=DEV)
        if arm == "V2":
            m_u, n_u = tw["getup"]
            Vu = value_of(env, m_u, n_u)
            vu_bar = Vu.clone() if t == 0 else (1 - ALPHA) * vu_bar + ALPHA * Vu
            above = th.where(vu_bar > EPS_UP_92, above + 1, th.zeros_like(above))
            abort_c = th.where(vu_bar < EPS_ABORT_92, abort_c + 1, th.zeros_like(abort_c))
            go_up = (st == 2) & (above >= K_UP) & can
            if wlow < UP_SUST:                       # debounced belief: sustained-clear, not instantaneous
                go_up = th.zeros_like(go_up)
            if cal_up:
                go_up = th.zeros_like(go_up)
                if wlow >= UP_SUST:
                    cal_vals.append(vu_bar[alive & (st == 2) & (settle >= 5)].clone())
            go_abort = (st == 3) & (abort_c >= K_ABORT)
        elif arm == "V1":
            v1_above = th.where(vs_bar > 0.15, v1_above + 1, th.zeros_like(v1_above))
            go_up = (st == 2) & (v1_above >= 25) & can          # blind return (no belief, no funnel)
        go_stand = (st == 3) & (standok >= 5)
        if arm == "V1":
            go_stand = (st == 3)
        st = th.where(go_brake, 4 * th.ones_like(st), st)
        st = th.where(go_desc, th.ones_like(st), st)
        st = th.where(go_rest, 2 * th.ones_like(st), st)
        st = th.where(go_up, 3 * th.ones_like(st), st)
        st = th.where(go_stand, th.zeros_like(st), st)
        st = th.where(go_abort, th.ones_like(st), st)
        walker.reset(go_stand)
        refr = th.where(go_brake | go_up | go_abort, th.full_like(refr, REFRACT), refr - 1)
        walking = (st == 0) & th.tensor(arm != "REST-ONLY", device=DEV)
        cmd = G.steer_cmd(d, goal, reached, walking)      # braking robots get cmd 0 (walker stops in place)
        cmd[t < hold] = 0.0
        a_w = walker.act(inner, cmd)
        with th.no_grad():
            a_r = th.clamp(m_r.policy._predict(n_r(obs), deterministic=True), -1, 1)
            a_st = th.clamp(m_s.policy._predict(n_s(obs), deterministic=True), -1, 1)
            a = th.where(walking.unsqueeze(-1), a_w, a_r)
            a = th.where((st == 4).unsqueeze(-1), a_st, a)   # BRAKE = stand_hi, the mode's own stabilizer
                                                             # (a zero-command walker freezes its gait clock and stumbles)
            if arm == "V2":
                m_d, n_d = tw["descend"]
                a_d = th.clamp(m_d.policy._predict(n_d(obs), deterministic=True), -1, 1)
                a = th.where((st == 1).unsqueeze(-1), a_d, a)          # the DEDICATED descent funnel
                m_u, n_u = tw["getup"]
                a_u = th.clamp(m_u.policy._predict(n_u(obs), deterministic=True), -1, 1)
                a = th.where((st == 3).unsqueeze(-1), a_u, a)
        obs, g, dones, touts, l = env.step_tensor(th.cat([a, dstb], dim=1))
        z = th.zeros(n, device=DEV)
        tip = inner.termination_manager._term_dones.get("fell_over", z).bool()
        newdead = tip & alive
        for si, sname in enumerate(STATES):
            deaths[sname] += int((newdead & (st == si)).sum())
        alive &= ~tip
        dist = th.linalg.norm(d.root_link_pos_w[:, :2] - goal, dim=1)
        newreach = (dist < GOAL_R) & alive & ~reached
        t_goal = th.where(newreach, th.full_like(t_goal, t * DT), t_goal)
        reached |= newreach
        S.append(float(alive.float().mean()))
        SUC.append(float(reached.float().mean()))
        if record and t % 4 == 0:
            traj.append(d.root_link_pos_w[:, :2].clone())
            sttr.append(st.clone()); alv_tr.append(alive.clone())
        if render:
            maj = int(th.mode(st[alive]).values) if alive.any() else 0
            tint = (GRAY + (RED - GRAY) * min(1.0, W / 250.0)) if maj == 0 else ST_COL[maj]
            for gg in trunk: mm.geom_rgba[gg] = tint
            if t % 2 == 0:
                frames.append(np.asarray(env.render()))
    env.close()
    tg = t_goal[reached]
    out = {"S": S, "SUC": SUC, "safe": S[-1], "success": SUC[-1],
           "t_goal_med": float(tg.median()) if tg.numel() else None,
           "early_desc": early_desc, "deaths": deaths, "cal": cal_vals}
    if record:
        out["traj"] = th.stack(traj).cpu().numpy(); out["sttr"] = th.stack(sttr).cpu().numpy()
        out["alv_tr"] = th.stack(alv_tr).cpu().numpy()
        out["spawn"] = traj[0].cpu().numpy(); out["yaw0"] = yaw0.cpu().numpy(); out["goal"] = goal.cpu().numpy()
        out["reached_mask"] = reached.cpu().numpy(); out["alive_mask"] = alive.cpu().numpy()
    if render:
        out["frames"] = frames
    return out


def claims_fig(sched, data, od):
    """The two-claims figure: survival S(t) (claim 1) + success-CDF (claim 2), same time axis."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    COL = {"WALK-ONLY": "#c0392b", "REST-ONLY": "#1e8449", "ONE-WAY": "#e67e22", "V1": "#8e44ad", "V2": "#1a5276"}
    t = np.arange(STEPS) * DT
    Wv = np.array([Wh_of(sched, k)[0] for k in range(STEPS)])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, key, ylab, title in ((axes[0], "S", "survival S(t)", "Claim 1 — safety (death = flip-over)"),
                                 (axes[1], "SUC", "success CDF (fraction of fleet at goal)",
                                  "Claim 2 — permissiveness (goal completions over time)")):
        ax2 = ax.twinx()
        ax2.fill_between(t, Wv, color="#777", alpha=0.10)
        ax2.set_ylim(0, 900); ax2.set_yticks([40, 220]); ax2.tick_params(labelsize=7, colors="#777")
        for arm in ARMS:
            ax.plot(t, data[arm][key], color=COL[arm], lw=2.2 if arm == "V2" else 1.5,
                    label=f"{arm} ({data[arm][key][-1]:.2f})")
        ax.set_xlim(0, t[-1]); ax.set_ylim(0, 1.02); ax.grid(alpha=0.25)
        ax.set_xlabel("time (s)"); ax.set_ylabel(ylab); ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8, loc="center right")
    fig.suptitle(f"E092 payload-swap walking — {sched} (tall heavy payload W 40→220 N, CoM 0.25→0.40 m)", y=1.03)
    fig.tight_layout()
    fig.savefig(f"{od}/claims_{sched}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def topdown_fig(sched, data, od):
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
        nsucc = int(r["reached_mask"].sum()); nal = int(r["alive_mask"].sum())
        ax.set_title(f"{arm}\nreached {nsucc}/256 | alive {nal}/256", fontsize=10)
        ax.set_xlim(-2, 14.5); ax.set_ylim(-5, 5); ax.set_aspect("equal"); ax.grid(alpha=0.2)
        ax.set_xlabel("progress toward goal (m)")
    axes[0].set_ylabel("lateral (m)")
    fig.suptitle(f"E092 top-down — {sched} (green=reached, red=flipped at X, gray=alive short of goal)", y=1.04)
    fig.tight_layout()
    fig.savefig(f"{od}/topdown_{sched}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", action="store_true")
    ap.add_argument("--cal-up", action="store_true",
                    help="V_up on settled-rest robots after the load clears, return disabled (EPS_UP_92)")
    args = ap.parse_args()
    if args.cal_up:
        r = rollout("period", "V2", cal_up=True)
        cal_summary("V_up on settled rest, load cleared (W=0, walking regime)", r["cal"], EPS_UP_92, "above")
        sys.exit(0)
    OD = os.path.expanduser(_ART + "/E092-payload-walk")
    os.makedirs(OD, exist_ok=True)
    if args.videos:
        # SINGLE BEST RUN per method (Buzi): up to 6 solo (n=1) rendered tries per arm; keep the first
        # that matches the arm's best-case (reached; else alive; else furthest). Graph strip shows the
        # FLEET curves from results.json (the quantitative record), not the solo robot.
        import imageio.v2 as imageio
        from E090_walk_viz import tag as _tag
        from PIL import Image, ImageDraw
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        def stack_overlay(img, W, h):
            im = Image.fromarray(np.ascontiguousarray(img)); dr = ImageDraw.Draw(im)
            x0, ybase = 10, 120
            nb = int(round(W / 40.0))
            lift = int((h - 0.25) * 200)
            for b in range(nb):
                y1 = ybase - lift - b * 12
                dr.rectangle([x0, y1 - 10, x0 + 26, y1], fill=(180, 60, 40), outline=(30, 30, 30))
            dr.text((x0, ybase + 6), f"W={W:.0f}N", fill=(255, 255, 255))
            dr.text((x0, ybase + 18), f"CoM h={h:.2f}m", fill=(255, 255, 255))
            return np.asarray(im)

        fleet = json.load(open(f"{OD}/results.json"))
        VARMS = ["WALK-ONLY", "ONE-WAY", "V1", "V2"]
        COL = {"WALK-ONLY": "#c0392b", "ONE-WAY": "#e67e22", "V1": "#8e44ad", "V2": "#1a5276"}
        STN = {0: "WALK", 1: "DESCENDING", 2: "REST", 3: "GETTING-UP", 4: "BRAKING"}
        for sched in ("period", "dip"):
            best = {}
            for arm in VARMS:
                pick, pickscore = None, (-1, -1.0)
                for k in range(6):
                    r = rollout(sched, arm, n=1, record=True, render=True)
                    prog = float(GOAL_D - np.linalg.norm(r["traj"][-1, 0] - r["goal"][0]))
                    score = (2 if r["success"] > 0 else (1 if r["safe"] > 0 else 0), prog)
                    if score > pickscore:
                        pick, pickscore = r, score
                    if score[0] == 2 or (arm in ("WALK-ONLY", "ONE-WAY") and score[0] >= (0 if arm == "WALK-ONLY" else 1) and k >= 2):
                        break
                best[arm] = pick
                print(f"solo {sched} {arm}: succ {pick['success']:.0f} safe {pick['safe']:.0f} tries {k+1}", flush=True)
            nfr = min(len(best[a]["frames"]) for a in VARMS)
            def gframe(i, W_, H_):
                tt_ = np.arange(nfr) * 2 * DT
                Wv = np.array([Wh_of(sched, k2 * 2)[0] for k2 in range(nfr)])
                fig = plt.figure(figsize=(W_ / 100, H_ / 100), dpi=100)
                ax = fig.add_subplot(111); ax2 = ax.twinx()
                ax2.fill_between(tt_, Wv, color="#777", alpha=0.12)
                ax2.plot(tt_[:i + 1], Wv[:i + 1], color="#444", lw=1.6, label="payload W(t)")
                ax2.set_ylim(0, 900); ax2.set_yticks([60, 130, 220]); ax2.set_ylabel("W (N)")
                for arm in VARMS:
                    fr = fleet[f"{sched}|{arm}"]
                    S = np.array(fr["SUC"])[::2][:nfr]
                    A = np.array(fr["S"])[::2][:nfr]
                    ax.plot(tt_[:i + 1], S[:i + 1], color=COL[arm], lw=2.4,
                            label=f"{arm} reached ({fr['success']:.2f})")
                    ax.plot(tt_[:i + 1], A[:i + 1], color=COL[arm], lw=1.1, ls="--", alpha=0.7)
                ax.axvline(tt_[i], color="#555", lw=1.5)
                ax.set_xlim(0, tt_[-1]); ax.set_ylim(0, 1.05)
                ax.set_xlabel("time (s)"); ax.set_ylabel("FLEET fraction (solid=reached, dashed=alive; N=256)")
                ax.legend(loc="upper left", fontsize=7); ax2.legend(loc="lower right", fontsize=7)
                fig.tight_layout(pad=0.5); fig.canvas.draw()
                buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy(); plt.close(fig)
                return buf
            comb = []
            for i in range(nfr):
                row = []
                for arm in VARMS:
                    r = best[arm]
                    k4 = min(i // 2, r["alv_tr"].shape[0] - 1)
                    al = bool(r["alv_tr"][k4][0])
                    stv = int(r["sttr"][k4][0])
                    status = ("REACHED" if (r["success"] > 0 and np.array(r["SUC"])[min(i*2, len(r["SUC"])-1)] > 0)
                              else (STN[stv] if al else "FLIPPED"))
                    txt = f"{arm}  [{status}]"
                    Wi, hi_ = Wh_of(sched, i * 2)
                    row.append(stack_overlay(_tag(r["frames"][i], txt), Wi, hi_))
                top = np.hstack(row)
                TW = top.shape[1]; GH = 300
                comb.append(np.vstack([top, gframe(i, TW, GH)[:GH, :TW]]))
            vp = f"{OD}/payload_{sched}_best.mp4"
            imageio.mimsave(vp, comb, fps=25, macro_block_size=1)
            print("video ->", vp, flush=True)
        sys.exit(0)
    out = {}
    for sched in ("pulse", "period", "dip"):
        print(f"\n===== {sched} =====  (goal {GOAL_D}m, heavy [{T0},{T1})s W{W_HI:.0f}/h{H_HI}, loose term)", flush=True)
        print(f"{'arm':>10} {'success':>8} {'safe':>6} {'t_goal':>7} {'earlydesc':>10}  deaths by state", flush=True)
        data = {}
        for arm in ARMS:
            r = rollout(sched, arm, record=True)
            data[arm] = r
            out[f"{sched}|{arm}"] = {k: r[k] for k in ("S", "SUC", "safe", "success", "t_goal_med", "early_desc", "deaths")}
            dd = " ".join(f"{k}:{v}" for k, v in r["deaths"].items() if v)
            tgs = f"{r['t_goal_med']:.1f}s" if r["t_goal_med"] else "--"
            print(f"{arm:>10} {r['success']:>8.2f} {r['safe']:>6.2f} {tgs:>7} {r['early_desc']:>10}  {dd}", flush=True)
        np.savez_compressed(f"{OD}/traj_{sched}.npz",
                            **{f"{arm}_{k}": data[arm][k] for arm in ARMS
                               for k in ("traj", "sttr", "alv_tr", "yaw0", "spawn", "goal",
                                         "reached_mask", "alive_mask")})
        claims_fig(sched, data, OD)
        topdown_fig(sched, data, OD)
        print(f"figures -> claims_{sched}.png, topdown_{sched}.png", flush=True)
    json.dump(out, open(f"{OD}/results.json", "w"))
    print(f"\nsaved -> {OD}/results.json", flush=True)
