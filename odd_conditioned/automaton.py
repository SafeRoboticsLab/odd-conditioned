"""The ODD-conditioned safety filter: an automaton over specification modes, and the baselines it is compared to.

Modes and the policy that flies each:

    TASK         the task — standing (STAND expert) or walking (nominal walker)
    BRAKE        walking only: stop in place before descending (STAND expert, or the walker with a zero command)
    DESCENDING   certified descent funnel STAND -> REST (or the REST expert, in the ablation / where noted)
    REST         REST expert: lie down, settle; certified safe under every ODD in the family
    GETTING_UP   certified get-up funnel REST -> STAND, monitored by its own certificate V_up

Transitions (all signals are EMA-smoothed, every switch arms a refractory period):

    TASK -> BRAKE | DESCENDING   descent trigger (per scenario): V̄_stand < eps, observed load W >= eps,
                                 or the leg torque-saturation residual > eps, held for K steps
    BRAKE -> DESCENDING          the robot has slowed down, or the brake has run BRAKE_CAP steps
    DESCENDING -> REST           inside the settled-rest target set for SETTLE steps
    REST -> GETTING_UP           V̄_up > eps_up for K_UP steps, AND the belief says the nominal ODD is back
    GETTING_UP -> TASK           inside the stance target set for SETTLE steps
    GETTING_UP -> DESCENDING     V̄_up < eps_abort for K_ABORT steps (certified abort)

Methods:
    odd               ODD-conditioned: the full automaton above
    odd-rest-descent  ablation: the REST expert flies the descent instead of the funnel
    direct            ODD-conditioned (direct): same descent; returns blind, when V̄_stand read while lying down
                      exceeds DIRECT_EPS_UP — no belief gate, no get-up funnel, no abort
    one-way           same descent; never returns
    task-only         the task policy alone (STAND-ONLY / WALK-ONLY)
    rest-only         the REST expert alone (REST-ONLY)

Deaths are the environment's own failure terminations; a dead robot stays dead (no respawn, no timeout).
"""
import math

import numpy as np
import torch as th

from . import sim
from .policies import AUTOMATON, LegResidual, Walker, load, steer_cmd, yaw_of
from .sim import DEV, DT

TASK, DESCENDING, REST, GETTING_UP, BRAKE = range(5)
METHODS = ("task-only", "rest-only", "one-way", "direct", "odd", "odd-rest-descent")
SWITCHING = ("one-way", "direct", "odd", "odd-rest-descent")
CERTIFIED = ("odd", "odd-rest-descent")

ALPHA = 0.1                          # EMA weight of every runtime signal
REFRACTORY = 50                      # steps after a switch before the next one may fire
K_UP, K_ABORT = 15, 10
SETTLE = 5                           # steps inside a target set that complete a transition
DIRECT_EPS_UP, DIRECT_K_UP = 0.15, 25
BRAKE_CAP, BRAKE_SLOW = 40, 0.35     # steps, m/s
LOAD_NOMINAL = 130.0                 # belief "load": W below this is the standing/walking ODD (N)
LEG_HEALTHY = 0.99                   # belief "leg": θ at or above this is a healthy leg
GOAL_RADIUS = 1.0


def label(method, task):
    return {"task-only": "STAND-ONLY" if task == "stand" else "WALK-ONLY", "rest-only": "REST-ONLY",
            "one-way": "ONE-WAY", "direct": "ODD-conditioned (direct)", "odd": "ODD-conditioned",
            "odd-rest-descent": "ODD-conditioned (rest descent)"}[method]


def state_names(task):
    return ["STAND" if task == "stand" else "WALK", "DESCENDING", "REST", "GETTING_UP", "BRAKE"]


def _ema(bar, x, t):
    return x.clone() if t == 0 else (1 - ALPHA) * bar + ALPHA * x


def _count(c, cond):
    return th.where(cond, c + 1, th.zeros_like(c))


def rollout(sc, method, *, push=None, n=256, seed=0, checkpoints=None, record=False, render=False,
            trace=False, allow_return=True):
    """Run ``method`` on scenario ``sc`` with ``n`` robots. Returns survival S(t), success (walking), deaths by
    mode, and optionally trajectories (``record``), rendered frames (``render``) or per-step signals (``trace``).

    ``checkpoints`` overrides automaton policies ({"stand": <name or .zip>, ...}); ``allow_return=False``
    disables every return (calibration: it keeps settled robots in REST to sample the return gate's population).
    """
    assert method in METHODS, method
    push = push or sc.push
    switching, certified = method in SWITCHING, method in CERTIFIED
    funnel = method == "odd" and sc.descent == "funnel"
    walk = sc.task == "walk"

    env = sim.make_env(sim.WEIGHT_TASK, n, seed, render)
    tw = load(AUTOMATON, checkpoints)
    sim.endless(env)
    if sc.loose_termination:
        sim.loose_termination(env)
    leg = sim.LegFault(env) if sc.odd(0)[2] is not None else None
    residual = LegResidual(env) if sc.trigger == "residual" else None
    walker = Walker(n) if walk else None
    dstb = sim.lateral_push(env)
    obs = sim.reset(env, seed)
    d = sim.robot(env)
    if walk:
        yaw0 = yaw_of(d)
        goal = d.root_link_pos_w[:, :2].clone()
        goal[:, 0] += sc.goal * th.cos(yaw0)
        goal[:, 1] += sc.goal * th.sin(yaw0)
        hold = th.randint(0, sc.stagger, (n,), device=DEV) if sc.stagger else None

    zeros = lambda: th.zeros(n, device=DEV)                       # noqa: E731
    no = lambda: th.zeros(n, dtype=th.bool, device=DEV)            # noqa: E731
    st = th.full((n,), REST if method == "rest-only" else TASK, dtype=th.long, device=DEV)
    vs_bar, vu_bar, er_bar = zeros(), zeros(), zeros()
    trig_c, up_c, abort_c, direct_c = zeros(), zeros(), zeros(), zeros()
    settle, standok, brake_c, refr = zeros(), zeros(), zeros(), zeros()
    belief_c = 0
    alive, reached = th.ones(n, dtype=th.bool, device=DEV), no()
    t_goal = th.full((n,), float("nan"), device=DEV)
    t_trig = th.full((n,), float("nan"), device=DEV)
    stand_time, alive_time = zeros(), zeros()
    names = state_names(sc.task)
    deaths = {s: 0 for s in names}
    early = 0
    S, SUC = [], []
    rec = {"traj": [], "st": [], "alive": []}
    tr = {k: [] for k in ("vs_bar", "vu_bar", "er_bar", "st_pre", "alive_pre", "settle", "belief_ok",
                          "st", "alive", "W", "theta")}
    frames = []
    if render:
        tint = _Tint(env)

    up_target = GETTING_UP if (certified or sc.brake) else TASK
    trigger_target = BRAKE if sc.brake else (DESCENDING if certified else REST)
    for t in range(sc.steps):
        W, h, theta = sc.odd(t)
        if leg is not None:
            leg.set(theta)
        sim.set_load(env, W, h)
        sim.set_push(env, sim.push_scale(push, t))

        # ── signals ──
        vs_bar = _ema(vs_bar, tw["stand"].value(env), t)
        if certified:
            vu_bar = _ema(vu_bar, tw["getup"].value(env), t)
        if residual is not None:
            er_bar = _ema(er_bar, residual(), t)
        trig_c = _count(trig_c, {"value": lambda: vs_bar < sc.trigger_eps,
                                 "load": lambda: th.full((n,), W >= sc.trigger_eps, device=DEV),
                                 "residual": lambda: (er_bar > sc.trigger_eps) & (st == TASK)}[sc.trigger]())
        nominal = (W < LOAD_NOMINAL) if sc.belief == "load" else (theta >= LEG_HEALTHY)
        belief_c = belief_c + 1 if nominal else 0
        belief_ok = belief_c >= sc.belief_sustain
        settle = _count(settle, sim.in_rest_target(env))
        standok = _count(standok, sim.in_stance_target(env))
        brake_c = _count(brake_c, st == BRAKE)
        if certified:
            up_c = _count(up_c, vu_bar > sc.eps_up)
            abort_c = _count(abort_c, vu_bar < sc.eps_abort)
        if method == "direct":
            direct_c = _count(direct_c, vs_bar > DIRECT_EPS_UP)
        can = (refr <= 0) & (t >= sc.arm_after)
        if trace:
            for k, v in (("vs_bar", vs_bar), ("vu_bar", vu_bar), ("er_bar", er_bar), ("st_pre", st),
                         ("alive_pre", alive), ("settle", settle)):
                tr[k].append(v.clone())
            tr["belief_ok"].append(belief_ok); tr["W"].append(W); tr["theta"].append(theta)

        # ── transitions (evaluated on the pre-step mode, applied in order) ──
        go_trigger = (st == TASK) & (trig_c >= sc.trigger_k) & can if switching else no()
        slow = th.linalg.norm(d.root_link_lin_vel_b, dim=1) < BRAKE_SLOW
        go_descend = (st == BRAKE) & (slow | (brake_c >= BRAKE_CAP))
        go_rest = (st == DESCENDING) & (settle >= SETTLE)
        if certified:
            go_up = (st == REST) & (up_c >= K_UP) & can
            if sc.gate_return and not belief_ok:
                go_up = no()
            go_task = (st == GETTING_UP) & (standok >= SETTLE)
            go_abort = (st == GETTING_UP) & (abort_c >= K_ABORT)
        else:
            go_up = (st == REST) & (direct_c >= DIRECT_K_UP) & can if method == "direct" else no()
            go_task = st == GETTING_UP
            go_abort = no()
        if not allow_return:
            go_up = no()
        if t * DT < (sc.change_at if sc.change_at is not None else 0.0):
            early += int((go_trigger & alive).sum())
        t_trig = th.where(go_trigger & th.isnan(t_trig), th.full_like(t_trig, t * DT), t_trig)
        st = th.where(go_trigger, th.full_like(st, trigger_target), st)
        st = th.where(go_descend, th.full_like(st, DESCENDING), st)
        st = th.where(go_rest, th.full_like(st, REST), st)
        st = th.where(go_up, th.full_like(st, up_target), st)
        st = th.where(go_task, th.full_like(st, TASK), st)
        st = th.where(go_abort, th.full_like(st, DESCENDING), st)
        refr = th.where(go_trigger | go_up | go_abort, th.full_like(refr, REFRACTORY), refr - 1)

        # ── act: each robot runs the policy of its mode ──
        a_rest = tw["rest"].act(obs)
        a = a_rest
        if walk:
            walker.reset(go_task | (go_up if up_target == TASK else no()))
            off = reached | (st != TASK)
            cmd = steer_cmd(d, goal, off)
            if hold is not None:
                cmd[t < hold] = 0.0
            a_task = walker.act(env, cmd)
        else:
            a_task = tw["stand"].act(obs)
        a = th.where((st == TASK).unsqueeze(-1), a_task, a)
        if sc.brake is not None:
            a_brake = tw["stand"].act(obs) if sc.brake == "stand" else a_task
            a = th.where((st == BRAKE).unsqueeze(-1), a_brake, a)
        if funnel:
            a = th.where((st == DESCENDING).unsqueeze(-1), tw["descend"].act(obs), a)
        if certified:
            a = th.where((st == GETTING_UP).unsqueeze(-1), tw["getup"].act(obs), a)
        obs, *_ = env.step_tensor(th.cat([a, dstb], dim=1))

        # ── outcomes ──
        dead = sim.failed(env)
        for si, s in enumerate(names):
            deaths[s] += int((dead & alive & (st == si)).sum())
        alive &= ~dead
        S.append(float(alive.float().mean()))
        if walk:
            dist = th.linalg.norm(d.root_link_pos_w[:, :2] - goal, dim=1)
            new = (dist < GOAL_RADIUS) & alive & ~reached
            t_goal = th.where(new, th.full_like(t_goal, t * DT), t_goal)
            reached |= new
            SUC.append(float(reached.float().mean()))
        else:
            stand_time += (sim.standing(env) & alive).float()
            alive_time += alive.float()
        if trace:
            tr["st"].append(st.clone()); tr["alive"].append(alive.clone())
        if record and t % 4 == 0:
            rec["traj"].append(d.root_link_pos_w[:, :2].clone()); rec["st"].append(st.clone())
            rec["alive"].append(alive.clone())
        if render:
            tint(st, alive, W, theta)
            if t % 2 == 0:
                frames.append(np.asarray(env.render()))
    env.close()

    tt = t_trig[~th.isnan(t_trig)]
    out = {"scenario": sc.name, "method": method, "push": push, "n": n, "seed": seed,
           "S": S, "safe": S[-1], "deaths": deaths, "early_triggers": early,
           "trigger_median": float(tt.median()) if tt.numel() else None}
    if walk:
        tg = t_goal[reached]
        out.update(SUC=SUC, success=SUC[-1], t_goal_median=float(tg.median()) if tg.numel() else None)
    else:
        out["afford"] = float((stand_time / alive_time.clamp_min(1)).mean())
    if record:
        out["record"] = {"traj": th.stack(rec["traj"]).cpu().numpy(), "st": th.stack(rec["st"]).cpu().numpy(),
                         "alive": th.stack(rec["alive"]).cpu().numpy(), "every": 4}
        if walk:
            out["record"].update(spawn=out["record"]["traj"][0], yaw0=yaw0.cpu().numpy(), goal=goal.cpu().numpy(),
                                 reached=reached.cpu().numpy(), alive_end=alive.cpu().numpy())
    if render:
        out["frames"] = frames
    if trace:
        out["trace"] = {k: (th.stack(v).cpu() if v and th.is_tensor(v[0]) else v) for k, v in tr.items()}
    return out


class _Tint:
    """Colour the host robot's trunk by the majority mode of the live robots (per-env colours don't render):
    TASK grey (reddening with load, orange with leg derating), DESCENDING orange, REST blue, GETTING_UP green,
    BRAKE yellow."""
    GRAY, RED, ORANGE = np.array([.5, .5, .5, 1.]), np.array([.85, .1, .1, 1.]), np.array([.95, .55, .1, 1.])
    MODE = {DESCENDING: np.array([.90, .49, .13, 1.]), REST: np.array([.15, .35, .90, 1.]),
            GETTING_UP: np.array([.12, .52, .29, 1.]), BRAKE: np.array([.95, .85, .2, 1.])}

    def __init__(self, env):
        import mujoco
        self.mm = env.mj.sim.mj_model
        base = mujoco.mj_name2id(self.mm, mujoco.mjtObj.mjOBJ_BODY, "robot/base_link")
        self.trunk = [g for g in range(self.mm.ngeom) if self.mm.geom_bodyid[g] == base]

    def __call__(self, st, alive, W, theta):
        maj = int(th.mode(st[alive]).values) if alive.any() else TASK
        if maj == TASK:
            c = (self.GRAY + (self.ORANGE - self.GRAY) * (1.0 - theta) if theta is not None
                 else self.GRAY + (self.RED - self.GRAY) * min(1.0, W / 250.0))
        else:
            c = self.MODE[maj]
        for g in self.trunk:
            self.mm.geom_rgba[g] = c
