"""Certificate experiments: does a mode's learned value V_stand know when its specification stops being certifiable?

    value_sweep     hold the ODD fixed, warm up under the STAND policy, read V_stand (and other twins) at the
                    reached states; sweep the ODD axis. A usable certificate CONTRACTS across the axis, measured
                    as discrimination = |mean V(certifiable band) - mean V(failed band)| / mean sigma.
    handoff_ramp    a demo ramp of the ODD with gusts: STAND policy until EMA V_stand < eps for HYST steps, then a
                    permanent handoff to the REST policy; compared to fixed-mode and single-spec baselines.
    forced_switch   the same ramp with the handoff forced at a fixed load: maps the safe switching window.
    region_grid     failure fraction of each mode's policy over an (ODD x push) grid: the certifiable regions.
    compound_matrix the compound STAND/REST policies at fixed leg derating θ while carrying 80 N.

Three ODD axes: the carried load W (weight ladder), the front-right leg torque fraction θ unloaded (leg — the
negative control: training absorbs it, the certificate stays flat), and θ while carrying 80 N (compound).
"""
from dataclasses import dataclass, field
from typing import Optional

import torch as th

from . import sim
from .policies import Twin, load
from .sim import DEV, DT, LOAD_H, WEIGHT_TASK

PULL = 0.20   # ambient lateral push of every certificate experiment: 10 N


# ── value sweeps ──────────────────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Sweep:
    name: str
    task: str
    readouts: tuple                # twins whose value is read; the first one is also the policy that flies
    axis: str                      # "W" (load, N) | "theta" (FR-leg torque fraction)
    values: tuple
    cert: tuple                    # axis values where standing is certifiable
    fail: tuple                    # axis values where standing fails
    load: Optional[tuple] = None   # constant (W, h) carried on a θ sweep
    n: int = 128
    warm: int = 150


SWEEPS = {
    "weight": Sweep("weight", WEIGHT_TASK, ("stand", "stand_wide"), "W", (0, 30, 60, 90, 120, 150, 200),
                    cert=(0, 30, 60, 90, 120), fail=(200,)),
    "weight-wide": Sweep("weight-wide", WEIGHT_TASK, ("stand_wide", "unified"), "W",
                         (0, 30, 60, 90, 120, 150, 200, 250), cert=(0, 30, 60, 90, 120, 150), fail=(200, 250)),
    "leg": Sweep("leg", "go2_leg_rest", ("leg_stand",), "theta", (1.0, 0.8, 0.6, 0.5, 0.4, 0.2, 0.1, 0.0),
                 cert=(1.0, 0.8, 0.6, 0.5), fail=(0.1, 0.0)),
    "compound": Sweep("compound", "go2_compound_rest", ("compound_stand",), "theta",
                      (1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0), cert=(1.0, 0.8, 0.6, 0.5), fail=(0.2, 0.1, 0.0),
                      load=(80.0, LOAD_H)),
}


def value_sweep(sw: Sweep, seed: int = 0) -> dict:
    env = sim.make_env(sw.task, sw.n, seed)
    tw = load(sw.readouts)
    policy = tw[sw.readouts[0]]
    leg = sim.LegFault(env) if sw.axis == "theta" else None
    dstb = sim.lateral_push(env)
    V = {k: {"mean": [], "std": []} for k in sw.readouts}
    stand_frac = []
    for x in sw.values:
        obs = sim.reset(env, seed)
        stood = th.zeros(sw.n, device=DEV)
        for _ in range(sw.warm):
            if sw.axis == "W":
                sim.set_load(env, float(x), LOAD_H)
            else:
                leg.set(x)
                if sw.load:
                    sim.set_load(env, *sw.load)
            sim.set_push(env, PULL)
            obs, *_ = env.step_tensor(th.cat([policy.act(obs), dstb], dim=1))
            stood += sim.standing(env).float()
        for k, twin in tw.items():
            v = twin.value(env)
            V[k]["mean"].append(float(v.mean()))
            V[k]["std"].append(float(v.std()))
        stand_frac.append(float(stood.mean()) / sw.warm)
    env.close()
    for k, r in V.items():
        at = dict(zip(sw.values, r["mean"]))
        r["cert_mean"] = sum(at[x] for x in sw.cert) / len(sw.cert)
        r["fail_mean"] = sum(at[x] for x in sw.fail) / len(sw.fail)
        r["eps"] = 0.5 * (r["cert_mean"] + r["fail_mean"])
        r["mean_std"] = sum(r["std"]) / len(r["std"])
        r["discrim"] = abs(r["cert_mean"] - r["fail_mean"]) / max(r["mean_std"], 1e-6)
    return {"name": sw.name, "task": sw.task, "policy": sw.readouts[0], "axis": sw.axis, "values": list(sw.values),
            "cert_band": list(sw.cert), "fail_band": list(sw.fail), "load": sw.load, "pull_N": PULL * sim.FORCE_MAX,
            "n": sw.n, "seed": seed, "stand_frac": stand_frac, "readouts": V}


def print_sweep(r):
    ks = list(r["readouts"])
    print(f"value sweep '{r['name']}': policy {r['policy']} on {r['task']}, {r['axis']} axis")
    print(f"{r['axis']:>6} " + " ".join(f"{'V_' + k:>20}" for k in ks) + f" {'stand%':>7}")
    for i, x in enumerate(r["values"]):
        print(f"{x:>6} " + " ".join(f"{r['readouts'][k]['mean'][i]:>+11.4f} ±{r['readouts'][k]['std'][i]:.4f}"
                                    for k in ks) + f" {r['stand_frac'][i]:>7.2f}")
    for k in ks:
        q = r["readouts"][k]
        print(f"  V_{k}: certifiable {q['cert_mean']:+.4f}  failed {q['fail_mean']:+.4f}  -> eps (midpoint) "
              f"{q['eps']:+.4f}  discrimination {q['discrim']:.2f}")


# ── handoff ramps ─────────────────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Ramp:
    name: str
    task: str
    axis: str                      # "W" | "theta"
    lo: float
    hi: float
    start: int                     # ramp from lo at step `start` to hi at step `end`, then hold
    end: int
    gusts: tuple                   # gust start steps
    readout: str                   # the trigger's value twin
    eps: float
    hyst: int
    alpha: float
    settle: int                    # no trigger before this step (spawn transient and the first gust)
    arms: dict                     # arm -> (stand-phase policy, rest-phase policy, rule: none | always | value)
    traced: tuple = ()             # twins whose mean value is traced every step (besides the readout)
    load: Optional[tuple] = None   # constant (W, h) carried on a θ ramp
    n: int = 128
    steps: int = 500
    gust_len: int = 25
    ambient: float = 0.20          # 10 N
    gust: float = 0.70             # 35 N during a gust

    def odd_at(self, t):
        if t < self.start:
            return self.lo
        if t < self.end:
            return self.lo + (t - self.start) / (self.end - self.start) * (self.hi - self.lo)
        return self.hi

    def in_gust(self, t):
        return any(g <= t < g + self.gust_len for g in self.gusts)


_W_RAMP = dict(task=WEIGHT_TASK, axis="W", lo=0.0, hi=250.0, start=100, end=400, gusts=(75, 250, 325, 425),
               hyst=8, alpha=0.10, settle=130)
RAMPS = {
    "weight-wide": Ramp("weight-wide", readout="stand_wide", eps=-0.04, traced=("unified",), arms={
        "STAND-ONLY": ("stand_wide", "stand_wide", "none"), "REST-ONLY": ("rest", "rest", "always"),
        "HANDOFF": ("stand_wide", "rest", "value"), "UNIFIED": ("unified", "unified", "none"),
        "UNIFIED-DISC": ("unified_discounted", "unified_discounted", "none")}, **_W_RAMP),
    "weight": Ramp("weight", readout="stand", eps=-0.05, arms={
        "STAND-ONLY": ("stand", "stand", "none"), "HANDOFF": ("stand", "rest", "value")}, **_W_RAMP),
    "compound": Ramp("compound", task="go2_compound_rest", axis="theta", lo=1.0, hi=0.1, start=100, end=300,
                     gusts=(75, 325, 400), readout="compound_stand", eps=-0.14, hyst=6, alpha=0.15, settle=110,
                     load=(80.0, LOAD_H), n=256, arms={
                         "STAND-ONLY": ("compound_stand", "compound_stand", "none"),
                         "REST-ONLY": ("compound_rest", "compound_rest", "always"),
                         "HANDOFF": ("compound_stand", "compound_rest", "value")}),
}


def handoff_arm(rp: Ramp, stand: str, rest: str, rule, *, settle=None, seed: int = 0, n=None, on_step=None) -> dict:
    """One arm of a ramp. ``rule``: "none" | "always" | "value" | ("forced", W): hand off once the load reaches W.
    ``on_step(env, t, x, switched)`` is called after every step (videos render through it)."""
    settle = rp.settle if settle is None else settle
    n = rp.n if n is None else n
    env = sim.make_env(rp.task, n, seed, render=on_step is not None)
    tw = load(dict.fromkeys((stand, rest, rp.readout, *rp.traced)))
    leg = sim.LegFault(env) if rp.axis == "theta" else None
    dstb = sim.lateral_push(env)
    obs = sim.reset(env, seed)
    switched = th.full((n,), rule == "always", dtype=th.bool, device=DEV)
    below = th.zeros(n, device=DEV)
    ema = None
    switch_step = th.full((n,), -1, dtype=th.long, device=DEV)
    switch_at = th.full((n,), -1.0, device=DEV)
    stood, stood_pre = th.zeros(n, device=DEV), th.zeros(n, device=DEV)
    tip, slam = th.zeros(n, dtype=th.bool, device=DEV), th.zeros(n, dtype=th.bool, device=DEV)
    V_trace = {k: [] for k in dict.fromkeys((rp.readout, *rp.traced))}
    h_trace, sw_trace = [], []
    for t in range(rp.steps):
        x = rp.odd_at(t)
        if rp.axis == "W":
            W = x
            sim.set_load(env, W, LOAD_H)
        else:
            W = rp.load[0]
            leg.set(x)
            sim.set_load(env, *rp.load)
        sim.set_push(env, rp.gust if rp.in_gust(t) else rp.ambient)
        for k in V_trace:
            v = tw[k].value(env)
            V_trace[k].append(float(v.mean()))
            if k == rp.readout:
                Vs = v
        ema = Vs.clone() if ema is None else rp.alpha * Vs + (1 - rp.alpha) * ema
        if rule == "value":
            below = th.where((ema < rp.eps) & (t >= settle), below + 1.0, th.zeros_like(below))
            new = (below >= rp.hyst) & ~switched
        elif isinstance(rule, tuple):
            new = (th.full((n,), W, device=DEV) >= rule[1]) & ~switched & (t >= settle)
        else:
            new = th.zeros(n, dtype=th.bool, device=DEV)
        switch_step[new] = t
        switch_at[new] = x
        switched |= new
        a = th.where(switched.unsqueeze(-1), tw[rest].act(obs), tw[stand].act(obs))
        obs, *_ = env.step_tensor(th.cat([a, dstb], dim=1))
        up = sim.standing(env)
        stood += up.float()
        stood_pre += (up & ~switched).float()
        slam |= sim.nonfoot_impact(env) > sim.slam_cap(W)
        tip |= sim.tipped(env)
        h_trace.append(float(sim.robot(env).root_link_pos_w[:, 2].mean()))
        sw_trace.append(float(switched.float().mean()))
        if on_step is not None:
            on_step(env, t, x, switched)
    h_end = float(sim.robot(env).root_link_pos_w[:, 2].mean())
    env.close()
    out = {"afford_pre": float(stood_pre.mean()) / rp.steps, "stand_frac": float(stood.mean()) / rp.steps,
           "tip": float(tip.float().mean()), "slam": float(slam.float().mean()), "h_end": h_end,
           "switched_frac": float(switched.float().mean()), "V_trace": V_trace, "h_trace": h_trace,
           "sw_trace": sw_trace}
    did = switch_step >= 0
    if did.any():
        out.update(switch_at_mean=float(switch_at[did].mean()), switch_at_std=float(switch_at[did].std()),
                   switch_t_mean=float(switch_step[did].float().mean()) * DT,
                   switch_t_std=float(switch_step[did].float().std()) * DT, switch_frac=float(did.float().mean()))
    return out


def handoff_ramp(rp: Ramp, seed: int = 0) -> dict:
    arms = {name: handoff_arm(rp, *spec, seed=seed) for name, spec in rp.arms.items()}
    return {"name": rp.name, "task": rp.task, "axis": rp.axis, "seed": seed, "n": rp.n, "dt": DT,
            "odd_trace": [rp.odd_at(t) for t in range(rp.steps)],
            "config": {"lo": rp.lo, "hi": rp.hi, "start": rp.start, "end": rp.end, "gusts": list(rp.gusts),
                       "gust_len": rp.gust_len, "ambient_N": rp.ambient * sim.FORCE_MAX,
                       "gust_N": rp.gust * sim.FORCE_MAX, "readout": rp.readout, "eps": rp.eps, "hyst": rp.hyst,
                       "alpha": rp.alpha, "settle": rp.settle, "load": rp.load},
            "arms": arms}


def print_ramp(r):
    ax = "W" if r["axis"] == "W" else "θ"
    print(f"handoff ramp '{r['name']}' ({ax} {r['config']['lo']}→{r['config']['hi']}, eps {r['config']['eps']})")
    print(f"{'arm':>13} {'afford':>7} {'stand%':>7} {'tip':>5} {'slam':>5} {'h_end':>6} "
          f"{'switch ' + ax:>12} {'t_sw':>5} {'fired':>6}")
    for arm, a in r["arms"].items():
        sw = (f"{a['switch_at_mean']:>6.2f}±{a['switch_at_std']:<5.2f}" if "switch_at_mean" in a else f"{'-':>12}")
        print(f"{arm:>13} {a['afford_pre']:>7.2f} {a['stand_frac']:>7.2f} {a['tip']:>5.2f} {a['slam']:>5.2f} "
              f"{a['h_end']:>6.2f} {sw:>12} {a.get('switch_t_mean', float('nan')):>5.2f} "
              f"{a.get('switch_frac', 0.0):>6.2f}")


FORCED_AT = (1, 40, 80, 120, 160, 200, 240, 9999)


def forced_switch(seed: int = 0) -> dict:
    """Hand off at a FIXED load on the wide-stand weight ramp (settle 100): where switching is mechanically safe.
    Places the demo triggers' eps: it should fire inside the window where tip and slam stay low."""
    rp = RAMPS["weight-wide"]
    rows = {}
    for w in FORCED_AT:
        r = handoff_arm(rp, "stand_wide", "rest", ("forced", float(w)), settle=100, seed=seed)
        rows[w] = {k: r[k] for k in ("afford_pre", "tip", "slam", "h_end", "switched_frac")}
        rows[w]["switch_at_mean"] = r.get("switch_at_mean")
    return {"name": "forced-switch", "ramp": rp.name, "settle": 100, "seed": seed, "rows": rows}


# ── certifiable-region grid ───────────────────────────────────────────────────────────────────────────────

GRID = {
    "weight": dict(task=WEIGHT_TASK, policies=("stand", "rest"), axis="W",
                   values=(0, 30, 60, 90, 120, 150, 180, 210, 240)),
    "compound": dict(task="go2_compound_rest_at_100", policies=("compound_stand", "compound_rest"), axis="theta",
                     values=(1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1), load=(80.0, LOAD_H)),
}
GRID_PUSH = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)   # x 50 N = 5..35 N
GRID_N, GRID_STEPS = 96, 250


def region_grid_policy(panel: str, policy: str, seed: int = 0) -> list:
    """Failure fraction (tip or slam) of ``policy`` per (ODD value, push) cell: rows = pushes, cols = values."""
    g = GRID[panel]
    env = sim.make_env(g["task"], GRID_N, seed)
    twin = Twin(policy)
    leg = sim.LegFault(env)
    dstb = sim.lateral_push(env)
    out = [[None] * len(g["values"]) for _ in GRID_PUSH]
    for j, x in enumerate(g["values"]):
        W, theta = (float(x), None) if g["axis"] == "W" else (g["load"][0], x)
        for i, push in enumerate(GRID_PUSH):
            sim.set_push(env, push)
            obs = sim.reset(env, seed)
            fail = th.zeros(GRID_N, dtype=th.bool, device=DEV)
            for _ in range(GRID_STEPS):
                sim.set_load(env, W, LOAD_H)
                if theta is not None:
                    leg.set(theta)
                obs, *_ = env.step_tensor(th.cat([twin.act(obs), dstb], dim=1))
                fail |= sim.tipped(env) | (sim.nonfoot_impact(env) > sim.slam_cap(W))
            out[i][j] = float(fail.float().mean())
        print(f"  {panel}/{policy} {g['axis']}={x}: " + " ".join(f"{out[i][j]:.2f}" for i in range(len(GRID_PUSH))),
              flush=True)
    env.close()
    return out


def region_grid(seed: int = 0) -> dict:
    res = {"push_N": [p * sim.FORCE_MAX for p in GRID_PUSH], "seed": seed, "panels": {}}
    for panel, g in GRID.items():
        res["panels"][panel] = {"axis": g["axis"], "values": list(g["values"]), "load": g.get("load"),
                                "fail": {p: region_grid_policy(panel, p, seed) for p in g["policies"]}}
    return res


# ── compound matrix ───────────────────────────────────────────────────────────────────────────────────────

MATRIX_THETAS = (1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0)
MATRIX_N, MATRIX_STEPS, MATRIX_GUST = 128, 300, (150, 175)


def compound_cell(policy: str, theta: float, seed: int = 0) -> dict:
    """``policy`` at a fixed FR-leg θ while carrying 80 N, 10 N push plus one 35 N gust at t = 3 s, 6 s."""
    W = 80.0
    env = sim.make_env("go2_compound_rest", MATRIX_N, seed)
    twin = Twin(policy)
    leg = sim.LegFault(env)
    dstb = sim.lateral_push(env)
    obs = sim.reset(env, seed)
    stood = th.zeros(MATRIX_N, device=DEV)
    tip, slam = th.zeros(MATRIX_N, dtype=th.bool, device=DEV), th.zeros(MATRIX_N, dtype=th.bool, device=DEV)
    for t in range(MATRIX_STEPS):
        leg.set(theta)
        sim.set_load(env, W, LOAD_H)
        sim.set_push(env, 0.70 if MATRIX_GUST[0] <= t < MATRIX_GUST[1] else PULL)
        obs, *_ = env.step_tensor(th.cat([twin.act(obs), dstb], dim=1))
        stood += sim.standing(env).float()
        slam |= sim.nonfoot_impact(env) > sim.slam_cap(W)
        tip |= sim.tipped(env)
    h_end = float(sim.robot(env).root_link_pos_w[:, 2].mean())
    env.close()
    return {"stand": float(stood.mean()) / MATRIX_STEPS, "tip": float(tip.float().mean()),
            "slam": float(slam.float().mean()), "h_end": h_end}


def compound_matrix(seed: int = 0) -> dict:
    out = {"thetas": list(MATRIX_THETAS), "seed": seed, "policies": {}}
    for p in ("compound_stand", "compound_rest"):
        out["policies"][p] = [compound_cell(p, th_, seed) for th_ in MATRIX_THETAS]
        print(f"  {p}: " + "  ".join(f"θ={t_}: stand {c['stand']:.2f} tip {c['tip']:.2f}"
                                     for t_, c in zip(MATRIX_THETAS, out["policies"][p])), flush=True)
    return out
