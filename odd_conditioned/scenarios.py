"""The evaluation scenarios: a task, a time-varying ODD, and how the safety filter is wired for that regime.

    standing-{square,sine}       standing under a load that keeps switching / oscillating, 20 s
    standing-{pulse,period}      standing through ONE load excursion (a step, or one sine period), 24 s
    payload-{pulse,period,dip}   walking to a goal 12 m away while a tall crate is loaded and unloaded, 40 s
    leg-fault                    walking to a goal 11 m away while the front-right leg derates and recovers, 30 s
    weight-walk-{pulse,period}   walking under a 220 N load excursion — the boundary case, 30 s

Each scenario fixes the descent trigger (which signal says "this ODD is leaving the mode's certified set"), the
return gate (the belief that the nominal ODD is back, and the get-up certificate thresholds), whether walking
brakes first, and the descent policy of the full method. Thresholds are calibrated per regime by
scripts/calibrate.py (docs/TRAINING.md, "Calibrating the switching thresholds").
"""
import math
from dataclasses import dataclass, field, replace
from typing import Callable, Optional

from .sim import DT, LOAD_H


@dataclass(frozen=True)
class Scenario:
    name: str
    title: str
    task: str                      # "stand" | "walk"
    steps: int
    odd: Callable                  # step -> (W [N], h [m], theta or None)
    trigger: str                   # descent trigger channel: "value" (V̄_stand < eps) | "load" (W >= eps) |
                                   # "residual" (torque-saturation residual > eps, while walking)
    trigger_eps: float
    trigger_k: int                 # consecutive steps the trigger condition must hold
    eps_up: float                  # return certificate: V̄_up > eps_up for K_UP steps
    eps_abort: float               # get-up abort: V̄_up < eps_abort for K_ABORT steps
    belief: str                    # what the return gate's belief reads: "load" (W < 130 N) | "leg" (θ >= 0.99)
    belief_sustain: int            # steps the belief must read nominal before the gate opens
    gate_return: bool = True       # False: the certified return ignores the belief (load waves)
    brake: Optional[str] = None    # walking only: stop before descending, with "stand" (STAND expert) | "walker"
    descent: str = "funnel"        # descent policy of method "odd": "funnel" (descend) | "rest"
    arm_after: int = 50            # no switch fires during the first steps (spawn transient)
    loose_termination: bool = False   # death = flip-over only (> 80 deg); contact termination off
    stagger: int = 0               # walking starts held back U{0, ..., stagger-1} steps
    goal: Optional[float] = None   # walking: goal distance along the initial heading (m)
    change_at: Optional[float] = None   # when the ODD starts changing (s); triggers before it are false alarms
    push: str = "benign"
    params: dict = field(default_factory=dict)   # the schedule's constants, for figures and docs

    def with_(self, **kw):
        return replace(self, **kw)


# ── standing: STAND expert as the task policy ────────────────────────────────────────────────────────────

def _standing_wave(kind):
    def odd(t):
        s = t * DT
        if kind == "square":                                     # 40 N <-> 220 N every 3 s
            return (40.0 if (int(s // 3.0) % 2 == 0) else 220.0), LOAD_H, None
        return 130.0 + 110.0 * math.sin(2 * math.pi * s / 8.0), LOAD_H, None   # 20..240 N, 8 s period
    return odd


def _standing_single(kind):
    def odd(t):
        s = t * DT
        if kind == "pulse":                                      # light | heavy | light, 8 s each
            return (220.0 if 8.0 <= s < 16.0 else 40.0), LOAD_H, None
        return 130.0 - 110.0 * math.cos(2 * math.pi * s / 24.0), LOAD_H, None  # one period, 20 -> 240 -> 20 N
    return odd


_STANDING = dict(task="stand", trigger="value", trigger_eps=-0.05, trigger_k=5, eps_up=0.10, eps_abort=-0.02,
                 belief="load", belief_sustain=1, descent="funnel", arm_after=50)

# ── payload walk: a tall crate (W, h) loaded at t = 5 s, unloaded by t = 13 s ────────────────────────────

PAYLOAD = dict(t0=5.0, t1=13.0, W_lo=0.0, W_hi=220.0, h_lo=0.25, h_hi=0.35, dip_at=9.7, dip_depth=0.75)


def _payload(kind):
    p = PAYLOAD

    def odd(t):
        s = t * DT
        if not (p["t0"] <= s < p["t1"]):
            return p["W_lo"], p["h_lo"], None
        if kind == "pulse":
            return p["W_hi"], p["h_hi"], None
        a = math.sin(math.pi * (s - p["t0"]) / (p["t1"] - p["t0"])) ** 2
        W = p["W_lo"] + (p["W_hi"] - p["W_lo"]) * a
        if kind == "dip":      # the load briefly lightens mid-window: below the 130 N gate for ~1.6 s
            W = W * (1.0 - p["dip_depth"] * math.exp(-(((s - p["dip_at"]) / 1.0) ** 2)))
        h = p["h_lo"] + (p["h_hi"] - p["h_lo"]) * (W - p["W_lo"]) / (p["W_hi"] - p["W_lo"])
        return W, h, None
    return odd


# ── leg-fault walk: the front-right motors derate 1.0 -> 0.15 over 5-8 s, stay derated, heal at 15 s ────────

LEG = dict(t_fault=5.0, t_derated=8.0, t_heal=15.0, theta_lo=0.15)


def _leg_fault(t):
    p, s = LEG, t * DT
    if s < p["t_fault"]:
        theta = 1.0
    elif s < p["t_derated"]:
        theta = 1.0 + (p["theta_lo"] - 1.0) * (s - p["t_fault"]) / (p["t_derated"] - p["t_fault"])
    elif s < p["t_heal"]:
        theta = p["theta_lo"]
    else:
        theta = 1.0
    return 0.0, LOAD_H, theta


# ── weight walk: a 220 N load at h = 0.25 m during [5, 13) s ────────────────────────────────────────────────

def _weight_walk(kind):
    def odd(t):
        s = t * DT
        if kind == "pulse":
            return (220.0 if 5.0 <= s < 13.0 else 0.0), LOAD_H, None
        return (220.0 * math.sin(math.pi * (s - 5.0) / 8.0) ** 2 if 5.0 <= s < 13.0 else 0.0), LOAD_H, None
    return odd


SCENARIOS = {}
for _k in ("square", "sine"):
    SCENARIOS[f"standing-{_k}"] = Scenario(
        f"standing-{_k}", f"standing, load {_k} wave (40-220 N)", steps=1000, odd=_standing_wave(_k),
        gate_return=False, **_STANDING)
for _k in ("pulse", "period"):
    SCENARIOS[f"standing-{_k}"] = Scenario(
        f"standing-{_k}", f"standing, single load {_k} (40-220 N)", steps=1200, odd=_standing_single(_k),
        gate_return=True, **_STANDING)
for _k in ("pulse", "period", "dip"):
    SCENARIOS[f"payload-{_k}"] = Scenario(
        f"payload-{_k}", f"payload-swap walk, {_k}", task="walk", steps=2000, odd=_payload(_k),
        trigger="load", trigger_eps=60.0, trigger_k=3, eps_up=-0.30, eps_abort=-0.45,
        belief="load", belief_sustain=100, brake="stand", descent="funnel", arm_after=100,
        loose_termination=True, stagger=30, goal=12.0, change_at=PAYLOAD["t0"], params=PAYLOAD)
SCENARIOS["leg-fault"] = Scenario(
    "leg-fault", "leg-fault walk (FR leg derates to 0.15, then heals)", task="walk", steps=1500, odd=_leg_fault,
    trigger="residual", trigger_eps=0.03, trigger_k=10, eps_up=-0.35, eps_abort=-0.45,
    belief="leg", belief_sustain=25, brake="walker", descent="rest", arm_after=100,
    goal=11.0, change_at=LEG["t_fault"], params=LEG)
for _k in ("pulse", "period"):
    SCENARIOS[f"weight-walk-{_k}"] = Scenario(
        f"weight-walk-{_k}", f"weight walk, 220 N {_k}", task="walk", steps=1500, odd=_weight_walk(_k),
        trigger="value", trigger_eps=-0.15, trigger_k=10, eps_up=0.10, eps_abort=-0.02,
        belief="load", belief_sustain=1, descent="rest", arm_after=50, goal=12.0, change_at=5.0)
