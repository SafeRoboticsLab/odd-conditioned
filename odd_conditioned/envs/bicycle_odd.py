"""Bicycle5D with a varying ODD — the quick "does vanilla ReachAvoidSAC already solve dynamic ODD?"
test env. Subclasses the vendored released vehicle; does NOT modify safety-stable-baselines.

THE ODD: a scalar CONTROL-AUTHORITY multiplier `c ∈ [c_lo, c_hi]`. The commanded control is scaled
`u_eff = c · u` before the dynamics. c=1 → full authority; c=0.4 → sluggish car (weak accel AND weak
steering rate) — degraded actuators / low grip. Low c ⇒ longer stopping/turning ⇒ the car must be
more cautious ⇒ a SMALLER reach-avoid set. Non-inert (unlike the bare steering limit): braking
authority sets stopping distance, which is what reach-avoid past an obstacle turns on.

BLIND BY DESIGN: the observation is UNCHANGED from the base env — the car is NOT told `c`. That is
the point. "Can vanilla ReachAvoidSAC solve dynamic ODD *implicitly*" = can a memoryless policy
trained across a distribution of `c` reach-avoid at every `c` without being told which one it's in?
(Predicted: no — it must either play worst-case-c conservative or over-drive at low c and collide.
This is the bicycle/reach-avoid analogue of E008c's `blind` arm, on the actual target env.)

DYNAMIC: `resample_prob > 0` re-rolls `c` mid-episode (a payload shift / ice patch), so the ODD
genuinely *changes during a run* — the dynamic-ODD condition, not just per-episode randomization.
"""
from typing import Optional, Tuple

import numpy as np

from odd_conditioned.envs import _bicycle5d_vendored as bk


class BicycleGoalODD(bk.BicycleGoal):
    def __init__(
        self,
        c_range: Tuple[float, float] = (0.4, 1.0),   # control-authority multiplier range
        fixed_c: Optional[float] = None,             # pin c (specialist / eval at one ODD)
        resample_prob: float = 0.0,                  # per-step mid-episode ODD change (dynamic ODD)
        expose_odd: bool = False,                    # False = blind (the implicit test); True = oracle
        seed: Optional[int] = None,
        **kw,
    ) -> None:
        super().__init__(**kw)
        self.c_range, self.fixed_c = c_range, fixed_c
        self.resample_prob, self.expose_odd = resample_prob, expose_odd
        self._crng = np.random.default_rng(seed)
        self.c = 1.0
        if expose_odd:                               # oracle variant: append normalized c to obs
            import gymnasium as gym
            lo = np.concatenate([self.observation_space.low, [-np.inf]]).astype(np.float32)
            hi = np.concatenate([self.observation_space.high, [np.inf]]).astype(np.float32)
            self.observation_space = gym.spaces.Box(lo, hi, dtype=np.float32)

    def _sample_c(self) -> None:
        self.c = self.fixed_c if self.fixed_c is not None else float(self._crng.uniform(*self.c_range))

    def _c_norm(self) -> float:
        lo, hi = self.c_range
        return 2 * (self.c - lo) / (hi - lo + 1e-9) - 1     # [c_lo,c_hi] -> [-1,1]

    def _augment(self, obs):
        return np.concatenate([obs, [self._c_norm()]]).astype(np.float32) if self.expose_odd else obs

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._crng = np.random.default_rng(seed)
        self._sample_c()
        obs, info = super().reset(seed=seed, options=options)
        info["c"] = self.c
        return self._augment(obs), info

    def step(self, action):
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        # scale ONLY the control channels by c; leave any (unused, single-agent) tail untouched
        a = a.copy()
        a[:2] = self.c * a[:2]
        if self._crng.random() < self.resample_prob:       # the ODD may change mid-run
            self._sample_c()
        obs, g, term, trunc, info = super().step(a)
        info["c"] = self.c
        return self._augment(obs), g, term, trunc, info
