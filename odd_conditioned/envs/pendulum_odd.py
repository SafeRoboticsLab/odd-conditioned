"""ODD-conditioned inverted pendulum — a safety_sb3-contract env with a varying ODD.

WHY THIS LIVES HERE AND NOT IN robot-safety-sandbox
---------------------------------------------------
Buzi (2026-07-16): "create the environment here for now. Once we figure out the training
algorithm and how to formulate the change of ODD into g_x and l_x, we will attempt to use the
sandbox later." The sandbox is mjlab/GPU (its TaskSpec.cfg_builder returns a
ManagerBasedRlEnvCfg); this is a 2-D ODE. Porting a toy analytic pendulum into mjlab would be
overkill, and the open question -- how ODD change enters g_x / l_x -- should be answered by
E005's result before we commit an interface to the shared repo.

`safety-stable-baselines` is consumed as a PINNED GIT RELEASE and never modified (Buzi's
constraint). This env implements its contract; it does not touch it.

PHYSICS — identical to E003/E005, so the learned critic is scored against real ground truth
--------------------------------------------------------------------------------------------
PBF Example 1 / the lab's R-CBF pendulum (Alan, Molnar, Ames, Orosz, L-CSS 2023, Eq. 14):

    theta_ddot = (g/l) sin(theta) + u/(m l^2) + F/(m l) cos(theta)

theta = angle from UPRIGHT (unstable). Failure: |theta| > pi/3.

THE ODD
-------
`static` : m is constant, drawn per episode from [m_lo, m_hi].       -> E003's family.
`sine`   : m(phi) = m0 + A sin(phi), phi_dot = w.                    -> E005's family.
           The ODD parameters (A, w) are drawn per episode; the PHASE is a state variable.

THE ODD IS RESAMPLED **WITHIN** EPISODES (`resample_prob`), not only at reset.
Per-episode sampling teaches the critic the slice family; only an ODD that shifts mid-episode
teaches it the HANDOFF -- which is the whole point, since this is a runtime filter and
E001 showed 25-33% of states are stranded by an ODD change (detection is too late). This is
also Walk These Ways' anti-collapse device (resample the mode within episodes).

THE safety_sb3 CONTRACT (see its README; getting this wrong silently breaks learning)
------------------------------------------------------------------------------------
  g(s) -- SAFETY margin, rides on the REWARD channel. g >= 0 iff outside the failure set.
          The env MUST terminate when g < 0.
  l(s) -- TARGET margin, rides on info["l_x"]. l >= 0 iff inside the target.
  NEVER normalize rewards: the reward IS the margin, so VecNormalize(norm_reward=True) moves
  the value function's zero level set -- i.e. the safety boundary itself.
  Terminal anchor is g, NOT min(g, l) -- except the TIMEOUT terminal, which must be min(g, l)
  or safely loitering forever is value-maximizing (a 94M-step run converged to 0% reach).
"""

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# --- PBF Eq. 14 / lab R-CBF pendulum -----------------------------------------------------
G, L = 10.0, 1.0
DT = 0.025
THETA_MAX = np.pi / 3          # |theta| beyond this = fallen
OMEGA_LIM = 6.5                # grid/obs bound; also a hard episode bound
U_MAX = 20.0                   # torque (Nm)
F_BAR = 2.0                    # disturbance force bound (N)

M0_SINE = 5.0                  # sine mode: mean mass. A in [0,3] => m in [2,8] = E003's range
MASS_RANGE = (2.0, 8.0)        # static mode


class PendulumODD(gym.Env):
    """ODD-conditioned pendulum with the safety_sb3 (g, l) contract.

    Observation (obs_mode):
      'oracle'  -> [sin th, cos th, omega, <odd features>]   -- the ODD is HANDED to the policy.
                   The permissiveness UPPER BOUND: tests representation only.
      'blind'   -> [sin th, cos th, omega]                   -- no ODD info at all.
                   The worst-case floor.
      'history' -> [sin th, cos th, omega] x history_len     -- the ODD must be INFERRED from
                   the observation trace, RMA-style. The deployable case.

    oracle - history = the cost of not knowing, which is where the danger lives: an encoder
    that is confidently wrong OOD makes the filter under-conservative exactly when the ODD
    just changed.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        odd_mode: str = "static",          # 'static' | 'sine'
        obs_mode: str = "oracle",          # 'oracle' | 'blind' | 'history'
        margin_mode: str = "avoid",        # 'avoid' | 'reach_avoid'  -- see target_margin()
        l_neg: float = -1.0,
        l_scale: float = 1.0,              # the RISK DIAL. See target_margin().
        history_len: int = 8,
        mass_range: Tuple[float, float] = MASS_RANGE,
        A_range: Tuple[float, float] = (0.0, 3.0),
        w_range: Tuple[float, float] = (0.15, 25.0),
        resample_prob: float = 0.004,      # per step; ~1 ODD change per 250 steps (6.25 s)
        max_steps: int = 400,              # 10 s at dt=0.025
        adversary: bool = True,            # expose the F channel for ISAACS
        spawn_theta_frac: float = 0.9,     # reset support as a fraction of the failure threshold
        spawn_omega_frac: float = 0.5,     # E008 used 0.5 -> covered only HALF the scoring grid's
                                           # omega range, so the boundary at high |omega| was never
                                           # supervised. Widen toward ~0.98 for the gate.
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        assert odd_mode in ("static", "sine")
        assert obs_mode in ("oracle", "blind", "history")
        assert margin_mode in ("avoid", "reach_avoid")
        self.odd_mode, self.obs_mode, self.margin_mode = odd_mode, obs_mode, margin_mode
        self.l_neg, self.l_scale = l_neg, l_scale
        self.history_len = history_len
        self.mass_range, self.A_range, self.w_range = mass_range, A_range, w_range
        self.resample_prob, self.max_steps = resample_prob, max_steps
        self.adversary = adversary
        self.spawn_theta_frac, self.spawn_omega_frac = spawn_theta_frac, spawn_omega_frac
        self._rng = np.random.default_rng(seed)

        # ISAACS wants ctrl and dstb in one action vector; safety_sb3's Isaacs* classes split it.
        self.action_space = spaces.Box(
            low=np.array([-U_MAX, -F_BAR][: 2 if adversary else 1], dtype=np.float32),
            high=np.array([U_MAX, F_BAR][: 2 if adversary else 1], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self._obs_dim(),), dtype=np.float32)

        self.theta = self.omega = self.phi = 0.0
        self.m_static = M0_SINE
        self.A = self.w = 0.0
        self._hist: list = []
        self._t = 0

    # ------------------------------------------------------------------ ODD

    def _obs_dim(self) -> int:
        if self.obs_mode == "history":
            return 3 * self.history_len
        base = 3
        if self.obs_mode == "oracle":
            base += 1 if self.odd_mode == "static" else 3   # m | (A, w, phase as sin/cos -> 3)
        return base

    def _sample_odd(self) -> None:
        if self.odd_mode == "static":
            self.m_static = float(self._rng.uniform(*self.mass_range))
        else:
            self.A = float(self._rng.uniform(*self.A_range))
            # log-uniform in w: the frequency axis spans 2+ decades, and E005 shows the
            # interesting structure is spread across them, not concentrated at the top.
            lo, hi = np.log10(self.w_range[0]), np.log10(self.w_range[1])
            self.w = float(10 ** self._rng.uniform(lo, hi))

    @property
    def mass(self) -> float:
        if self.odd_mode == "static":
            return self.m_static
        return M0_SINE + self.A * np.sin(self.phi)

    def odd_features(self) -> np.ndarray:
        """What an oracle policy is told, NORMALIZED to ~[-1, 1] to match sin/cos/omega scale.

        E008 fed raw `m ∈ [2,8]` (magnitude ~5, ±3 variation) alongside sin/cos ∈ [-1,1] — a
        near-constant large input a net learns to ignore, a candidate cause of 'conditioned ≈
        blind'. Normalization is the specific fix. The scorer must call `normalize_mass` with the
        SAME range so train-time and eval-time features agree.
        """
        if self.odd_mode == "static":
            return np.array([self.normalize_mass(self.m_static)], dtype=np.float32)
        A_n = 2 * (self.A - self.A_range[0]) / (self.A_range[1] - self.A_range[0] + 1e-9) - 1
        lo, hi = np.log10(self.w_range[0]), np.log10(self.w_range[1])
        w_n = 2 * (np.log10(max(self.w, 1e-6)) - lo) / (hi - lo + 1e-9) - 1
        return np.array([A_n, w_n, self.phi / np.pi - 1], dtype=np.float32)

    def normalize_mass(self, m: float) -> float:
        """[mass_lo, mass_hi] -> [-1, 1]. Public so the scorer maps eval masses identically."""
        lo, hi = self.mass_range
        return 2 * (m - lo) / (hi - lo + 1e-9) - 1

    # ------------------------------------------------------------------ margins

    def safety_margin(self, theta: float, omega: float) -> float:
        """g >= 0 iff outside the failure set. Each term normalized to O(1) by a FIXED constant
        (min = AND). Unnormalized minima over mixed units (rad, rad/s) make the argmin
        meaningless -- whichever term carries larger raw numbers dominates."""
        return float(min((THETA_MAX - abs(theta)) / THETA_MAX,
                         (OMEGA_LIM - abs(omega)) / OMEGA_LIM))

    def target_margin(self, theta: float, omega: float) -> float:
        """l >= 0 iff inside the 'parked upright' target.

        margin_mode='avoid'  -- l is a NEGATIVE CONSTANT (`l_neg`). This is not a hack: under a
            reach-avoid learner the backup is min(g, max(l, gamma V')), so a constant l < inf-
            imum of gamma V' makes max(l, gamma V') = gamma V' and the backup collapses to
            min(g, gamma V') -- EXACTLY the two-player AVOID problem that E003/E005 compute as
            ground truth. Without this we would be scoring a reach-avoid critic against an
            avoid grid: different objects. (Vault gotcha: "for an avoid-only task under a
            reach-avoid learner, use l_neg" -- and NEVER l_zero, which makes the target set the
            whole space, clips all negative futures so coming failures never propagate, and
            destroys a warm-started policy in <25M steps.)

        margin_mode='reach_avoid' -- distance to the parked-upright ball, normalized AND
            CLAMPED to +-l_scale.

            THE CLAMP IS THE RISK DIAL, NOT COSMETICS. A committed maneuver beats stopping when
            p*l > (1-p)*|g|, so the break-even attempt probability is p* = |g|/(|g|+l). An
            UNCLAMPED l reaches -18 here (|theta|=0.9 against a 0.05 threshold), which both
            injects a huge negative into every backup and pins p* near 1 -- the policy stops
            attempting anything while retaining the skill. Most formulations set this by
            accident; we set it on purpose, because proposal 4-revised says the ODD must move
            (|g|, l) MAGNITUDES for the predicted collapse mechanism to be in play at all.
            l_scale is therefore a candidate ODD coordinate, not a constant.
        """
        if self.margin_mode == "avoid":
            return float(self.l_neg)
        raw = min((0.05 - abs(theta)) / 0.05, (0.25 - abs(omega)) / 0.25)
        return float(np.clip(raw, -self.l_scale, self.l_scale))

    # ------------------------------------------------------------------ gym API

    def _obs(self) -> np.ndarray:
        core = np.array([np.sin(self.theta), np.cos(self.theta), self.omega], dtype=np.float32)
        if self.obs_mode == "blind":
            return core
        if self.obs_mode == "oracle":
            return np.concatenate([core, self.odd_features()]).astype(np.float32)
        self._hist.append(core)
        self._hist = self._hist[-self.history_len:]
        while len(self._hist) < self.history_len:
            self._hist.insert(0, self._hist[0])
        return np.concatenate(self._hist).astype(np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._sample_odd()
        self.phi = float(self._rng.uniform(0, 2 * np.pi))
        # Spawn anywhere in the safe set. Check margins against the reset distribution's own
        # physics: a margin violated by the spawns themselves condemns that state space by
        # construction.
        self.theta = float(self._rng.uniform(-self.spawn_theta_frac * THETA_MAX,
                                             self.spawn_theta_frac * THETA_MAX))
        self.omega = float(self._rng.uniform(-self.spawn_omega_frac * OMEGA_LIM,
                                             self.spawn_omega_frac * OMEGA_LIM))
        self._hist, self._t = [], 0
        obs = self._obs()
        return obs, {"l_x": self.target_margin(self.theta, self.omega), "mass": self.mass}

    def step(self, action):
        a = np.atleast_1d(np.asarray(action, dtype=np.float64))
        u = float(np.clip(a[0], -U_MAX, U_MAX))
        F = float(np.clip(a[1], -F_BAR, F_BAR)) if (self.adversary and a.size > 1) else 0.0

        m = self.mass
        acc = (G / L) * np.sin(self.theta) + u / (m * L * L) + (F / (m * L)) * np.cos(self.theta)
        self.omega = self.omega + acc * DT
        self.theta = self.theta + self.omega * DT
        if self.odd_mode == "sine":
            self.phi = (self.phi + self.w * DT) % (2 * np.pi)

        # The ODD may change MID-EPISODE -- this is what teaches the handoff.
        if self._rng.random() < self.resample_prob:
            self._sample_odd()

        self._t += 1
        g = self.safety_margin(self.theta, self.omega)
        l_x = self.target_margin(self.theta, self.omega)

        terminated = g < 0                     # the env MUST terminate on failure
        truncated = self._t >= self.max_steps
        obs = self._obs()
        # reward IS the safety margin -- never normalize it downstream
        return obs, g, terminated, truncated, {"l_x": l_x, "mass": m, "odd": self.odd_features()}


assert 0.05 < THETA_MAX and 0.25 < OMEGA_LIM, "nesting invariant violated: l must be strictly inside g"
