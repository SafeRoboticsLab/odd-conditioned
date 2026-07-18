"""FrictionBicycleEnv — the mode-switched env for the ODD-conditioned safety filter (friction toy).

Env-fork DECISION (professor, 2026-07-18): ONE mode-switched env, not several files. Two orthogonal
plug points:
  * μ-provider : `constant` | `stripe_field` | `scripted`  — the friction field μ(x,y), FIXED within an
                 episode, RE-RANDOMIZED across episodes. Spatial (not temporal): dwell = patch/ v, and
                 w ∝ 1/v ("slowing buys guaranteed-μ horizon") is expressible — temporal RNG cannot.
  * obs-mode  : `mu_local` | `blind` | `preview(d_obs)` | `history`  — how much of the ODD the agent sees.

⚠ CERTIFIED-CRITIC TRAINING USES `constant` ONLY. Training the conditioned critic on a spatial field
while conditioning on the instantaneous local μ makes the Bellman target bootstrap ACROSS patch
boundaries ⇒ it learns the DR/field-AVERAGED value (E008b blind_dyn pathology), not the per-μ slice.
The field modes are for EVALUATION and for the belief/estimator only — NEVER for slice-critic TD.

Dynamics/geometry are the E011 friction-circle model EXACTLY (parity guarded by
`tests/test_friction_parity.py`), so the trained critic V(x;μ) is directly scorable against the
E011/E012 grid truth (IoU-vs-μ, the E008c protocol). State [x,y,ψ,v], control projected onto the
friction circle a_long²+(v²κ)² ≤ (μG)².

safety_sb3 contract (matches the vendored BicycleGoal): reward = g (safety margin, terminate g<0),
info["l_x"] = l (target margin, ≥0 in goal); reset -> (obs, {"l_x": l}).
"""
from typing import Callable, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np

# --- physics constants — MUST MATCH experiments/E011_friction_grid_gate.py (parity test guards this) --
G = 9.8
V_MIN, V_MAX = 0.0, 8.0
DT = 0.05
WHEELBASE, DELTA_LIM = 0.257, 0.35
KAPPA_MAX = np.tan(DELTA_LIM) / WHEELBASE          # ~1.42 /m
CAR_HALF = 0.16
R_GAP = 1.20
MU_RANGE = (0.1, 1.0)                              # ice -> dry

# --- geometry (E011 v6.2 "fast-approach dead-end with a tight exit") ---------------------------------
GOAL = (2.70, 2.45, 0.45)
X_DEAD, X_TOPEND = 3.60, 1.55
ARENA_X, ARENA_Y = (-0.6, 4.2), (-1.4, 3.1)        # for spawns / field extent


def _hwall(y, x0, x1, r=0.28, step=0.26):
    return [[x, y, r] for x in np.arange(x0, x1 + 1e-9, step)]


def _vwall(x, y0, y1, r=0.28, step=0.26):
    return [[x, y, r] for y in np.arange(y0, y1 + 1e-9, step)]


OBSTACLES = np.array(
    _hwall(-1.00, -0.4, X_DEAD) + _hwall(+1.00, -0.4, X_TOPEND) +
    _vwall(X_DEAD, -1.0, 1.0) + _vwall(X_DEAD, 1.0, 2.9, r=0.30) +
    [[1.30, 2.75, 0.34], [1.70, 2.9, 0.30]], dtype=np.float64)


# --- friction-circle control projection & one-step dynamics (Euler, matches E011) -------------------
def project_control(a_cmd: float, kappa_cmd: float, v: float, mu: float) -> Tuple[float, float]:
    """Project a raw (a_long, curvature) command onto the friction circle at speed v, friction mu.
    Friction-limit the curvature FIRST (|κ| ≤ μG/v²), then spend the remaining budget on a_long — the
    E011 v6.2 bugfix: a turn that alone busts the circle is infeasible regardless of throttle."""
    a_fric = mu * G
    kmax = min(KAPPA_MAX, a_fric / max(v * v, 1e-6))
    kappa = float(np.clip(kappa_cmd, -kmax, kmax))
    a_lat = v * v * abs(kappa)
    rem = float(np.sqrt(max(a_fric * a_fric - a_lat * a_lat, 0.0)))
    a_long = float(np.clip(a_cmd, -rem, rem))
    return a_long, kappa


def step_state(s: np.ndarray, a_long: float, kappa: float) -> np.ndarray:
    """One Euler step of [x, y, ψ, v] — identical to the E011 grid backup dynamics."""
    x, y, psi, v = s
    return np.array([
        x + v * np.cos(psi) * DT,
        y + v * np.sin(psi) * DT,
        (psi + v * kappa * DT + np.pi) % (2 * np.pi) - np.pi,
        np.clip(v + a_long * DT, V_MIN, V_MAX),
    ], dtype=np.float64)


def g_of(s: np.ndarray) -> float:
    """Safety margin: min signed distance from the car (circular, radius CAR_HALF) to any obstacle.
    Clipped to [-1,1] (E011 convention) — g>=0 iff collision-free."""
    d = np.hypot(OBSTACLES[:, 0] - s[0], OBSTACLES[:, 1] - s[1]) - (OBSTACLES[:, 2] + CAR_HALF)
    return float(np.clip(d.min(), -1.0, 1.0))


def l_of(s: np.ndarray, goal_value: float = 0.3, l_scale: float = 4.0) -> float:
    """Target margin from distance-to-goal. GRADED outside the goal (piecewise, like the vendored env)
    so the reach gradient survives out to the max spawn distance (~3.6 m) — E011's clip-±1 l would be
    flat -1 at spawn and the policy would never learn to move. NOTE: the E011/E012 GRID used clip-±1 l;
    for exact value-parity at Rung 1, re-solve the grid with this l (cheap) or compare sign-sets only."""
    d = float(np.hypot(s[0] - GOAL[0], s[1] - GOAL[1]))
    gr = GOAL[2]
    inside = goal_value * (gr - d) / gr
    outside = (gr - d) / l_scale
    return float(np.clip(inside if d <= gr else outside, -1.0, 1.0))


# =================================================================================================
# μ-providers: field(x, y, t) -> μ, fixed within an episode, re-randomized on reset(rng).
# =================================================================================================
class ConstantMu:
    """Uniform field: one μ per episode. THE certified-training provider (no cross-boundary bootstrap)."""
    def __init__(self, mu_range: Tuple[float, float] = MU_RANGE, fixed: Optional[float] = None):
        self.mu_range, self.fixed, self.mu = mu_range, fixed, 1.0

    def reset(self, rng):
        self.mu = float(self.fixed) if self.fixed is not None else float(rng.uniform(*self.mu_range))

    def __call__(self, x, y, t):
        return self.mu

    def ahead(self, x, y, psi, d, t):
        return self.mu


class StripeFieldMu:
    """Spatial μ field as vertical (along-x) stripes ≥ L_min wide — travel direction, clean dwell = L/v.
    EVAL/ESTIMATOR ONLY (never certified TD)."""
    def __init__(self, n_stripes: int = 4, mu_range: Tuple[float, float] = MU_RANGE,
                 L_min: float = 1.5, x_lo: float = ARENA_X[0], x_hi: float = X_DEAD):
        self.n, self.mu_range, self.L_min = n_stripes, mu_range, L_min
        self.x_lo, self.x_hi = x_lo, x_hi
        self.edges = np.array([x_lo, x_hi]); self.mus = np.array([1.0])

    def reset(self, rng):
        span = self.x_hi - self.x_lo
        n = max(1, min(self.n, int(span // self.L_min)))     # respect L_min
        cuts = np.sort(rng.uniform(self.x_lo, self.x_hi, n - 1)) if n > 1 else np.array([])
        self.edges = np.concatenate([[self.x_lo], cuts, [self.x_hi]])
        # enforce L_min by merging too-thin stripes into the previous edge
        keep = [self.edges[0]]
        for e in self.edges[1:]:
            if e - keep[-1] >= self.L_min or e == self.edges[-1]:
                keep.append(e)
        self.edges = np.array(keep)
        self.mus = rng.uniform(*self.mu_range, len(self.edges) - 1)

    def _mu_at_x(self, x):
        i = int(np.clip(np.searchsorted(self.edges, x) - 1, 0, len(self.mus) - 1))
        return float(self.mus[i])

    def __call__(self, x, y, t):
        return self._mu_at_x(x)

    def ahead(self, x, y, psi, d, t):
        return self._mu_at_x(x + d * np.cos(psi))            # μ a distance d ahead along heading


class ScriptedMu:
    """Explicit field for worst-case eval, e.g. dry everywhere with an ICE STRIPE at the corner.
    segments = [(x_threshold, μ), ...] sorted; μ for x < first threshold is `base`."""
    def __init__(self, segments: Sequence[Tuple[float, float]], base: float = 1.0):
        self.segments = sorted(segments); self.base = base

    def reset(self, rng):
        pass

    def _mu_at_x(self, x):
        mu = self.base
        for xt, m in self.segments:
            if x >= xt:
                mu = m
        return float(mu)

    def __call__(self, x, y, t):
        return self._mu_at_x(x)

    def ahead(self, x, y, psi, d, t):
        return self._mu_at_x(x + d * np.cos(psi))


def ice_at_corner(mu_dry: float = 1.0, mu_ice: float = 0.15,
                  x0: float = 1.4, x1: float = 2.6) -> ScriptedMu:
    """Dry approach, ICE exactly across the corner-exit band [x0,x1], dry again past it (unreachable)."""
    return ScriptedMu([(ARENA_X[0], mu_dry), (x0, mu_ice), (x1, mu_dry)], base=mu_dry)


# =================================================================================================
class FrictionBicycleEnv(gym.Env):
    """Reach-avoid friction-circle bicycle with a mode-switched friction ODD. See module docstring."""
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, mu_provider=None, obs_mode: str = "mu_local", d_obs: float = 1.0,
                 hist_len: int = 8, randomize: bool = True, timeout: int = 400,
                 terminate_on_goal: bool = True, mu_range: Tuple[float, float] = MU_RANGE,
                 seed: Optional[int] = None):
        assert obs_mode in ("mu_local", "blind", "preview", "history"), obs_mode
        self.mu_provider = mu_provider if mu_provider is not None else ConstantMu(mu_range)
        self.obs_mode, self.d_obs, self.hist_len = obs_mode, d_obs, hist_len
        self.randomize, self.timeout, self.terminate_on_goal = randomize, int(timeout), terminate_on_goal
        self.mu_range = mu_range
        self._rng = np.random.default_rng(seed)
        # obs = [x, y, sinψ, cosψ, v] (+ ODD channel(s))
        base = 5
        extra = {"mu_local": 1, "blind": 0, "preview": 2, "history": hist_len}[obs_mode]
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (base + extra,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)   # (a_long, κ), scaled inside
        self.ctrl_action_dim = 2
        self.s = np.zeros(4); self.t = 0; self._hist = np.zeros(hist_len)

    # --- ODD helpers ----------------------------------------------------------
    def _mu_norm(self, mu: float) -> float:
        lo, hi = self.mu_range
        return 2 * (mu - lo) / (hi - lo + 1e-9) - 1

    def _mu_lower_estimate(self, s_prev, a_long, kappa) -> float:
        """Closed-form set-membership LOWER bound on μ from the last realized control: any executed
        (a_long, a_lat=v²κ) proves μ ≥ ‖(a_long, a_lat)‖ / G. Tightens under excitation (E009)."""
        a_lat = s_prev[3] ** 2 * abs(kappa)
        return float(np.hypot(a_long, a_lat) / G)

    def _obs(self) -> np.ndarray:
        x, y, psi, v = self.s
        base = [x, y, np.sin(psi), np.cos(psi), v]
        if self.obs_mode == "mu_local":
            base.append(self._mu_norm(self.mu_provider(x, y, self.t)))
        elif self.obs_mode == "preview":
            base.append(self._mu_norm(self.mu_provider(x, y, self.t)))
            base.append(self._mu_norm(self.mu_provider.ahead(x, y, psi, self.d_obs, self.t)))
        elif self.obs_mode == "history":
            base.extend(self._hist.tolist())
        return np.asarray(base, dtype=np.float32)

    # --- gym ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.mu_provider.reset(self._rng)
        if self.randomize:
            # broad spawns over the full grid-covered region + all headings (the critic must match the
            # grid everywhere); reject in-obstacle spawns so they don't waste immediate-terminate episodes
            for _ in range(50):
                s = np.array([self._rng.uniform(-0.4, 3.2), self._rng.uniform(-0.9, 2.7),
                              self._rng.uniform(-np.pi, np.pi), self._rng.uniform(0.0, V_MAX)])
                if g_of(s) >= 0.0:
                    break
            self.s = s
        else:
            self.s = np.array([0.0, 0.0, 0.0, 2.0])
        self.t = 0; self._hist = np.full(self.hist_len, self._mu_norm(self.mu_range[0]))
        return self._obs(), {"l_x": l_of(self.s), "mu": self.mu_provider(self.s[0], self.s[1], 0)}

    def step(self, action):
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        mu = float(self.mu_provider(self.s[0], self.s[1], self.t))
        a_long, kappa = project_control(a[0] * G, a[1] * KAPPA_MAX, self.s[3], mu)  # v = s[3]
        s_prev = self.s.copy()
        self.s = step_state(self.s, a_long, kappa)
        self.t += 1
        # roll the set-membership belief history (for obs_mode="history")
        self._hist = np.roll(self._hist, -1)
        self._hist[-1] = self._mu_norm(min(max(self._mu_lower_estimate(s_prev, a_long, kappa),
                                               self.mu_range[0]), self.mu_range[1]))
        g, l = g_of(self.s), l_of(self.s)
        reached = l >= 0.0
        terminated = bool(g < 0.0) or bool(reached and self.terminate_on_goal)
        truncated = self.t >= self.timeout
        return (self._obs(), float(g), terminated, truncated,
                {"l_x": l, "reached": bool(reached), "collided": bool(g < 0.0), "mu": mu})

    # --- render ---------------------------------------------------------------
    def render_frame(self, ax=None, trail=None):
        import matplotlib.patches as mp
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(figsize=(5, 4))
        for ox, oy, r in OBSTACLES:
            ax.add_patch(mp.Circle((ox, oy), r, color="#444", zorder=2))
        ax.add_patch(mp.Circle(GOAL[:2], GOAL[2], color="#27ae60", alpha=0.4, zorder=1))
        if trail is not None and len(trail):
            ax.plot(trail[:, 0], trail[:, 1], "-", color="#2980b9", lw=1.4, zorder=3)
        x, y, psi, v = self.s
        car = mp.Rectangle((-0.21, -0.095), 0.42, 0.19, color="#c0392b", zorder=4)
        car.set_transform(plt.matplotlib.transforms.Affine2D().rotate(psi).translate(x, y) + ax.transData)
        ax.add_patch(car)
        ax.set_aspect("equal"); ax.set_xlim(*ARENA_X); ax.set_ylim(*ARENA_Y)
        return ax
