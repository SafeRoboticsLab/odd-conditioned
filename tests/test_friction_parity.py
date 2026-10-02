"""Parity + sanity for FrictionBicycleEnv. Guards that env physics == E011 grid physics (required for
grid-vs-RL critic scoring), and that the mode-switched ODD + safety_sb3 contract work."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))
import E011_friction_grid_gate as E                       # grid ground truth
from odd_conditioned.envs import friction_bicycle as FB   # the new env


def test_constants_match():
    for k in ("G", "V_MIN", "V_MAX", "DT", "WHEELBASE", "DELTA_LIM", "KAPPA_MAX", "CAR_HALF", "R_GAP"):
        assert np.isclose(getattr(E, k), getattr(FB, k)), (k, getattr(E, k), getattr(FB, k))
    assert np.allclose(E.OBSTACLES, FB.OBSTACLES), "geometry drift vs E011"
    assert np.allclose(E.GOAL, FB.GOAL)
    print("  constants+geometry: MATCH E011")


def test_dynamics_match():
    """The env's friction projection + one-step must reproduce E011's controls()+backup dynamics."""
    rng = np.random.default_rng(0)
    maxerr = 0.0
    for _ in range(2000):
        x, y = rng.uniform(-0.5, 3.5), rng.uniform(-1.2, 2.8)
        psi, v = rng.uniform(-np.pi, np.pi), rng.uniform(0.05, FB.V_MAX)
        mu = rng.uniform(0.1, 1.0)
        # pick one of E011's discrete controls and check the env reproduces its next state
        frac = rng.choice([-1.0, -0.5, 0.0, 0.5, 1.0]); sign = rng.choice([1.0, 0.0, -1.0])
        a_fric = mu * E.G
        kmax_fric = a_fric / max(v * v, 1e-6)
        k_e = np.clip(frac * E.KAPPA_MAX, -kmax_fric, kmax_fric)         # E011 controls() curvature
        rem = np.sqrt(max(a_fric ** 2 - (v * v * abs(k_e)) ** 2, 0.0))
        a_e = sign * rem
        # E011 next state (its grid backup dynamics)
        sx = np.array([x + v * np.cos(psi) * E.DT, y + v * np.sin(psi) * E.DT,
                       (psi + v * k_e * E.DT + np.pi) % (2 * np.pi) - np.pi,
                       np.clip(v + a_e * E.DT, E.V_MIN, E.V_MAX)])
        # env: feed the SAME raw command (a_e, k_e already friction-feasible) and step
        a_long, kappa = FB.project_control(a_e, k_e, v, mu)
        sf = FB.step_state(np.array([x, y, psi, v]), a_long, kappa)
        maxerr = max(maxerr, np.abs(sx - sf).max())
    assert maxerr < 1e-9, f"dynamics mismatch {maxerr}"
    print(f"  dynamics: MATCH E011 (max err {maxerr:.1e} over 2000 states)")


def test_margins_match():
    rng = np.random.default_rng(1)
    grid = E.Grid4(21, 17, 8, 6)
    g, l = E.margins(grid)   # vectorized grid margins (g clipped [-1,1]); l here is E011's clip-±1
    # compare env g_of to E011 g at random grid points (l differs by design -> only check g)
    err = 0.0
    for _ in range(500):
        ix, iy, ip, iv = (rng.integers(n) for n in grid.shape)
        s = np.array([grid.x[ix], grid.y[iy], grid.psi[ip], grid.v[iv]])
        err = max(err, abs(FB.g_of(s) - g[ix, iy, ip, iv]))
    assert err < 1e-9, f"g margin mismatch {err}"
    print(f"  g-margin: MATCH E011 (max err {err:.1e}); NOTE l is graded in env by design")


def test_gym_contract_and_modes():
    for mode in ("mu_local", "blind", "preview", "history"):
        env = FB.FrictionBicycleEnv(obs_mode=mode, seed=3)
        obs, info = env.reset(seed=3)
        assert env.observation_space.shape == obs.shape, mode
        assert "l_x" in info
        for _ in range(30):
            obs, r, term, trunc, info = env.step(env.action_space.sample())
            assert np.isfinite(r) and set(("l_x", "reached", "collided")) <= info.keys()
            if term or trunc:
                obs, info = env.reset()
    # provider modes produce sensible fields
    sf = FB.StripeFieldMu(n_stripes=4, L_min=1.0); sf.reset(np.random.default_rng(0))
    mus = [sf(x, 0, 0) for x in np.linspace(-0.4, 3.5, 20)]
    assert all(0.1 - 1e-6 <= m <= 1.0 + 1e-6 for m in mus) and len(set(mus)) >= 2, "stripe field flat/oob"
    ice = FB.ice_at_corner(); assert ice(0.0, 0, 0) > 0.8 and ice(2.0, 0, 0) < 0.3, "ice-at-corner wrong"
    print("  gym contract + all obs-modes + providers: OK")


def test_grid_optimal_rollout_parity():
    """Roll the E011 grid-optimal policy IN the env; outcomes must match E012's standalone rollout
    (dynamics parity end-to-end): (0.8,0,+x,4.0) reaches at μ=1.0, crashes at μ=0.1."""
    gate = "results/E011_v6/gate.npz"            # produced by experiments/E011_friction_grid_gate.py
    if not os.path.exists(gate):
        import pytest
        pytest.skip(f"{gate} not present (regenerate with experiments/E011_friction_grid_gate.py)")
    d = np.load(gate)
    grid = E.Grid4(len(d["x"]), len(d["y"]), len(d["psi"]), len(d["v"]))
    gg, ll = E.margins(grid)
    outcomes = {}
    for mu in (1.0, 0.1):
        env = FB.FrictionBicycleEnv(mu_provider=FB.ConstantMu(fixed=mu), obs_mode="mu_local",
                                    randomize=False, seed=0)
        env.reset(); env.s = np.array([0.8, 0.0, 0.0, 4.0]); V = d[f"V_{mu}"]
        oc = "timeout"
        for _ in range(220):
            s = env.s
            # greedy lookahead on the grid value with goal-directed tie-break (== E012 render policy)
            cands = []
            for a_l, k in E.controls(np.array(s[3]), mu):
                sn = FB.step_state(s, float(a_l), float(k))
                vn = float(np.ravel(grid.interp(V, *[np.array([q]) for q in sn]))[0])
                dg = float(np.hypot(sn[0] - FB.GOAL[0], sn[1] - FB.GOAL[1]))
                cands.append((vn, dg, float(k), float(a_l)))
            bV = max(c[0] for c in cands)
            _, _, bk, ba = min((c for c in cands if c[0] >= bV - 1e-3), key=lambda c: c[1])
            obs, r, term, trunc, info = env.step([ba / FB.G, bk / FB.KAPPA_MAX])
            if info["collided"]: oc = "COLLIDED"; break
            if info["reached"]: oc = "reached"; break
        outcomes[mu] = oc
    print(f"  grid-optimal-in-env: μ=1.0 -> {outcomes[1.0]} , μ=0.1 -> {outcomes[0.1]}")
    assert outcomes[1.0] == "reached", outcomes
    assert outcomes[0.1] == "COLLIDED", outcomes


if __name__ == "__main__":
    for fn in (test_constants_match, test_dynamics_match, test_margins_match,
               test_gym_contract_and_modes, test_grid_optimal_rollout_parity):
        fn()
    print("ALL PARITY/SANITY CHECKS PASS")
