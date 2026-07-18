"""E011 rollouts — render the car taking the SAME corner under different friction (the ODD).

NOT a trained NN policy (none exists on this toy yet — the RL is downstream, gated by E011). This
rolls out the GROUND-TRUTH HJ-optimal safe controller for each mu (the per-cell argmax `best_k`,
`best_a` from the grid solve in gate.npz) from a fixed start state, under the REAL friction-circle
dynamics. Same start, different mu => you SEE the mode flip: thread the corner at speed (grip) vs
brake-and-creep or fail (ice). This is the visual of "least-conservative-per-ODD + strategy differs".

Usage:
  python experiments/E011_render_rollouts.py --scan          # print outcome table over candidate starts
  python experiments/E011_render_rollouts.py --start X Y PSIdeg V --mus 1.0 0.35 0.1   # render one GIF
"""
import argparse
import os

import numpy as np

import E011_friction_grid_gate as E  # same dir (added to path below)


def load(npz):
    d = np.load(npz)
    return d


def nearest_idx(grid, s):
    xi = int(np.argmin(np.abs(grid.x - s[0]))); yi = int(np.argmin(np.abs(grid.y - s[1])))
    pi = int(np.argmin(np.abs(((grid.psi - s[2] + np.pi) % (2 * np.pi)) - np.pi)))
    vi = int(np.argmin(np.abs(grid.v - s[3])))
    return xi, yi, pi, vi


def _gl_at(grid, g, l, s):
    xi, yi, pi, vi = nearest_idx(grid, s)
    return float(g[xi, yi, pi, vi]), float(l[xi, yi, pi, vi])


def rollout(grid, g, l, V, mu, s0, max_steps=220):
    """GREEDY HJ-OPTIMAL rollout: at each state pick the friction-feasible control that maximizes the
    interpolated value V(f(s,u)) — the true optimal safe policy for this mu — under real dynamics.
    (Cleaner than stored per-cell argmax, which limit-cycles near the goal.) Returns traj + outcome."""
    s = np.array(s0, dtype=float)
    traj = []
    outcome = "timeout"
    for t in range(max_steps):
        gs, ls = _gl_at(grid, g, l, s)
        # choose best control by one-step lookahead on V; tie-break toward the goal so the greedy
        # policy doesn't stall on the reach-avoid value PLATEAU (many next-states share the max V).
        cands = []
        for a, k in E.controls(np.array(s[3]), mu):
            a = float(a); k = float(k)
            xn = s[0] + s[3] * np.cos(s[2]) * E.DT
            yn = s[1] + s[3] * np.sin(s[2]) * E.DT
            pn = (s[2] + s[3] * k * E.DT + np.pi) % (2 * np.pi) - np.pi
            vn = np.clip(s[3] + a * E.DT, E.V_MIN, E.V_MAX)
            Vn = float(grid.interp(V, np.array([xn]), np.array([yn]), np.array([pn]), np.array([vn]))[0])
            dgoal = np.hypot(xn - E.GOAL[0], yn - E.GOAL[1])
            cands.append((Vn, dgoal, k, a))
        bestV = max(c[0] for c in cands)
        # among controls within eps of optimal V (equally safe), take the one closest to the goal
        near = [c for c in cands if c[0] >= bestV - 1e-3]
        _, _, bk, ba = min(near, key=lambda c: c[1])
        mode = "coast"
        if abs(bk) > 0.55 * E.KAPPA_MAX:
            mode = "THREAD"
        elif ba < -0.3:
            mode = "BRAKE"
        elif ba > 0.3:
            mode = "accel"
        traj.append((s.copy(), bk, ba, mode, gs, ls))
        if gs < 0:
            outcome = "COLLIDED"; break
        if ls >= 0:
            outcome = "reached"; break
        s = np.array([s[0] + s[3] * np.cos(s[2]) * E.DT,
                      s[1] + s[3] * np.sin(s[2]) * E.DT,
                      (s[2] + s[3] * bk * E.DT + np.pi) % (2 * np.pi) - np.pi,
                      np.clip(s[3] + ba * E.DT, E.V_MIN, E.V_MAX)])
    return traj, outcome


def scan(grid, g, l, d, mus):
    print(f"{'start (x,y,psideg,v)':>26} | " + " ".join(f"{m:>16}" for m in mus))
    cand = []
    for x0 in (-0.2, 0.3, 0.8, 1.3):
        for v0 in (2.0, 3.0, 4.0, 5.0):
            s0 = (x0, 0.0, 0.0, v0)
            row = []
            modes = []
            for m in mus:
                traj, oc = rollout(grid, g, l, d[f"V_{m}"], m, s0)
                dom = _dominant_mode(traj)
                row.append(f"{oc}/{dom}")
                modes.append((oc, dom))
            print(f"{str(s0):>26} | " + " ".join(f"{c:>16}" for c in row))
            cand.append((s0, modes))
    return cand


def _dominant_mode(traj):
    ms = [t[3] for t in traj if t[3] in ("THREAD", "BRAKE")]
    if not ms:
        return "coast"
    return max(set(ms), key=ms.count)


def render(grid, d, mus, s0, out, fps=20):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    g, l = E.margins(grid)

    trajs = {}
    for m in mus:
        traj, oc = rollout(grid, g, l, d[f"V_{m}"], m, s0)
        trajs[m] = (traj, oc)
        print(f"  mu={m}: {oc} in {len(traj)} steps, dominant={_dominant_mode(traj)}")
    T = max(len(t) for t, _ in trajs.values())

    n = len(mus)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 4.0))
    if n == 1:
        axes = [axes]
    L = 0.42; Wd = 0.22  # car body (m)

    def draw_static(ax, m):
        for ox, oy, r in E.OBSTACLES:
            ax.add_patch(plt.Circle((ox, oy), r, color="#444", alpha=0.9, zorder=1))
        ax.add_patch(plt.Circle(E.GOAL[:2], E.GOAL[2], color="#27ae60", alpha=0.35, zorder=1))
        ax.set_xlim(grid.x[0], grid.x[-1]); ax.set_ylim(grid.y[0], grid.y[-1])
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        grip = "DRY" if m >= 0.8 else ("WET" if m >= 0.3 else "ICE")
        ax.set_title(f"$\\mu$={m}  ({grip})", fontsize=11)

    cars, paths, txts = [], [], []
    for ax, m in zip(axes, mus):
        draw_static(ax, m)
        (path,) = ax.plot([], [], "-", color="#2980b9", lw=1.4, alpha=0.7, zorder=2)
        car = plt.Polygon(np.zeros((4, 2)), closed=True, color="#c0392b", zorder=3)
        ax.add_patch(car)
        txt = ax.text(0.03, 0.97, "", transform=ax.transAxes, va="top", ha="left",
                      fontsize=9, family="monospace",
                      bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))
        cars.append(car); paths.append(path); txts.append(txt)

    def body(s):
        x, y, psi, v = s
        c, sn = np.cos(psi), np.sin(psi)
        corners = np.array([[L / 2, Wd / 2], [L / 2, -Wd / 2], [-L / 2, -Wd / 2], [-L / 2, Wd / 2]])
        R = np.array([[c, -sn], [sn, c]])
        return (corners @ R.T) + np.array([x, y])

    def update(f):
        arts = []
        for m, car, path, txt in zip(mus, cars, paths, txts):
            traj, oc = trajs[m]
            i = min(f, len(traj) - 1)
            s, k, a, mode, gs, ls = traj[i]
            car.set_xy(body(s))
            xs = [t[0][0] for t in traj[:i + 1]]; ys = [t[0][1] for t in traj[:i + 1]]
            path.set_data(xs, ys)
            done = ("✓ reached" if oc == "reached" else ("✗ CRASH" if oc == "COLLIDED" else "…")) if i == len(traj) - 1 else ""
            col = "#c0392b" if (oc == "COLLIDED" and i == len(traj) - 1) else "#c0392b"
            car.set_color("#7f8c8d" if (oc == "COLLIDED" and i == len(traj) - 1) else col)
            txt.set_text(f"v={s[3]:.1f} m/s\nmode={mode}\n{done}")
            arts += [car, path, txt]
        return arts

    anim = FuncAnimation(fig, update, frames=T + 12, interval=1000 / fps, blit=True)
    fig.suptitle(f"Same corner, same start ({s0[0]:.1f},{s0[1]:.1f},{np.degrees(s0[2]):.0f}°,{s0[3]:.1f} m/s) "
                 f"— HJ-optimal safe policy per friction $\\mu$", fontsize=11)
    fig.tight_layout()
    anim.save(out, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"wrote {out}")


def main():
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="results/E011_v6/gate.npz")
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--start", type=float, nargs=4, default=None, help="x y psi_deg v")
    ap.add_argument("--mus", type=float, nargs="+", default=[1.0, 0.35, 0.1])
    ap.add_argument("--out", default="results/E011_v6/rollout.gif")
    args = ap.parse_args()

    d = load(args.npz)
    x, y, psi, v = d["x"], d["y"], d["psi"], d["v"]
    grid = E.Grid4(len(x), len(y), len(psi), len(v))
    g, l = E.margins(grid)
    avail = list(d["mus"])
    print(f"npz mus available: {avail}")
    mus = [m for m in args.mus if any(abs(m - a) < 1e-6 for a in avail)]
    mus = [next(a for a in avail if abs(m - a) < 1e-6) for m in mus]

    if args.scan:
        scan(grid, g, l, d, avail)
        return
    s0 = (args.start[0], args.start[1], np.radians(args.start[2]), args.start[3]) if args.start \
        else (0.3, 0.0, 0.0, 4.0)
    print(f"start {s0}, mus {mus}")
    render(grid, d, mus, s0, args.out)


if __name__ == "__main__":
    main()
