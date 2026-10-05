"""Rendered demo videos -> $ODD_OUTPUTS/videos/. Slow (MuJoCo rendering); a GPU with EGL is required.

    payload   per profile: one solo robot per method, side by side; the strip below plots the FLEET curves
              from payload/results.json (run the payload target first). A solo robot is a single draw: each
              method gets up to TRIES seeds and shows its best outcome (reached > alive > furthest), so the
              panels illustrate behaviour; the fleet curves are the result.
    leg       nine robots per method; strip: θ(t) and the median distance to goal of the live robots
    standing  sixteen robots per method; strip: load W(t), survival, and the ODD-conditioned mode ribbon
    compound  the leg-death-while-loaded ramp, twelve robots per arm; the dying FR leg reddens, a robot that
              has handed off turns blue; strip: θ(t), base heights, V_stand

Trunk colour = the majority mode of the live robots: TASK grey (reddening with load / orange with leg
derating), BRAKE yellow, DESCENDING orange, REST blue, GETTING_UP green. The simulator respawns a fallen robot
(it no longer counts as alive), so a robot may visibly stand up again after its death.
"""
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import certificates as C  # noqa: E402
from .automaton import BRAKE, DESCENDING, GETTING_UP, LOAD_NOMINAL, REST, TASK, label, rollout  # noqa: E402
from .paths import OUTPUTS, output_dir  # noqa: E402
from .scenarios import SCENARIOS  # noqa: E402
from .sim import DEV, DT  # noqa: E402

COL = {"odd": "#1a5276", "direct": "#8e44ad", "one-way": "#e67e22", "task-only": "#c0392b", "rest-only": "#1e8449"}
MODE_NAME = {TASK: "TASK", BRAKE: "BRAKING", DESCENDING: "DESCENDING", REST: "REST", GETTING_UP: "GETTING-UP"}
MODE_HEX = {TASK: "#b03a2e", DESCENDING: "#e67e22", REST: "#1a5276", GETTING_UP: "#1e8449", BRAKE: "#d4ac0d"}
TRIES = 6


def tag(img, txt, sub=None, border=None):
    """Darken a header bar and write ``txt`` (and ``sub``) on it; optional coloured border."""
    from PIL import Image, ImageDraw
    img = np.ascontiguousarray(img).copy()
    img[:26] = (img[:26] * 0.3).astype(img.dtype)
    if border is not None:
        b = 5
        img[:b, :] = border; img[-b:, :] = border; img[:, :b] = border; img[:, -b:] = border
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    dr.text((8, 6), txt, fill=(255, 255, 255))
    if sub:
        dr.text((8, 16), sub, fill=(150, 200, 255))
    return np.asarray(im)


def _strip(draw, W, H):
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    draw(fig)
    fig.tight_layout(pad=0.5)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return buf[:H, :W]


def _write(frames, name, fps):
    import imageio.v2 as imageio
    f = os.path.join(output_dir("videos"), name)
    imageio.mimsave(f, frames, fps=fps, macro_block_size=1)
    print(f"video -> {f}", flush=True)


def _mode_word(st, task):
    return ("WALK" if task == "walk" else "STAND") if st == TASK else MODE_NAME[st]


# ── payload walk ──────────────────────────────────────────────────────────────────────────────────────────

def _crate(img, W, h):
    """Draw the carried crate as a stack of boxes (one per 40 N), raised with its centre of mass."""
    from PIL import Image, ImageDraw
    im = Image.fromarray(np.ascontiguousarray(img))
    dr = ImageDraw.Draw(im)
    x0, ybase, lift = 10, 120, int((h - 0.25) * 200)
    for b in range(int(round(W / 40.0))):
        y1 = ybase - lift - b * 12
        dr.rectangle([x0, y1 - 10, x0 + 26, y1], fill=(180, 60, 40), outline=(30, 30, 30))
    dr.text((x0, ybase + 6), f"W={W:.0f}N", fill=(255, 255, 255))
    dr.text((x0, ybase + 18), f"CoM h={h:.2f}m", fill=(255, 255, 255))
    return np.asarray(im)


def payload(profile="period", methods=("task-only", "one-way", "direct", "odd")):
    sc = SCENARIOS[f"payload-{profile}"]
    fleet = json.load(open(os.path.join(OUTPUTS, "payload", "results.json")))
    best = {}
    for m in methods:
        pick, score = None, (-1, -1.0)
        for k in range(TRIES):
            r = rollout(sc, m, n=1, seed=k, record=True, render=True)
            rec = r["record"]
            prog = float(sc.goal - np.linalg.norm(rec["traj"][-1, 0] - rec["goal"][0]))
            s = (2 if r["success"] > 0 else (1 if r["safe"] > 0 else 0), prog)
            if s > score:
                pick, score = r, s
            if s[0] == 2:
                break
        best[m] = pick
        print(f"solo {profile} {m}: reached {pick['success']:.0f} alive {pick['safe']:.0f} (seed {pick['seed']})",
              flush=True)
    nfr = min(len(best[m]["frames"]) for m in methods)
    tt = np.arange(nfr) * 2 * DT
    Wv = np.array([sc.odd(2 * k)[0] for k in range(nfr)])
    n_fleet = fleet[f"{profile}|odd"]["n"]

    def draw(i):
        def f(fig):
            ax = fig.add_subplot(111)
            ax2 = ax.twinx()
            ax2.fill_between(tt, Wv, color="#777", alpha=0.12)
            ax2.plot(tt[:i + 1], Wv[:i + 1], color="#444", lw=1.6, label="payload W(t)")
            ax2.set_ylim(0, 900)
            ax2.set_yticks([sc.trigger_eps, LOAD_NOMINAL, sc.params["W_hi"]])
            ax2.set_ylabel("W (N)")
            for m in methods:
                r = fleet[f"{profile}|{m}"]
                ax.plot(tt[:i + 1], np.array(r["SUC"])[::2][:i + 1], color=COL[m], lw=2.4,
                        label=f"{label(m, 'walk')} reached ({r['success']:.2f})")
                ax.plot(tt[:i + 1], np.array(r["S"])[::2][:i + 1], color=COL[m], lw=1.1, ls="--", alpha=0.7)
            ax.axvline(tt[i], color="#555", lw=1.5)
            ax.set_xlim(0, tt[-1])
            ax.set_ylim(0, 1.05)
            ax.set_xlabel("time (s)")
            ax.set_ylabel(f"fleet fraction (solid = reached, dashed = alive; N={n_fleet})")
            ax.legend(loc="upper left", fontsize=7)
            ax2.legend(loc="lower right", fontsize=7)
        return f

    frames = []
    for i in range(nfr):
        row = []
        for m in methods:
            r, rec = best[m], best[m]["record"]
            k4 = min(i // 2, rec["alive"].shape[0] - 1)
            alive, st = bool(rec["alive"][k4][0]), int(rec["st"][k4][0])
            done = r["success"] > 0 and r["SUC"][min(2 * i, len(r["SUC"]) - 1)] > 0
            status = "REACHED" if done else (_mode_word(st, "walk") if alive else "FLIPPED")
            W, h, _ = sc.odd(2 * i)
            row.append(_crate(tag(r["frames"][i], f"{label(m, 'walk')}  [{status}]"), W, h))
        top = np.hstack(row)
        frames.append(np.vstack([top, _strip(draw(i), top.shape[1], 300)]))
    _write(frames, f"payload_{profile}.mp4", fps=25)


# ── leg-fault walk ────────────────────────────────────────────────────────────────────────────────────────

def leg(methods=("task-only", "direct", "odd"), n=9):
    sc = SCENARIOS["leg-fault"]
    R = {m: rollout(sc, m, n=n, seed=0, record=True, render=True) for m in methods}
    nfr = min(len(R[m]["frames"]) for m in methods)
    tt = np.arange(nfr) * 2 * DT
    thv = np.array([sc.odd(2 * k)[2] for k in range(nfr)])
    med = {}
    for m in methods:
        rec = R[m]["record"]
        c, s = np.cos(-rec["yaw0"]), np.sin(-rec["yaw0"])
        rel = rec["traj"] - rec["spawn"][None]
        gx = rel[..., 0] * c[None] - rel[..., 1] * s[None]
        gy = rel[..., 0] * s[None] + rel[..., 1] * c[None]
        dist = np.sqrt((gx - sc.goal) ** 2 + gy ** 2)
        alv = rec["alive"].astype(bool)
        mm = np.array([np.median(dist[k][alv[k]]) if alv[k].any() else np.nan for k in range(dist.shape[0])])
        med[m] = np.interp(np.arange(nfr) * 2, np.arange(dist.shape[0]) * rec["every"], mm)

    def draw(i):
        def f(fig):
            ax = fig.add_subplot(111)
            ax2 = ax.twinx()
            ax2.fill_between(tt, thv, color="#e67e22", alpha=0.12)
            ax2.plot(tt[:i + 1], thv[:i + 1], color="#e67e22", lw=1.6, label="leg θ(t)")
            ax2.set_ylim(0, 3.2)
            ax2.set_yticks([sc.params["theta_lo"], 1.0])
            ax2.set_ylabel("θ", color="#e67e22")
            for m in methods:
                ax.plot(tt, med[m], color=COL[m], lw=1.0, alpha=0.25)
                ax.plot(tt[:i + 1], med[m][:i + 1], color=COL[m], lw=2.2, label=label(m, "walk"))
            ax.axhline(1.0, color="#1a5276", ls=":", lw=1)
            ax.axvline(tt[i], color="#555", lw=1.5)
            ax.set_xlim(0, tt[-1])
            ax.set_ylim(0, sc.goal + 2)
            ax.set_xlabel("time (s)")
            ax.set_ylabel("median distance to goal (live)")
            ax.legend(loc="upper right", fontsize=7.5)
            ax2.legend(loc="upper left", fontsize=7.5)
        return f

    frames = []
    for i in range(nfr):
        row = []
        for m in methods:
            rec = R[m]["record"]
            k4 = min(i // 2, rec["alive"].shape[0] - 1)
            alv = rec["alive"][k4].astype(bool)
            word = ""
            if m != "task-only" and alv.any():
                word = f" [{_mode_word(int(np.bincount(rec['st'][k4][alv]).argmax()), 'walk')}]"
            row.append(tag(R[m]["frames"][i], f"{label(m, 'walk')}{word}   alive {int(alv.sum())}/{n}"))
        top = np.hstack(row)
        frames.append(np.vstack([top, _strip(draw(i), top.shape[1], 300)]))
    _write(frames, "leg_fault.mp4", fps=25)


# ── standing ──────────────────────────────────────────────────────────────────────────────────────────────

def standing(profile="period", push="benign", methods=("direct", "odd", "one-way"), n=16):
    sc = SCENARIOS[f"standing-{profile}"]
    R = {m: rollout(sc, m, push=push, n=n, seed=0, record=True, render=True) for m in methods}
    nfr = min(len(R[m]["frames"]) for m in methods)
    tt = np.arange(nfr) * 2 * DT
    Wv = np.array([sc.odd(2 * k)[0] for k in range(nfr)])
    surv = {m: np.array(R[m]["S"])[::2][:nfr] for m in methods}
    rec = R["odd"]["record"] if "odd" in R else None

    def ribbon(k):
        if rec is None:
            return None
        alv = rec["alive"][min(k // 2, rec["alive"].shape[0] - 1)].astype(bool)
        sts = rec["st"][min(k // 2, rec["st"].shape[0] - 1)]
        return int(np.bincount(sts[alv]).argmax()) if alv.any() else TASK
    modes = [ribbon(i) for i in range(nfr)]

    def draw(i):
        def f(fig):
            ax = fig.add_subplot(111)
            ax2 = ax.twinx()
            ax2.plot(tt, Wv, color="#222", lw=1.0, alpha=0.25)
            ax2.plot(tt[:i + 1], Wv[:i + 1], color="#222", lw=1.8, label="load W(t)")
            if push == "gusty":
                for g0 in np.arange(0, tt[-1], 2.0):
                    ax2.axvspan(g0, g0 + 0.5, color="#f39c12", alpha=0.10)
            for m in methods:
                ax.plot(tt, surv[m], color=COL[m], lw=1.0, alpha=0.25)
                ax.plot(tt[:i + 1], surv[m][:i + 1], color=COL[m], lw=2.4, label=f"{label(m, 'stand')} survival")
            if rec is not None:
                for k in range(i + 1):
                    ax.axvspan(tt[k], tt[k] + 2 * DT, ymin=0.0, ymax=0.05, color=MODE_HEX[modes[k]], alpha=0.9, lw=0)
                ax.text(0.05, 0.075, "ODD-conditioned mode", fontsize=7, color="#333",
                        transform=ax.get_xaxis_transform())
            ax.axvline(tt[i], color="#555", lw=1.5)
            ax.set_xlim(0, tt[-1])
            ax.set_ylim(0, 1.05)
            ax2.set_ylim(0, 260)
            ax.set_xlabel("time (s)")
            ax.set_ylabel("survival (no respawn)")
            ax2.set_ylabel("W (N)")
            ax.legend(loc="lower left", fontsize=7.5, framealpha=0.9)
            ax2.legend(loc="upper right", fontsize=8)
        return f

    frames = []
    for i in range(nfr):
        row = []
        for m in methods:
            word = f" [{_mode_word(modes[i], 'stand')}]" if m == "odd" else ""
            row.append(tag(R[m]["frames"][i], f"{label(m, 'stand')}{word}   alive {int(round(surv[m][i] * n))}/{n}"))
        top = np.hstack(row)
        frames.append(np.vstack([top, _strip(draw(i), top.shape[1], 320)]))
    _write(frames, f"standing_{profile}_{push}.mp4", fps=25)


# ── compound certificate demo ─────────────────────────────────────────────────────────────────────────────

FR_BODIES, TRUNK_BODIES = (6, 7, 8), (2,)


def compound(n=12):
    import torch as th
    rp = C.RAMPS["compound"]
    data = json.load(open(os.path.join(OUTPUTS, "certificates", "ramp_compound.json")))
    GRAY, RED, BLUE = np.array([.5, .5, .5, 1.]), np.array([.9, .08, .08, 1.]), np.array([.15, .4, .95, 1.])
    grids = {}
    for arm, spec in rp.arms.items():
        frames = []

        def on_step(env, t, theta, switched, _arm=arm, _frames=frames):
            mm, tm = env.mj.sim.mj_model, env.mj.sim.model
            fr = [g for g in range(mm.ngeom) if mm.geom_bodyid[g] in FR_BODIES]
            tk = [g for g in range(mm.ngeom) if mm.geom_bodyid[g] in TRUNK_BODIES]
            tint = GRAY + (RED - GRAY) * (rp.lo - theta) / (rp.lo - rp.hi)
            tm.geom_rgba[:, fr, :] = th.tensor(tint, device=DEV, dtype=tm.geom_rgba.dtype)
            for g in fr:
                mm.geom_rgba[g] = tint
            if _arm == "HANDOFF":
                sw = switched.float().reshape(-1, 1, 1)
                gray = th.tensor(GRAY, device=DEV, dtype=tm.geom_rgba.dtype)[None, None, :]
                blue = th.tensor(BLUE, device=DEV, dtype=tm.geom_rgba.dtype)[None, None, :]
                tm.geom_rgba[:, tk, :] = gray + (blue - gray) * sw
                mm.geom_rgba[tk[0]] = GRAY + (BLUE - GRAY) * float(switched.float().mean() > 0.5)
            _frames.append(np.asarray(env.render()))

        C.handoff_arm(rp, *spec, seed=0, n=n, on_step=on_step)
        grids[arm] = frames
    arms = data["arms"]
    steps = len(data["odd_trace"])
    t = np.arange(steps) * data["dt"]
    thv = np.array(data["odd_trace"])
    V = np.array(arms["HANDOFF"]["V_trace"][rp.readout])
    COLS = {"STAND-ONLY": "#e74c3c", "HANDOFF": "#2980b9", "REST-ONLY": "#27ae60"}
    order = ("STAND-ONLY", "HANDOFF", "REST-ONLY")

    def draw(i):
        def f(fig):
            ax = fig.add_subplot(111)
            ax2 = ax.twinx()
            for g in rp.gusts:
                ax.axvspan(g * DT, (g + rp.gust_len) * DT, color="#f39c12", alpha=0.18, zorder=0)
            ax.plot(t, thv, color="#333", lw=1.0, alpha=0.3)
            ax.plot(t[:i + 1], thv[:i + 1], color="#333", lw=2.2, label="θ (FR torque)")
            for arm in order:
                h = np.array(arms[arm]["h_trace"])
                ax2.plot(t, h, color=COLS[arm], lw=1.0, alpha=0.25)
                ax2.plot(t[:i + 1], h[:i + 1], color=COLS[arm], lw=2.2,
                         label=f"{'ODD-conditioned' if arm == 'HANDOFF' else arm} height")
            ax2.plot(t[:i + 1], 0.17 + V[:i + 1] * 0.10, color="#1a5276", ls="--", lw=1.8, label="V_stand (scaled)")
            ax2.axhline(0.17 + rp.eps * 0.10, color="r", ls=":", lw=1.1, alpha=0.7)
            ax.axvline(t[i], color="#555", lw=1.5)
            ax.set_xlim(0, t[-1])
            ax.set_ylim(0, 1.1)
            ax2.set_ylim(0.05, 0.30)
            ax.set_xlabel("time (s)")
            ax.set_ylabel("θ (FR torque fraction)")
            ax2.set_ylabel("base height (m) / V_stand")
            ax.legend(loc="upper right", fontsize=8)
            ax2.legend(loc="lower left", fontsize=7, ncol=2)
        return f

    frames = []
    for i in range(min(len(grids[a]) for a in order)):
        gust = rp.in_gust(i)
        row = [tag(grids[a][i], "ODD-conditioned" if a == "HANDOFF" else a,
                   f"handed off {np.array(arms[a]['sw_trace'])[i] * 100:.0f}% (fleet)" if a == "HANDOFF" else None,
                   border=(230, 80, 40) if gust else None) for a in order]
        top = np.hstack(row)
        frames.append(np.vstack([top, _strip(draw(i), top.shape[1], 340)]))
    _write(frames, "compound_leg_death.mp4", fps=30)


VIDEOS = {"payload": lambda: [payload(p) for p in ("period", "dip")], "leg": leg,
          "standing": lambda: [standing(p) for p in ("pulse", "period")], "compound": compound}
