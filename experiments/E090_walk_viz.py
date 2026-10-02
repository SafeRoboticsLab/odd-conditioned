"""E090 — visualizations of the walking-with-safety-filter experiment (E089, corrected accounting).

1. Stats pass (N=256, record): per scenario (pulse/period x benign/gusty) x 5 arms ->
   - topdown_<sched>_<cond>.png : goal-frame top-down trajectories per arm (spawn at origin, goal at (9,0);
     green = reached, red + X = died (X at death point), gray = alive-not-reached)
   - dist_timeline_<sched>_<cond>.png : median (IQR band) distance-to-goal among live robots per arm + W(t)
   - traj npz per scenario for later reuse
2. Video pass (n=9, render): pulse/benign + period/benign, 3 panels WALK-ONLY | V2-REUSE | ONE-WAY,
   camera follows the robot, trunk tinted by automaton state, tag = arm [state] alive/reached counts,
   graph strip = W(t) + live median distance-to-goal per arm.
"""
import os, sys, math, json
from _paths import _ART
import numpy as np
sys.path.insert(0, "external/robot-safety-sandbox"); sys.path.insert(0, "experiments")
sys.path.insert(0, "external/go2_atomic_skills")
os.environ.setdefault("MUJOCO_GL", "egl")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio
try:
    from PIL import Image, ImageDraw
    HAVE_PIL = True
except Exception:
    HAVE_PIL = False
import E089_goal_walk as G

G.EPS_DN_WALK = -0.10
ARMS = ["WALK-ONLY", "REST-ONLY", "ONE-WAY", "V1", "V2-REUSE"]
VARMS = ["WALK-ONLY", "V2-REUSE", "ONE-WAY"]
COL = {"WALK-ONLY": "#c0392b", "REST-ONLY": "#1e8449", "ONE-WAY": "#e67e22", "V1": "#8e44ad", "V2-REUSE": "#1a5276"}
STNAME = {0: "WALK", 1: "DESCENDING", 2: "REST", 3: "GETTING-UP"}
OD = os.path.expanduser(_ART + "/E089-goal-walk")
DT4 = 4 * G.DT


def to_goal_frame(r):
    """Rotate world trajectories into each robot's spawn frame: spawn -> origin, goal -> (GOAL_D, 0)."""
    yaw0 = r["yaw0"]; spawn = r["spawn"]
    c, s = np.cos(-yaw0), np.sin(-yaw0)
    rel = r["traj"] - spawn[None]
    x = rel[..., 0] * c[None] - rel[..., 1] * s[None]
    y = rel[..., 0] * s[None] + rel[..., 1] * c[None]
    dxy = r["death_xy"]
    if len(dxy):
        # death entries are ragged over robots; transform with matched spawn/yaw is not tracked per-death,
        # so approximate: match by nearest final traj point is overkill — store world deaths per robot index
        pass
    return np.stack([x, y], axis=-1)      # (T, n, 2)


def topdown_fig(sched, cond, data):
    fig, axes = plt.subplots(1, len(ARMS), figsize=(4 * len(ARMS), 4.4), sharex=True, sharey=True)
    for ax, arm in zip(axes, ARMS):
        r = data[arm]
        gtraj = to_goal_frame(r)
        T, n, _ = gtraj.shape
        alv = r["alv_tr"]                                     # (T, n)
        reached = r["reached_mask"]; alive_end = r["alive_mask"]
        for i in range(n):
            aliv = alv[:, i]
            last = int(aliv.sum())                            # steps alive (prefix; death censors)
            tr = gtraj[:max(last, 1), i]
            if reached[i]:
                ax.plot(tr[:, 0], tr[:, 1], color="#1e8449", lw=0.7, alpha=0.35)
            elif not alive_end[i]:
                ax.plot(tr[:, 0], tr[:, 1], color="#c0392b", lw=0.7, alpha=0.30)
                ax.plot(tr[-1, 0], tr[-1, 1], "x", color="#c0392b", ms=4, alpha=0.7)
            else:
                ax.plot(tr[:, 0], tr[:, 1], color="#777", lw=0.6, alpha=0.30)
        gc = plt.Circle((G.GOAL_D, 0), G.GOAL_R, fill=False, color="#1a5276", lw=2)
        ax.add_patch(gc)
        ax.plot(0, 0, "k^", ms=8)
        ax.set_title(f"{arm}\nsucc {r['success']:.2f} / safe {r['safe']:.2f}", fontsize=10)
        ax.set_xlim(-2, 14.5); ax.set_ylim(-5, 5); ax.set_aspect("equal"); ax.grid(alpha=0.2)
        ax.set_xlabel("progress toward goal (m)")
    axes[0].set_ylabel("lateral (m)")
    fig.suptitle(f"E089 top-down trajectories — {sched}/{cond} (green=reached, red=died, gray=alive)", y=1.04)
    fig.tight_layout()
    fig.savefig(f"{OD}/topdown_{sched}_{cond}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def dist_timeline_fig(sched, cond, data):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax2 = ax.twinx()
    tt = np.arange(data[ARMS[0]]["traj"].shape[0]) * DT4
    Wv = np.array([G.W_of(sched, int(t / G.DT)) for t in tt])
    ax2.fill_between(tt, Wv, color="#777", alpha=0.10)
    ax2.plot(tt, Wv, color="#777", lw=1, alpha=0.6); ax2.set_ylim(0, 700)
    ax2.set_yticks([0, 130, 220]); ax2.set_ylabel("W (N)", color="#777")
    for arm in ARMS:
        r = data[arm]
        gtraj = to_goal_frame(r)
        dist = np.linalg.norm(gtraj - np.array([G.GOAL_D, 0.0])[None, None], axis=-1)   # (T, n)
        alv = r["alv_tr"].astype(bool)
        med = np.array([np.median(dist[k][alv[k]]) if alv[k].any() else np.nan for k in range(len(tt))])
        lo = np.array([np.percentile(dist[k][alv[k]], 25) if alv[k].any() else np.nan for k in range(len(tt))])
        hi = np.array([np.percentile(dist[k][alv[k]], 75) if alv[k].any() else np.nan for k in range(len(tt))])
        ax.plot(tt, med, color=COL[arm], lw=2, label=arm)
        ax.fill_between(tt, lo, hi, color=COL[arm], alpha=0.12)
    ax.axhline(G.GOAL_R, color="#1a5276", ls=":", lw=1)
    ax.text(0.2, G.GOAL_R + 0.15, "goal radius", fontsize=8, color="#1a5276")
    ax.set_xlim(0, tt[-1]); ax.set_ylim(0, 14)
    ax.set_xlabel("time (s)"); ax.set_ylabel("distance to goal among LIVE robots (m, median + IQR)")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title(f"E089 — {sched}/{cond}")
    fig.tight_layout()
    fig.savefig(f"{OD}/dist_timeline_{sched}_{cond}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def tag(img, txt):
    img = np.ascontiguousarray(img).copy(); img[:26] = (img[:26] * 0.3).astype(img.dtype)
    if HAVE_PIL:
        im = Image.fromarray(img); ImageDraw.Draw(im).text((8, 6), txt, fill=(255, 255, 255)); img = np.asarray(im)
    return img


def graph_frame(i, sched, meds, W, H, nfr):
    tt = np.arange(nfr) * 2 * G.DT
    Wv = np.array([G.W_of(sched, k * 2) for k in range(nfr)])
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = fig.add_subplot(111); ax2 = ax.twinx()
    ax2.fill_between(tt, Wv, color="#777", alpha=0.12)
    ax2.plot(tt[:i + 1], Wv[:i + 1], color="#444", lw=1.6, label="W(t)")
    ax2.set_ylim(0, 700); ax2.set_yticks([0, 130, 220]); ax2.set_ylabel("W (N)")
    for arm in VARMS:
        ax.plot(tt, meds[arm], color=COL[arm], lw=1.0, alpha=0.25)
        ax.plot(tt[:i + 1], meds[arm][:i + 1], color=COL[arm], lw=2.2, label=arm)
    ax.axhline(G.GOAL_R, color="#1a5276", ls=":", lw=1)
    ax.axvline(tt[i], color="#555", lw=1.5)
    ax.set_xlim(0, tt[-1]); ax.set_ylim(0, 14)
    ax.set_xlabel("time (s)"); ax.set_ylabel("median dist to goal (live)")
    ax.legend(loc="upper right", fontsize=7.5); ax2.legend(loc="upper left", fontsize=7.5)
    fig.tight_layout(pad=0.5); fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy(); plt.close(fig)
    return buf


if __name__ == "__main__":
    # ── pass 1: stats + figures ──
    for cond in ("benign", "medium", "gusty"):
        for sched in ("pulse", "period"):
            data = {}
            for arm in ARMS:
                r = G.rollout(sched, cond, arm, record=True)
                data[arm] = r
                print(f"stats {sched}/{cond} {arm}: succ {r['success']:.2f} safe {r['safe']:.2f}", flush=True)
            np.savez_compressed(f"{OD}/traj_{sched}_{cond}.npz",
                                **{f"{arm}_{k}": data[arm][k] for arm in ARMS
                                   for k in ("traj", "sttr", "alv_tr", "yaw0", "spawn", "goal",
                                             "reached_mask", "alive_mask")})
            topdown_fig(sched, cond, data)
            dist_timeline_fig(sched, cond, data)
            print(f"figures -> topdown/dist_timeline {sched}_{cond}", flush=True)

    # ── pass 2: 3-panel walking videos (benign) ──
    for sched, cond in (("pulse", "benign"), ("period", "benign"), ("pulse", "medium"), ("period", "medium")):
        R = {}
        for arm in VARMS:
            R[arm] = G.rollout(sched, cond, arm, n=9, record=True, render=True)
            print(f"video {sched} {arm}: succ {R[arm]['success']:.2f} safe {R[arm]['safe']:.2f}", flush=True)
        nfr = min(len(R[a]["frames"]) for a in VARMS)
        meds = {}
        for arm in VARMS:
            g = to_goal_frame(R[arm])
            dist = np.linalg.norm(g - np.array([G.GOAL_D, 0.0])[None, None], axis=-1)
            alv = R[arm]["alv_tr"].astype(bool)
            m = np.array([np.median(dist[k][alv[k]]) if alv[k].any() else np.nan for k in range(dist.shape[0])])
            meds[arm] = np.interp(np.arange(nfr) * 2, np.arange(dist.shape[0]) * 4, m)
        comb = []
        for i in range(nfr):
            row = []
            for arm in VARMS:
                r = R[arm]
                k4 = min(i // 2, r["alv_tr"].shape[0] - 1)
                nal = int(r["alv_tr"][k4].sum())
                stmaj = int(np.bincount(r["sttr"][k4][r["alv_tr"][k4].astype(bool)]).argmax()) if nal else 0
                nrc = int((np.nan_to_num(r["traj"][:k4 + 1] - r["goal"][None], nan=99)).shape[0] and 0)
                txt = f"{arm}"
                if arm == "V2-REUSE":
                    txt += f" [{STNAME[stmaj]}]"
                txt += f"   alive {nal}/9"
                row.append(tag(r["frames"][i], txt))
            top = np.hstack(row)
            TW = top.shape[1]; GH = 300
            comb.append(np.vstack([top, graph_frame(i, sched, meds, TW, GH, nfr)[:GH, :TW]]))
        vp = f"{OD}/walk_{sched}_benign.mp4"
        imageio.mimsave(vp, comb, fps=25, macro_block_size=1)
        print("video ->", vp, flush=True)
