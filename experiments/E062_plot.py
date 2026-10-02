"""Draw the E062 ODD-dynamics figure from saved traces (no rollouts). Edit + rerun freely for style tweaks."""
import os, numpy as np
from _paths import _ART
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.expanduser(_ART + "/E062-odd-dynamics")
d = np.load(f"{OUT}/traces.npz", allow_pickle=True)
STEPS, T0, DT = int(d["steps"]), int(d["t0"]), float(d["dt"])
SCHED = list(d["sched"]); ARMS = ["blind", "history", "conditioned"]
COL = {"blind": "#e67e22", "history": "#8e44ad", "conditioned": "#1a5276"}
tvec = np.arange(STEPS) * DT

plt.rcParams.update({"font.size": 16})
# WIDE, rectangular survival panels: large width, thin top θ strip.
fig, axes = plt.subplots(2, len(SCHED), figsize=(6.8 * len(SCHED), 6.0),
                         gridspec_kw={"height_ratios": [0.42, 1.58]})
for j, s in enumerate(SCHED):
    axes[0, j].plot(tvec, d[f"theta|{s}"], color="#333", lw=2.5)
    axes[0, j].axvline(T0 * DT, color="r", ls="--", lw=1.4, alpha=.6)
    axes[0, j].set_title(s, fontsize=24, fontweight="bold"); axes[0, j].set_ylim(0, 1.1)
    axes[0, j].set_ylabel("θ" if j == 0 else "", fontsize=20); axes[0, j].set_xticklabels([])
    axes[0, j].tick_params(labelsize=15)
    for n in ARMS:
        axes[1, j].plot(tvec, d[f"surv|{s}|{n}"], color=COL[n], lw=3.2, label=n)
    axes[1, j].axvline(T0 * DT, color="r", ls="--", lw=1.4, alpha=.6); axes[1, j].set_ylim(0, 1.02)
    axes[1, j].set_xlabel("time (s)", fontsize=20); axes[1, j].grid(alpha=.3); axes[1, j].tick_params(labelsize=16)
    if j == 0:
        axes[1, j].set_ylabel("survival", fontsize=22)
        axes[1, j].legend(fontsize=20, loc="lower left", handlelength=2.4, borderpad=0.9, labelspacing=0.7)
fig.suptitle("ODD-change dynamics (θ shape) × arm — soft-rest, 15 N.  conditioned (knows live θ) tracks every shape.",
             fontsize=23)
fig.tight_layout()
fig.savefig(f"{OUT}/dynamics_survival.png", dpi=130, bbox_inches="tight")
print("saved", f"{OUT}/dynamics_survival.png")
