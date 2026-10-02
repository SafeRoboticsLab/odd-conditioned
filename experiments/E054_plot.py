"""Plot the E054 payoff figure: survival vs FR-torque, specialists (frontier) + blind/conditioned/history."""
import json, os
from _paths import _ART
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = os.path.expanduser(_ART + "/E054-conditioned-eval")
d = json.load(open(f"{out}/results.json"))
forces = sorted({float(k.split("|")[0]) for k in d})
pcts = [100, 70, 50, 30, 20, 10]
STYLE = {  # name -> (color, linestyle, marker, z)
    "spec normal(1.0)": ("#7fb3d5", "--", "o", 1), "spec weak50(0.5)": ("#7dcea0", "--", "s", 1),
    "spec weak20(0.2)": ("#f1948a", "--", "^", 1),
    "blind": ("#e67e22", "-", "o", 3), "conditioned": ("#1a5276", "-", "D", 4), "history": ("#8e44ad", "-", "v", 3),
}
fig, axes = plt.subplots(1, len(forces), figsize=(14, 5), sharey=True)
if len(forces) == 1: axes = [axes]
for ax, f in zip(axes, forces):
    for name, (c, ls, mk, z) in STYLE.items():
        ys = [d.get(f"{f}|{name}|{p}", [float('nan')])[0] for p in pcts]
        ax.plot(pcts, ys, ls, marker=mk, color=c, lw=2.2 if ls == "-" else 1.4,
                ms=7 if ls == "-" else 5, label=name, zorder=z, alpha=0.95 if ls == "-" else 0.7)
    ax.set_title(f"scripted pull {int(f*50)} N"); ax.set_xlabel("FR-leg allowable torque at TEST (%)")
    ax.invert_xaxis(); ax.grid(alpha=0.3)
axes[0].set_ylabel("falls / env-second  (lower = safer)")
axes[0].legend(fontsize=8.5, title="dashed=fixed-θ specialists\nsolid=one policy over θ")
fig.suptitle("E054 — Value of ODD-conditioning (leg torque): does one conditioned policy match the specialist frontier?", fontsize=12.5)
fig.tight_layout(); fig.savefig(f"{out}/payoff_curves.png", dpi=130, bbox_inches="tight")
print("saved", f"{out}/payoff_curves.png")
