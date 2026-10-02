"""E078 (T004, DEMO 2) Task 5 — F7 UPGRADE to 3 curves: the stand certificate vs a normalized ODD-degradation
axis, across THREE ODD types, on one shared axis. Overwrites E077-figures/F7_cross_demo_contraction.png.

  * WEIGHT  V_stand(W)      — E075 recal (weight-ladder ODD). Non-absorbable (raised-CoM instability) → CONTRACTS.
  * UNLOADED-LEG V(θ) FLAT  — E076 partB (leg-at-stance, the NEGATIVE CONTROL). Absorbable → stays flat.
  * COMPOUND V(θ)           — E078 task2 (leg death WHILE loaded). Non-absorbable → CONTRACTS.

Shared x = normalized ODD degradation (0 = nominal/benign → 1 = maximal): W/250 for weight, (1−θ) for the legs.
Title: the stand certificate contracts IFF the ODD change is non-absorbable — the filter knows when it is needed.
"""
import os, json
from _paths import _ART
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

A = os.path.expanduser(_ART)
WRC = f"{A}/E075-recal-eval/partA_value.json"        # weight (contracts)
LEG = f"{A}/E076-leg-demo/partB_value.json"          # unloaded leg (flat negative control)
CMP = f"{A}/E078-compound-demo/task2_value.json"     # compound (contracts) — the new curve
OUT = f"{A}/E077-figures"


def main():
    wrc = json.load(open(WRC)); leg = json.load(open(LEG)); cmp = json.load(open(CMP))

    # WEIGHT (recal): rows keyed by W (N); degradation = W/250
    Wr = sorted(int(k) for k in wrc["rows"])
    vW = [wrc["rows"][str(W)]["V_recal"] for W in Wr]
    sW = [wrc["rows"][str(W)]["V_recal_std"] for W in Wr]
    xW = np.array(Wr) / 250.0

    # LEG (unloaded, flat): rows keyed by θ; degradation = 1-θ
    Tl = sorted((float(k) for k in leg["rows"]), reverse=True)
    vL = [leg["rows"][str(t)]["V"] for t in Tl]
    sL = [leg["rows"][str(t)]["V_std"] for t in Tl]
    xL = 1.0 - np.array(Tl)

    # COMPOUND (loaded leg, contracts): rows keyed by θ; degradation = 1-θ
    Tc = sorted((float(k) for k in cmp["rows"]), reverse=True)
    vC = [cmp["rows"][str(t)]["V"] for t in Tc]
    sC = [cmp["rows"][str(t)]["V_std"] for t in Tc]
    xC = 1.0 - np.array(Tc)

    plt.rcParams.update({"font.size": 12})
    fig, ax = plt.subplots(figsize=(11.5, 6.6))
    ax.axhline(0, color="#888", lw=1.0, zorder=1)
    ax.axhspan(-0.6, 0, color="#fdecea", alpha=0.6, zorder=0)   # "certificate lost" region
    ax.text(0.015, -0.055, "V < 0 : stance no longer certifiable  →  handoff needed",
            color="#b03a2e", fontsize=10, style="italic")

    ax.errorbar(xC, vC, yerr=sC, color="#c0392b", lw=3.0, marker="D", ms=7, capsize=3, zorder=5,
                label=f"COMPOUND  leg death WHILE loaded (E078)  —  CONTRACTS, discrim {cmp['discrim']:.2f}")
    ax.errorbar(xW, vW, yerr=sW, color="#1a5276", lw=2.6, marker="s", ms=6, capsize=3, zorder=4,
                label=f"WEIGHT  raised-CoM load (E075)  —  CONTRACTS, discrim {wrc.get('discrim_recal', 1.16):.2f}")
    ax.errorbar(xL, vL, yerr=sL, color="#7f8c8d", lw=2.4, marker="o", ms=6, capsize=3, zorder=3,
                label=f"UNLOADED LEG  leg-at-stance (E076, NEGATIVE CONTROL)  —  FLAT, discrim {leg['discrim']:.2f}")
    ax.axhline(cmp["eps"], color="#c0392b", ls=":", lw=1.6, alpha=0.8,
               label=f"compound handoff ε = {cmp['eps']:+.3f}")

    ax.set_xlabel("normalized ODD degradation      0 = nominal  →  1 = maximal   "
                  "(W/250N for weight;  1−θ for the legs)", fontsize=11.5)
    ax.set_ylabel("V_stand   (reach-avoid value;   ≥ 0 = stance certifiable)")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.55, 0.12)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=10, loc="lower left", framealpha=0.95)
    ax.set_title("The stand certificate contracts iff the ODD change is non-absorbable\n"
                 "— the filter knows when it is needed —", fontsize=14, pad=12)
    # annotate the compound crossing (θ_c band)
    ax.annotate("compound crosses ε near θ_c≈0.25\n(the certified leg handoff)",
                xy=(0.8, cmp["rows"]["0.2"]["V"]), xytext=(0.42, -0.34), fontsize=10, color="#c0392b",
                arrowprops=dict(arrowstyle="->", color="#c0392b"))
    ax.annotate("unloaded leg: reactive policy\nabsorbs the axis → V never contracts",
                xy=(0.9, vL[-1]), xytext=(0.5, 0.045), fontsize=10, color="#5d6d7e",
                arrowprops=dict(arrowstyle="->", color="#7f8c8d"))

    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "F7_cross_demo_contraction.png")
    fig.savefig(p, dpi=140, bbox_inches="tight")
    print("wrote", p)
    print(f"  compound discrim {cmp['discrim']:.2f} | weight {wrc.get('discrim_recal', 1.16):.2f} | leg (flat) {leg['discrim']:.2f}")


if __name__ == "__main__":
    main()
