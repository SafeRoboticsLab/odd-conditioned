"""E077 — F7 CROSS-DEMO figure: the stand certificate V vs a normalized "ODD degradation" axis, two ODD types.

LEFT  : V_stand(W) — weight-ladder ODD. E074 stand_hi (task2_value.json) + the E075 RECAL curve (partA_value).
RIGHT : V_leg_stand(θ) — leg-degradation ODD (E076 partB_value.json).
Shared x-axis = normalized ODD degradation (0 = nominal / benign, 1 = maximal degradation): W/250 for weights,
(1-θ) for the leg.

HONEST FINDING: the weight certificate CONTRACTS cleanly (crosses 0 -> deeply negative as W grows); the leg
certificate stays FLAT (~0, non-monotone) because the reactive stance policy ABSORBS the single-axis leg ODD
(E064/E066/E068 arc). So the intended "contracts across ODD types" signature holds for WEIGHT but NOT for the
leg — the figure is a CONTRAST, and the title says so.
"""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

W74 = os.path.expanduser("~/artifacts/odd-conditioned/E074-hicom-demo/task2_value.json")
WRC = os.path.expanduser("~/artifacts/odd-conditioned/E075-recal-eval/partA_value.json")
LEG = os.path.expanduser("~/artifacts/odd-conditioned/E076-leg-demo/partB_value.json")
OUT = os.path.expanduser("~/artifacts/odd-conditioned/E077-figures")


def main():
    w74 = json.load(open(W74)); wrc = json.load(open(WRC)); leg = json.load(open(LEG))
    # weight curves
    Ws = sorted(int(k) for k in w74["rows"])
    v74 = [w74["rows"][str(W)]["V_stand"] for W in Ws]
    s74 = [w74["rows"][str(W)]["V_stand_std"] for W in Ws]
    Wr = sorted(int(k) for k in wrc["rows"])
    vrc = [wrc["rows"][str(W)]["V_recal"] for W in Wr]
    src = [wrc["rows"][str(W)]["V_recal_std"] for W in Wr]
    xW = np.array(Ws) / 250.0
    xWr = np.array(Wr) / 250.0
    # leg curve
    Ts = sorted((float(k) for k in leg["rows"]), reverse=True)   # 1.0 -> 0.0
    vl = [leg["rows"][str(t) if str(t) in leg["rows"] else f"{t}"]["V"] for t in Ts]
    sl = [leg["rows"][str(t)]["V_std"] for t in Ts]
    xL = 1.0 - np.array(Ts)

    plt.rcParams.update({"font.size": 12})
    fig, (aL, aR) = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)

    # LEFT: weight
    aL.axhline(0, color="#999", lw=0.9)
    aL.errorbar(xW, v74, yerr=s74, color="#8e44ad", lw=2.4, marker="o", ms=5, capsize=3,
                label="V_stand_hi (E074, trained W~U[0,150])", alpha=0.9)
    aL.errorbar(xWr, vrc, yerr=src, color="#1a5276", lw=2.6, marker="s", ms=5, capsize=3,
                label="V_stand RECAL (E075, W~U[0,120])")
    aL.axhline(wrc["eps"], color="r", ls=":", lw=1.6, label=f"recal ε={wrc['eps']:+.3f}")
    aL.set_title("WEIGHT-ladder ODD: certificate CONTRACTS", fontsize=13, color="#1a5276")
    aL.set_xlabel("ODD degradation   W/250N  (0=no load → 1=250N)")
    aL.set_ylabel("V_stand  (reach-avoid value;  ≥0 = stance certifiable)")
    aL.legend(fontsize=9, loc="lower left"); aL.grid(alpha=0.25)
    aL.set_ylim(-0.55, 0.20)
    # secondary top axis: actual W
    axt = aL.twiny(); axt.set_xlim(aL.get_xlim()); axt.set_xticks(np.array(Ws)/250.0)
    axt.set_xticklabels([str(W) for W in Ws], fontsize=8); axt.set_xlabel("W (N)", fontsize=9)

    # RIGHT: leg
    aR.axhline(0, color="#999", lw=0.9)
    aR.errorbar(xL, vl, yerr=sl, color="#c0392b", lw=2.6, marker="D", ms=5, capsize=3,
                label="V_leg_stand (E076, trained θ~U[0.5,1.0])")
    aR.axhline(leg["eps"], color="r", ls=":", lw=1.6, label=f"ε={leg['eps']:+.3f}")
    aR.axhspan(-0.55, 0.20, xmin=0.0, xmax=1.0, color="#f5f5f5", zorder=0)
    aR.set_title(f"LEG-degradation ODD: certificate FLAT (absorbed)\n"
                 f"discrimination |Δ|/σ = {leg['discrim']:.2f}", fontsize=13, color="#c0392b")
    aR.set_xlabel("ODD degradation   (1−θ)  (0=full torque → 1=dead leg)")
    aR.legend(fontsize=9, loc="lower left"); aR.grid(alpha=0.25)
    axt2 = aR.twiny(); axt2.set_xlim(aR.get_xlim()); axt2.set_xticks(1.0 - np.array(Ts))
    axt2.set_xticklabels([f"{t:.1f}" for t in Ts], fontsize=8); axt2.set_xlabel("θ (FR torque frac)", fontsize=9)

    fig.suptitle("The stand certificate contracts with ODD degradation — but only when the ODD is not "
                 "reactively absorbable\n(weight = raised-CoM instability contracts V; leg-torque is absorbed "
                 "at stance → V stays flat: the E064/E068 lesson, at the certificate level)",
                 fontsize=12.5, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "F7_cross_demo_contraction.png")
    fig.savefig(p, dpi=140, bbox_inches="tight")
    print("wrote", p)


if __name__ == "__main__":
    main()
