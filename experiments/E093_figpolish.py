"""E093 — T005 figure polish (Buzi's review notes), regenerated from on-disk data with the paper naming
(the system = "ODD-conditioned"; V1 = "ODD-conditioned (direct)"; no V2-variant discussion).

F2 certifiable_regions : keep the geometry; ALL overlay text -> a proper legend (proxy handles), no inline
                         labels, handoff numbers in panel subtitles.
F7 cross_demo          : σ-across-states bars swamped the means -> LEFT: means with light σ bands;
                         RIGHT: the discrimination directly, d(x) = (V(0)-V(x))/σ̄  (contraction in σ units).
F3 mode_ribbon         : audit y-limits from the DATA (no clipping), label the switch window, new names.
Outputs overwrite $ODD_ARTIFACTS/E077-figures/ (default ~/artifacts/odd-conditioned) and copy into PAPER-draft/figs.
"""
import os, sys, json, shutil
from _paths import _ART
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({                       # publication sizes (Buzi: figures shrink in the paper document)
    "font.size": 14, "axes.titlesize": 15, "axes.labelsize": 14,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 13, "figure.titlesize": 17,
})
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

ART = os.path.expanduser(_ART)
OUT = f"{ART}/E077-figures"
PD = f"{ART}/PAPER-draft/figs"
SYS = "ODD-conditioned"
FORCE_MAX = 50.0          # evaluation env force_max (N): a force_scale s is a lateral pull of s * FORCE_MAX

# Every number drawn on a figure is computed here: protocol constants come from the script that ran the
# experiment, statistics from its saved results. Nothing in this file states a result.


def _path_at_gusts(mod, kind):
    """ODD value at each gust start of a demo ramp (the spikes drawn on the demo path), from the ramp script."""
    out = []
    for g in mod.GUST_STARTS:
        u = min(max((g - mod.RAMP_START) / (mod.RAMP_END - mod.RAMP_START), 0.0), 1.0)
        out.append(mod.W_MAX * u if kind == "W" else mod.TH_HI + (mod.TH_LO - mod.TH_HI) * u)
    return out


def fig2():
    d = json.load(open(f"{OUT}/F2_grid.json"))
    WS, THS, PULLS = d["WS"], d["THS"], d["PULLS"]
    pn = [p * 50 for p in PULLS]

    def grid(res, keys):
        return np.array([[res[f"{a}|{fr}"] for a in keys] for fr in PULLS])

    fw = grid(d["res"]["w_stand"], WS)
    fc = grid(d["res"]["c_stand"], THS)
    # the two demo ramps: protocol from their scripts, handoff statistics from their results
    import E075_ramp as RW, E078_ramp as RC
    hw = json.load(open(f"{ART}/E075-recal-eval/partA_ramp.json"))["arms"]["HANDOFF(recal)"]
    hc = json.load(open(f"{ART}/E078-compound-demo/task3_ramp.json"))["arms"]["HANDOFF"]
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.2))
    cf = None
    for ax, F, xs, xlab, title, sub, star_x, mod, kind in (
        (axes[0], fw, WS, "carried load W (N)", "Weight ladder: (W × disturbance) plane",
         f"certified handoff at W ≈ {hw['switchW_mean']:.0f} ± {hw['switchW_std']:.0f} N",
         hw["switchW_mean"], RW, "W"),
        (axes[1], fc, THS, f"FR-leg torque fraction θ   (W = {RC.W:.0f} N carried)",
         "Compound: (θ × disturbance) plane",
         f"certified handoff at θ ≈ {hc['switchTheta_mean']:.2f} ± {hc['switchTheta_std']:.2f}",
         hc["switchTheta_mean"], RC, "theta"),
    ):
        amb, gust = mod.AMBIENT * FORCE_MAX, mod.GUST_SCALE * FORCE_MAX
        cf = ax.contourf(xs, pn, F, levels=np.linspace(0, 1, 11), cmap="RdYlGn_r")
        ax.contour(xs, pn, F, levels=[0.2], colors="k", linewidths=2.6)
        both_bad = F > 0.8
        if both_bad.any():
            ax.contourf(xs, pn, both_bad.astype(float), levels=[0.5, 1.5],
                        colors="none", hatches=["////"])
        ax.axhline(amb, color="#1a5276", lw=2.0)
        for x in _path_at_gusts(mod, kind):
            ax.plot([x, x], [amb, gust], color="#1a5276", lw=1.6)
        ax.plot(star_x, amb, marker="*", ms=20, color="#f1c40f", mec="k", mew=0.8)
        ax.set_title(f"{title}\n{sub}", fontsize=15)
        ax.set_xlabel(xlab)
        if xs == THS:
            ax.invert_xaxis()
    axes[0].set_ylabel("disturbance pull (N)")
    handles = [
        Line2D([], [], color="k", lw=2.6, label="STAND certifiable boundary (20% failure)"),
        Line2D([], [], color="#1a5276", lw=2.0,
               label=f"demo ODD path ({RW.AMBIENT * FORCE_MAX:.0f} N ambient, gusts to {RW.GUST_SCALE * FORCE_MAX:.0f} N)"),
        Line2D([], [], marker="*", ms=15, color="#f1c40f", mec="k", ls="",
               label=f"certified handoff ({SYS} trigger)"),
        Patch(facecolor="none", hatch="////", edgecolor="k", label="neither mode certifiable"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=14, frameon=True,
               bbox_to_anchor=(0.44, -0.10))
    cb = fig.colorbar(cf, ax=axes, fraction=0.035, pad=0.02)
    cb.set_label("STAND-policy failure fraction (tip ∨ slam)")
    fig.suptitle("Certifiable-region geometry: STAND is an island, REST covers the plane —\n"
                 f"the {SYS} handoff fires where the demo path exits the island", fontsize=16, y=1.04)
    fig.savefig(f"{OUT}/F2_certifiable_regions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig7():
    w74 = json.load(open(f"{ART}/E074-hicom-demo/task2_value.json"))
    wrc = json.load(open(f"{ART}/E075-recal-eval/partA_value.json"))
    leg = json.load(open(f"{ART}/E076-leg-demo/partB_value.json"))
    comp = json.load(open(f"{ART}/E078-compound-demo/task2_value.json"))

    def curve(rows, vkey, skey, xfn):
        ks = sorted(rows, key=lambda k: float(k))
        x = np.array([xfn(float(k)) for k in ks])
        v = np.array([rows[k][vkey] for k in ks])
        s = np.array([rows[k][skey] for k in ks])
        o = np.argsort(x)
        return x[o], v[o], s[o]

    curves = []
    xw, vw, sw = curve(wrc["rows"], "V_recal", "V_recal_std", lambda W: W / 250.0)
    curves.append(("Weight (raised-CoM load)", "#1f4e79", xw, vw, sw))
    xl, vl, sl = curve(leg["rows"], "V", "V_std", lambda t: 1.0 - t)
    curves.append(("Unloaded leg (negative control)", "#7f8c8d", xl, vl, sl))
    if comp:
        xc, vc, sc = curve(comp["rows"], "V", "V_std", lambda t: 1.0 - t)
        curves.append(("Compound (leg death while loaded)", "#c0392b", xc, vc, sc))

    fig, (aL, aR) = plt.subplots(1, 2, figsize=(14.5, 5.8))
    for name, col, x, v, s in curves:
        aL.plot(x, v, color=col, lw=2.4, marker="o", ms=4, label=name)
        aL.fill_between(x, v - s, v + s, color=col, alpha=0.13)
    aL.axhline(0, color="k", lw=1)
    aL.set_xlabel("normalized ODD degradation (0 = nominal → 1 = maximal)")
    aL.set_ylabel("V̄_stand  (mean; band = ±σ over states)", labelpad=8)
    aL.set_title("Certificate value along each ODD axis", fontsize=15)
    aL.legend(fontsize=14, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1, frameon=False)
    aL.grid(alpha=0.25)
    for name, col, x, v, s in curves:
        sbar = max(float(np.mean(s)), 1e-6)
        dsc = (v[0] - v) / sbar
        aR.plot(x, dsc, color=col, lw=2.4, marker="o", ms=4,
                label=f"{name} — final {dsc[-1]:.2f}σ")
    aR.axhline(1.0, color="k", ls=":", lw=1.2)
    aR.text(0.02, 1.05, "1σ contraction", fontsize=16)
    aR.axhline(0, color="k", lw=1)
    aR.set_xlabel("normalized ODD degradation (0 = nominal → 1 = maximal)")
    aR.set_ylabel("contraction  (V(0) − V(x)) / σ̄   [σ units]")
    aR.set_title("The discrimination directly: contraction in σ units", fontsize=15)
    aR.legend(fontsize=14, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1, frameon=False)
    aR.grid(alpha=0.25)
    fig.suptitle("The stand certificate contracts iff the ODD change is non-absorbable — "
                 "the filter knows when it is needed", fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT}/F7_cross_demo_contraction.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig3():
    d = json.load(open(f"{ART}/E074-hicom-demo/task3_ramp.json"))
    cfg = d.get("config", {})
    W = np.array(d["W_trace"])
    DT = cfg.get("dt", 0.02)
    t = np.arange(len(W)) * DT
    arms = d["arms"]
    NAME = {"STAND-ONLY": ("STAND-ONLY", "#e74c3c"), "REST-ONLY": ("REST-ONLY", "#1e8449"),
            "HANDOFF": (SYS, "#1a5276"), "UNIFIED": ("Unified-spec baseline", "#8e44ad")}
    H = arms["HANDOFF"]
    vs = np.array(H["Vs_trace"]); vu = np.array(arms["UNIFIED"]["Vs_trace"]) if "Vs_trace" in arms["UNIFIED"] else np.array(H["Vu_trace"])
    vu = np.array(H["Vu_trace"])
    sw = np.array(H["sw_trace"])
    eps = cfg.get("eps", -0.04)
    g0s = cfg.get("gust_starts", []); gn = cfg.get("gust_steps", 25)
    gusts = [(s0 * DT, (s0 + gn) * DT) for s0 in g0s]
    fig, axes = plt.subplots(4, 1, figsize=(13.5, 12),
                             gridspec_kw={"height_ratios": [1.1, 1.6, 2.0, 0.45]}, sharex=True)
    aW, aV, aH, aM = axes

    def shade(ax):
        for (g0, g1) in gusts:
            ax.axvspan(g0, g1, color="#f6b26b", alpha=0.35)

    aW.plot(t, W, color="k", lw=2)
    shade(aW)
    aW.set_ylabel("load W (N)")
    aW.set_title("Dynamic ODD: growing load + gusts (orange)", fontsize=15)

    aV.plot(t[:len(vs)], vs, color="#1a5276", lw=1.8, label="V$_{stand}$ (family)")
    aV.plot(t[:len(vu)], vu, color="#8e44ad", lw=1.8, label="V$_{unified}$ (merged spec)")
    aV.axhline(eps, color="r", ls=":", lw=1.6, label="trigger ε")
    if sw.max() > 0.05:
        i0 = int(np.argmax(sw > 0.1)); i1 = int(np.argmax(sw > 0.9)) or len(sw) - 1
        aV.axvspan(t[i0], t[i1], color="#95a5a6", alpha=0.25, label="switch window (fleet 10–90%)")
    lo = min(vs.min(), vu.min(), eps) - 0.05
    hi = max(vs.max(), vu.max()) + 0.05
    aV.set_ylim(lo, hi)                      # audited: NO clipping (Buzi note 3)
    shade(aV)
    aV.set_ylabel("certificate value")
    aV.set_title("The certification deficit: V$_{stand}$ falls, V$_{unified}$ rises — "
                 "only the family signals the switch", fontsize=15)
    aV.legend(fontsize=14, ncol=1, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)

    hmin, hmax = 1e9, -1e9
    for k, (nm, col) in NAME.items():
        h = np.array(arms[k]["h_trace"])
        aH.plot(t[:len(h)], h, color=col, lw=2, label=nm)
        hmin = min(hmin, h.min()); hmax = max(hmax, h.max())
    aH.set_ylim(hmin - 0.01, hmax + 0.01)    # audited limits
    shade(aH)
    aH.set_ylabel("base height (m)")
    aH.legend(fontsize=14, ncol=1, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)

    frac = np.clip(sw, 0, 1)
    aM.imshow(frac[None, :], aspect="auto", cmap="RdBu", vmin=0, vmax=1,
              extent=[t[0], t[min(len(sw), len(t)) - 1], 0, 1])
    aM.set_yticks([])
    aM.set_ylabel("mode", rotation=0, ha="right", va="center")
    aM.text(t[len(t) // 8], 0.5, "STAND", color="w", fontsize=14, fontweight="bold", va="center")
    aM.text(t[int(len(t) * 0.85)], 0.5, "REST", color="w", fontsize=14, fontweight="bold", va="center")
    aM.set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/F3_mode_ribbon_timeline.png", dpi=150, bbox_inches="tight")
    plt.close(fig)




# ── part 2: renaming pass over the current headline figures (mask V2-REUSE; paper names) ──
LBL = {"V2": SYS, "V2-REUSE": None, "V1": f"{SYS} (direct)", "ONE-WAY": "ONE-WAY",
       "WALK-ONLY": "WALK-ONLY", "REST-ONLY": "REST-ONLY", "STAND-ONLY": "STAND-ONLY"}
CL = {"V2": "#1a5276", "V1": "#8e44ad", "ONE-WAY": "#e67e22", "WALK-ONLY": "#c0392b",
      "REST-ONLY": "#1e8449", "STAND-ONLY": "#e74c3c"}


def walking_claims():
    import math
    sys.path.insert(0, "experiments")
    import E092_payload_walk as E92
    d = json.load(open(f"{ART}/E092-payload-walk/results.json"))
    for sched in ("pulse", "period", "dip"):
        t = np.arange(E92.STEPS) * E92.DT
        Wv = np.array([E92.Wh_of(sched, k)[0] for k in range(E92.STEPS)])
        fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2))
        for ax, key, ylab, title in ((axes[0], "S", "survival S(t)", "Safety (death = flip-over)"),
                                     (axes[1], "SUC", "success CDF (fraction of fleet at goal)",
                                      "Permissiveness (goal completions over time)")):
            ax2 = ax.twinx()
            ax2.fill_between(t, Wv, color="#888", alpha=0.22)
            ax2.plot(t, Wv, color="#555", lw=2.5, label="payload W(t)")
            ax2.set_ylim(0, 900); ax2.set_yticks([E92.W_TRIG, E92.UP_W_GATE, E92.W_HI])
            ax2.tick_params(labelsize=11, colors="#555")
            for arm in ("WALK-ONLY", "REST-ONLY", "ONE-WAY", "V1", "V2"):
                if LBL[arm] is None or f"{sched}|{arm}" not in d:
                    continue
                r = d[f"{sched}|{arm}"]
                ax.plot(t[:len(r[key])], r[key], color=CL[arm], lw=2.4 if arm == "V2" else 1.5,
                        label=f"{LBL[arm]} ({r[key][-1]:.2f})")
            ax.set_xlim(0, t[-1]); ax.set_ylim(-0.05, 1.05); ax.grid(alpha=0.25)
            ax.set_xlabel("time (s)"); ax.set_ylabel(ylab); ax.set_title(title, fontsize=15)
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, fontsize=12, loc="upper center",
                      bbox_to_anchor=(0.5, -0.20), ncol=2, frameon=False, columnspacing=1.0)
        fig.suptitle(f"Payload-swap walking — {sched} (tall crate {E92.W_LO:.0f}→{E92.W_HI:.0f} N, CoM "
                     f"{E92.H_LO:.2f}→{E92.H_HI:.2f} m; N={E92.N}, no respawn)", y=1.05, fontsize=15)
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.34)
        fig.savefig(f"{ART}/E092-payload-walk/claims_{sched}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    print("E092 claims figures renamed")


def standing_survival():
    sys.path.insert(0, "experiments")
    import E084_automaton as E84
    for fn, tag, conds in [
        (f"{ART}/E084-automaton/results.json", "waves",
         [("benign", "square"), ("benign", "sine"), ("gusty", "square"), ("gusty", "sine")]),
        (f"{ART}/E084-automaton/results_single.json", "single",
         [("benign", "pulse"), ("benign", "period"), ("gusty", "pulse"), ("gusty", "period")])]:
        d = json.load(open(fn))
        fig, axes = plt.subplots(1, 4, figsize=(20, 3.4), sharey=True)
        steps = max(len(v["S"]) for v in d.values())
        t = np.arange(steps) * E84.DT
        for axx, (cond, sched) in zip(axes, conds):
            for arm in ("V2", "V1", "ONE-WAY", "STAND-ONLY", "REST-ONLY"):
                S = d[f"{cond}|{sched}|{arm}"]["S"]
                axx.plot(t[:len(S)], S, color=CL[arm], lw=2 if arm in ("V2", "REST-ONLY") else 1.4,
                         label=LBL[arm] if (cond, sched) == conds[0] else None)
            axx.set_title(f"{sched} / {cond}", fontsize=15)
            axx.set_xlabel("time (s)"); axx.set_ylim(-0.05, 1.05); axx.grid(alpha=0.25)
        axes[0].set_ylabel("survival S(t) — env-spec accounting")
        hh, ll = axes[0].get_legend_handles_labels()
        fig.legend(hh, ll, fontsize=14, loc="upper center", bbox_to_anchor=(0.5, 0.04), ncol=5, frameon=False)
        fig.suptitle(f"Standing automaton ({tag}): {SYS} vs baselines (N={E84.N}, no respawn)", y=1.02, fontsize=15)
        fig.tight_layout()
        fig.savefig(f"{ART}/E084-automaton/survival_{tag}.png", dpi=140, bbox_inches="tight")
        plt.close(fig)
    print("standing survival figures renamed")


def walking_topdown():
    sys.path.insert(0, "experiments")
    import E092_payload_walk as E92
    LBLT = {"V2": SYS, "V1": f"{SYS} (direct)", "ONE-WAY": "ONE-WAY",
            "WALK-ONLY": "WALK-ONLY", "REST-ONLY": "REST-ONLY"}
    for sched in ("pulse", "period", "dip"):
        z = np.load(f"{ART}/E092-payload-walk/traj_{sched}.npz")
        arms = ["WALK-ONLY", "REST-ONLY", "ONE-WAY", "V1", "V2"]
        fig, axes = plt.subplots(1, len(arms), figsize=(4.4 * len(arms), 4.9), sharex=True, sharey=True)
        for ax, arm in zip(axes, arms):
            yaw0 = z[f"{arm}_yaw0"]; spawn = z[f"{arm}_spawn"]; traj = z[f"{arm}_traj"]
            alv = z[f"{arm}_alv_tr"]; reached = z[f"{arm}_reached_mask"]; alive = z[f"{arm}_alive_mask"]
            c, s = np.cos(-yaw0), np.sin(-yaw0)
            rel = traj - spawn[None]
            gx = rel[..., 0] * c[None] - rel[..., 1] * s[None]
            gy = rel[..., 0] * s[None] + rel[..., 1] * c[None]
            for i in range(gx.shape[1]):
                last = max(int(alv[:, i].sum()), 1)
                if reached[i]:
                    ax.plot(gx[:last, i], gy[:last, i], color="#1e8449", lw=0.7, alpha=0.35)
                elif not alive[i]:
                    ax.plot(gx[:last, i], gy[:last, i], color="#c0392b", lw=0.7, alpha=0.30)
                    ax.plot(gx[last - 1, i], gy[last - 1, i], "x", color="#c0392b", ms=4, alpha=0.7)
                else:
                    ax.plot(gx[:last, i], gy[:last, i], color="#777", lw=0.6, alpha=0.30)
            ax.add_patch(plt.Circle((E92.GOAL_D, 0), E92.GOAL_R, fill=False, color="#1a5276", lw=2))
            ax.plot(0, 0, "k^", ms=9)
            ax.set_title(f"{LBLT[arm]}\nreached {int(reached.sum())}/{len(reached)} | "
                         f"alive {int(alive.sum())}/{len(alive)}", fontsize=14)
            ax.set_xlim(-2, E92.GOAL_D + 2.5); ax.set_ylim(-5, 5); ax.set_aspect("equal"); ax.grid(alpha=0.2)
            ax.set_xlabel("progress toward goal (m)")
        axes[0].set_ylabel("lateral (m)")
        fig.suptitle(f"Payload-swap walking, top-down — {sched} "
                     "(green=reached, red=flipped at X, gray=alive short of goal)", y=1.03)
        fig.tight_layout()
        fig.savefig(f"{ART}/E092-payload-walk/topdown_{sched}.png", dpi=140, bbox_inches="tight")
        plt.close(fig)
    print("walking topdowns regenerated")


if __name__ == "__main__":
    fig2(); print("F2 done")
    fig7(); print("F7 done")
    fig3(); print("F3 done")
    os.makedirs(PD, exist_ok=True)
    for f in ("F2_certifiable_regions.png", "F7_cross_demo_contraction.png", "F3_mode_ribbon_timeline.png"):
        shutil.copy(f"{OUT}/{f}", f"{PD}/{f}")
    print("copied to PAPER-draft/figs")
    walking_claims()
    standing_survival()
    walking_topdown()
    for f in ("claims_pulse.png", "claims_period.png", "claims_dip.png",
              "topdown_pulse.png", "topdown_period.png", "topdown_dip.png"):
        shutil.copy(f"{ART}/E092-payload-walk/{f}", f"{PD}/{f}")
    for f in ("survival_waves.png", "survival_single.png"):
        shutil.copy(f"{ART}/E084-automaton/{f}", f"{PD}/{f}")
    print("all figures copied to PAPER-draft/figs")
