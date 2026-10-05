"""Figures, regenerated from the saved results (run the matching evaluate targets first). Every number drawn is
computed here from data or taken from the protocol definitions in scenarios.py / certificates.py.

    certifiable_regions.png      region grid + where each demo's handoff fired          (certificates)
    certificate_contraction.png  V_stand along the weight / leg / compound axes          (certificates)
    certification_deficit.png    the wide-stand weight ramp: V_stand vs V_unified, modes (certificates)
    compound_timeline.png        leg dies while carrying 80 N: θ, V_stand, base heights  (certificates)
    payload_claims_<p>.png       survival and goal-completion curves                      (payload)
    payload_topdown_<p>.png      trajectories in each robot's goal frame                   (payload)
    leg_topdown.png              the same for the leg-fault walk                           (leg)
    standing_survival_<t>.png    survival curves of the standing automaton                 (standing)
"""
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from . import certificates as C  # noqa: E402
from .automaton import LOAD_NOMINAL, label  # noqa: E402
from .paths import OUTPUTS, output_dir  # noqa: E402
from .scenarios import SCENARIOS  # noqa: E402
from .sim import DT, FORCE_MAX  # noqa: E402

plt.rcParams.update({"font.size": 14, "axes.titlesize": 15, "axes.labelsize": 14, "xtick.labelsize": 12,
                     "ytick.labelsize": 12, "legend.fontsize": 13, "figure.titlesize": 17})
SYS = "ODD-conditioned"
COL = {"odd": "#1a5276", "direct": "#8e44ad", "one-way": "#e67e22", "task-only": "#c0392b", "rest-only": "#1e8449",
       "odd-rest-descent": "#5dade2"}
W_FULL = 250.0          # top of the load ladder (N): normalizes the weight axis to [0, 1]


def _load(*path):
    f = os.path.join(OUTPUTS, *path)
    if not os.path.exists(f):
        raise FileNotFoundError(f)
    return json.load(open(f))


def _save(fig, name, dpi=150):
    f = os.path.join(output_dir("figures"), name)
    fig.savefig(f, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"figure -> {f}", flush=True)


# ── certificates ──────────────────────────────────────────────────────────────────────────────────────────

def certifiable_regions():
    g = _load("certificates", "region_grid.json")
    hw = _load("certificates", "ramp_weight.json")["arms"]["HANDOFF"]
    hc = _load("certificates", "ramp_compound.json")["arms"]["HANDOFF"]
    pn = g["push_N"]
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.2))
    cf = None
    for ax, panel, h, rp, xlab, title, sub in (
            (axes[0], "weight", hw, C.RAMPS["weight"], "carried load W (N)", "Weight ladder: (W × disturbance) plane",
             f"certified handoff at W ≈ {hw['switch_at_mean']:.0f} ± {hw['switch_at_std']:.0f} N"),
            (axes[1], "compound", hc, C.RAMPS["compound"],
             f"FR-leg torque fraction θ   (W = {C.RAMPS['compound'].load[0]:.0f} N carried)",
             "Compound: (θ × disturbance) plane",
             f"certified handoff at θ ≈ {hc['switch_at_mean']:.2f} ± {hc['switch_at_std']:.2f}")):
        p = g["panels"][panel]
        xs = p["values"]
        F = np.array(p["fail"][C.GRID[panel]["policies"][0]])   # the STAND policy's failure fraction
        cf = ax.contourf(xs, pn, F, levels=np.linspace(0, 1, 11), cmap="RdYlGn_r")
        ax.contour(xs, pn, F, levels=[0.2], colors="k", linewidths=2.6)
        if (F > 0.8).any():
            ax.contourf(xs, pn, (F > 0.8).astype(float), levels=[0.5, 1.5], colors="none", hatches=["////"])
        amb, gust = rp.ambient * FORCE_MAX, rp.gust * FORCE_MAX
        ax.axhline(amb, color="#1a5276", lw=2.0)
        for g0 in rp.gusts:
            x = rp.odd_at(g0)
            ax.plot([x, x], [amb, gust], color="#1a5276", lw=1.6)
        ax.plot(h["switch_at_mean"], amb, marker="*", ms=20, color="#f1c40f", mec="k", mew=0.8)
        ax.set_title(f"{title}\n{sub}", fontsize=15)
        ax.set_xlabel(xlab)
        if p["axis"] == "theta":
            ax.invert_xaxis()
    axes[0].set_ylabel("disturbance push (N)")
    rw = C.RAMPS["weight"]
    handles = [Line2D([], [], color="k", lw=2.6, label="STAND certifiable boundary (20% failure)"),
               Line2D([], [], color="#1a5276", lw=2.0, label=f"demo ODD path ({rw.ambient * FORCE_MAX:.0f} N ambient, "
                                                             f"gusts to {rw.gust * FORCE_MAX:.0f} N)"),
               Line2D([], [], marker="*", ms=15, color="#f1c40f", mec="k", ls="",
                      label=f"certified handoff ({SYS} trigger)"),
               Patch(facecolor="none", hatch="////", edgecolor="k", label="STAND fails > 80%")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=14, frameon=True, bbox_to_anchor=(0.44, -0.10))
    cb = fig.colorbar(cf, ax=axes, fraction=0.035, pad=0.02)
    cb.set_label("STAND-policy failure fraction (tip ∨ slam)")
    fig.suptitle("Certifiable-region geometry: STAND is an island, REST covers the plane —\n"
                 f"the {SYS} handoff fires where the demo path exits the island", fontsize=16, y=1.04)
    _save(fig, "certifiable_regions.png")


def certificate_contraction():
    curves = []
    for key, readout, name, col, xfn in (
            ("weight", "stand", "Weight (raised-CoM load)", "#1f4e79", lambda W: W / W_FULL),
            ("leg", "leg_stand", "Unloaded leg (negative control)", "#7f8c8d", lambda t: 1.0 - t),
            ("compound", "compound_stand", "Compound (leg death while loaded)", "#c0392b", lambda t: 1.0 - t)):
        r = _load("certificates", f"value_{key}.json")
        x = np.array([xfn(float(v)) for v in r["values"]])
        v, s = np.array(r["readouts"][readout]["mean"]), np.array(r["readouts"][readout]["std"])
        o = np.argsort(x)
        curves.append((name, col, x[o], v[o], s[o]))
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
        dsc = (v[0] - v) / max(float(np.mean(s)), 1e-6)
        aR.plot(x, dsc, color=col, lw=2.4, marker="o", ms=4, label=f"{name} — final {dsc[-1]:.2f}σ")
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
    _save(fig, "certificate_contraction.png")


def certification_deficit():
    d = _load("certificates", "ramp_weight-wide.json")
    cfg, arms = d["config"], d["arms"]
    W = np.array(d["odd_trace"])
    t = np.arange(len(W)) * d["dt"]
    H = arms["HANDOFF"]
    vs, vu = np.array(H["V_trace"][cfg["readout"]]), np.array(H["V_trace"]["unified"])
    sw = np.array(H["sw_trace"])
    gusts = [(g * d["dt"], (g + cfg["gust_len"]) * d["dt"]) for g in cfg["gusts"]]
    NAME = {"STAND-ONLY": ("STAND-ONLY", "#e74c3c"), "REST-ONLY": ("REST-ONLY", "#1e8449"),
            "HANDOFF": (SYS, "#1a5276"), "UNIFIED": ("Unified-spec baseline", "#8e44ad")}
    fig, (aW, aV, aH, aM) = plt.subplots(4, 1, figsize=(13.5, 12), sharex=True,
                                         gridspec_kw={"height_ratios": [1.1, 1.6, 2.0, 0.45]})

    def shade(ax):
        for g0, g1 in gusts:
            ax.axvspan(g0, g1, color="#f6b26b", alpha=0.35)

    aW.plot(t, W, color="k", lw=2)
    shade(aW)
    aW.set_ylabel("load W (N)")
    aW.set_title("Dynamic ODD: growing load + gusts (orange)", fontsize=15)
    aV.plot(t, vs, color="#1a5276", lw=1.8, label="V$_{stand}$ (family)")
    aV.plot(t, vu, color="#8e44ad", lw=1.8, label="V$_{unified}$ (merged spec)")
    aV.axhline(cfg["eps"], color="r", ls=":", lw=1.6, label="trigger ε")
    if sw.max() > 0.05:
        i0 = int(np.argmax(sw > 0.1))
        i1 = int(np.argmax(sw > 0.9)) or len(sw) - 1
        aV.axvspan(t[i0], t[i1], color="#95a5a6", alpha=0.25, label="switch window (fleet 10–90%)")
    aV.set_ylim(min(vs.min(), vu.min(), cfg["eps"]) - 0.05, max(vs.max(), vu.max()) + 0.05)
    shade(aV)
    aV.set_ylabel("certificate value")
    aV.set_title("The certification deficit: V$_{stand}$ falls, V$_{unified}$ does not — "
                 "only the family signals the switch", fontsize=15)
    aV.legend(fontsize=14, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    hs = []
    for k, (nm, col) in NAME.items():
        h = np.array(arms[k]["h_trace"])
        hs.append(h)
        aH.plot(t, h, color=col, lw=2, label=f"{nm} (tip {arms[k]['tip']:.2f})")
    aH.set_ylim(min(h.min() for h in hs) - 0.01, max(h.max() for h in hs) + 0.01)
    shade(aH)
    aH.set_ylabel("base height (m)")
    aH.legend(fontsize=14, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    aM.imshow(np.clip(sw, 0, 1)[None, :], aspect="auto", cmap="RdBu", vmin=0, vmax=1, extent=[t[0], t[-1], 0, 1])
    aM.set_yticks([])
    aM.set_ylabel("mode", rotation=0, ha="right", va="center")
    aM.text(t[len(t) // 8], 0.5, "STAND", color="w", fontsize=14, fontweight="bold", va="center")
    aM.text(t[int(len(t) * 0.85)], 0.5, "REST", color="w", fontsize=14, fontweight="bold", va="center")
    aM.set_xlabel("time (s)")
    fig.tight_layout()
    _save(fig, "certification_deficit.png")


def compound_timeline():
    d = _load("certificates", "ramp_compound.json")
    disc = _load("certificates", "value_compound.json")["readouts"]["compound_stand"]["discrim"]
    cfg, arms = d["config"], d["arms"]
    theta = np.array(d["odd_trace"])
    t = np.arange(len(theta)) * d["dt"]
    COLS = {"STAND-ONLY": "#e74c3c", "HANDOFF": "#2980b9", "REST-ONLY": "#27ae60"}
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 8.2), sharex=True, gridspec_kw={"height_ratios": [1, 1.3]})
    for a in (a1, a2):
        for g in cfg["gusts"]:
            a.axvspan(g * d["dt"], (g + cfg["gust_len"]) * d["dt"], color="#f39c12", alpha=0.20, zorder=0)
    a1.plot(t, theta, color="#333", lw=2.5, label="θ (FR torque fraction)")
    a1.set_ylabel("θ")
    a1.set_ylim(0, 1.1)
    b = a1.twinx()
    V = np.array(arms["HANDOFF"]["V_trace"][cfg["readout"]])
    b.plot(t, V, color="#1a5276", lw=2.3, label="V_stand (compound)")
    b.axhline(cfg["eps"], color="r", ls=":", lw=2, label=f"trigger ε = {cfg['eps']:+.2f}")
    b.axhline(0, color="#999", lw=0.8)
    b.set_ylabel("value")
    b.set_ylim(min(V.min(), cfg["eps"]) - 0.05, max(V.max(), 0) + 0.05)
    sw = np.array(arms["HANDOFF"]["sw_trace"])
    if (sw > 0.1).any():
        i0, i1 = int(np.argmax(sw > 0.1)), (int(np.argmax(sw > 0.9)) or len(sw) - 1)
        b.axvspan(t[i0], t[i1], color="#2980b9", alpha=0.12, label="switch window (fleet 10–90%)")
    h1, l1 = a1.get_legend_handles_labels()
    h2, l2 = b.get_legend_handles_labels()
    a1.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=10)
    a1.set_title(f"Leg death while carrying {cfg['load'][0]:.0f} N: V_stand contracts as θ falls "
                 f"(discrimination {disc:.2f}σ) and crosses ε → certified handoff", fontsize=12)
    for arm in ("STAND-ONLY", "HANDOFF", "REST-ONLY"):
        a = arms[arm]
        nm = SYS if arm == "HANDOFF" else arm
        a2.plot(t, a["h_trace"], color=COLS[arm], lw=2.6,
                label=f"{nm}: tip {a['tip']:.2f}, slam {a['slam']:.2f}, standing {a['afford_pre']:.2f}")
    a2.set_ylabel("base height (m)")
    a2.set_xlabel("time (s)")
    a2.legend(fontsize=11)
    a2.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, "compound_timeline.png", dpi=130)


# ── walking ───────────────────────────────────────────────────────────────────────────────────────────────

WALK_FIG_METHODS = ("task-only", "rest-only", "one-way", "direct", "odd")


def payload_claims(profile):
    d = _load("payload", "results.json")
    sc = SCENARIOS[f"payload-{profile}"]
    p = sc.params
    t = np.arange(sc.steps) * DT
    Wv = np.array([sc.odd(k)[0] for k in range(sc.steps)])
    n, seed = next(iter(d.values()))["n"], next(iter(d.values()))["seed"]
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2))
    for ax, key, ylab, title in ((axes[0], "S", "survival S(t)", "Safety (death = flip-over)"),
                                 (axes[1], "SUC", "success CDF (fraction of fleet at goal)",
                                  "Permissiveness (goal completions over time)")):
        ax2 = ax.twinx()
        ax2.fill_between(t, Wv, color="#888", alpha=0.22)
        ax2.plot(t, Wv, color="#555", lw=2.5, label="payload W(t)")
        ax2.set_ylim(0, 900)
        ax2.set_yticks([sc.trigger_eps, LOAD_NOMINAL, p["W_hi"]])
        ax2.tick_params(labelsize=11, colors="#555")
        for m in WALK_FIG_METHODS:
            r = d.get(f"{profile}|{m}")
            if r is None:
                continue
            ax.plot(t[:len(r[key])], r[key], color=COL[m], lw=2.4 if m == "odd" else 1.5,
                    label=f"{label(m, 'walk')} ({r[key][-1]:.2f})")
        ax.set_xlim(0, t[-1])
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.25)
        ax.set_xlabel("time (s)")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=15)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=12, loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=2,
                  frameon=False, columnspacing=1.0)
    fig.suptitle(f"Payload-swap walking — {profile} (tall crate {p['W_lo']:.0f}→{p['W_hi']:.0f} N, CoM "
                 f"{p['h_lo']:.2f}→{p['h_hi']:.2f} m; N={n}, seed {seed}, no respawn)", y=1.05, fontsize=15)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.34)
    _save(fig, f"payload_claims_{profile}.png")


def _topdown(z, goal_d, title, name):
    methods = [m for m in WALK_FIG_METHODS if f"{m}_traj" in z]
    fig, axes = plt.subplots(1, len(methods), figsize=(4.4 * len(methods), 4.9), sharex=True, sharey=True)
    for ax, m in zip(axes, methods):
        yaw0, spawn, traj = z[f"{m}_yaw0"], z[f"{m}_spawn"], z[f"{m}_traj"]
        alv, reached, alive = z[f"{m}_alive"], z[f"{m}_reached"], z[f"{m}_alive_end"]
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
        ax.add_patch(plt.Circle((goal_d, 0), 1.0, fill=False, color="#1a5276", lw=2))
        ax.plot(0, 0, "k^", ms=9)
        ax.set_title(f"{label(m, 'walk')}\nreached {int(reached.sum())}/{len(reached)} | "
                     f"alive {int(alive.sum())}/{len(alive)}", fontsize=14)
        ax.set_xlim(-2, goal_d + 2.5)
        ax.set_ylim(-5, 5)
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)
        ax.set_xlabel("progress toward goal (m)")
    axes[0].set_ylabel("lateral (m)")
    fig.suptitle(f"{title} (green = reached, red = died at X, gray = alive short of goal)", y=1.03)
    fig.tight_layout()
    _save(fig, name, dpi=140)


def payload_topdown(profile):
    z = np.load(os.path.join(OUTPUTS, "payload", f"traj_{profile}.npz"))
    _topdown(z, SCENARIOS[f"payload-{profile}"].goal, f"Payload-swap walking, top-down — {profile}",
             f"payload_topdown_{profile}.png")


def leg_topdown():
    z = np.load(os.path.join(OUTPUTS, "leg", "traj.npz"))
    _topdown(z, SCENARIOS["leg-fault"].goal, "Leg-fault walking, top-down", "leg_topdown.png")


# ── standing ──────────────────────────────────────────────────────────────────────────────────────────────

def standing_survival(table):
    d = _load("standing", f"{table}.json")
    profiles = ("square", "sine") if table == "waves" else ("pulse", "period")
    conds = [(c, p) for c in ("benign", "gusty") for p in profiles]
    fig, axes = plt.subplots(1, 4, figsize=(20, 3.4), sharey=True)
    for ax, (cond, prof) in zip(axes, conds):
        steps = SCENARIOS[f"standing-{prof}"].steps
        t = np.arange(steps) * DT
        for m in ("odd", "direct", "one-way", "task-only", "rest-only"):
            S = d[f"{cond}|{prof}|{m}"]["S"]
            ax.plot(t[:len(S)], S, color=COL[m], lw=2 if m in ("odd", "rest-only") else 1.4,
                    label=label(m, "stand") if (cond, prof) == conds[0] else None)
        ax.set_title(f"{prof} / {cond}", fontsize=15)
        ax.set_xlabel("time (s)")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("survival S(t)")
    hh, ll = axes[0].get_legend_handles_labels()
    fig.legend(hh, ll, fontsize=14, loc="upper center", bbox_to_anchor=(0.5, 0.04), ncol=5, frameon=False)
    n, seed = next(iter(d.values()))["n"], next(iter(d.values()))["seed"]
    fig.suptitle(f"Standing automaton ({table}): {SYS} vs baselines (N={n}, seed {seed}, no respawn)", y=1.02,
                 fontsize=15)
    fig.tight_layout()
    _save(fig, f"standing_survival_{table}.png", dpi=140)


FIGURES = {
    "certificates": [certifiable_regions, certificate_contraction, certification_deficit, compound_timeline],
    "payload": [lambda: [f(p) for p in ("pulse", "period", "dip") for f in (payload_claims, payload_topdown)]],
    "leg": [leg_topdown],
    "standing": [lambda: [standing_survival(t) for t in ("waves", "single")]],
}


def make(groups=tuple(FIGURES)):
    """Draw every figure whose inputs exist; report the ones that are missing and which target makes them."""
    missing = []
    for g in groups:
        for fn in FIGURES[g]:
            try:
                fn()
            except FileNotFoundError as e:
                missing.append((g, str(e)))
    for g, f in missing:
        print(f"[figures] skipped: {f} missing — run: bash scripts/reproduce.sh {g}")
    return not missing
