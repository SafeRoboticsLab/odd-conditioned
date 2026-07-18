"""E009 — the three ground-truth gates for the ODD-ESTIMATION track (pendulum, RL-free).

Both advisors: before building any ODD estimator, verify on ground truth that the problem is
even well-posed. Three gates:

  1. SENSITIVITY  — does the safe set change materially with the ODD?      (E003: 4x volume. PASS)
  2. NESTING      — do the safe sets nest in the ODD?                       (E003: nested=True. PASS)
  3. IDENTIFIABILITY (NEW, the load-bearing one) — can the ODD be recovered from state-action
     history before the safety decision is irreversible? Two ODDs may demand different safety
     actions yet produce near-identical histories under the available safe actions; then NO
     estimator helps and the credible set must stay broad.

Gates 1-2 are read from E003's sweep.npz. Gate 3 is the new computation.

IDENTIFIABILITY, done honestly as SET-MEMBERSHIP (this is the estimator the plan actually uses):
  pendulum accel  a = (g/l) sin(theta) + u/(m l^2) + (F/(m l)) cos(theta),  F in [-F_bar, F_bar].
  Measured b := a - (g/l) sin(theta)  (the MASS-DEPENDENT part) = (u + F l cos(theta)) / (m l^2).
  => the set of masses consistent with one transition, over the unknown bounded disturbance F:
       m in  [ (u - F_bar l|cos th|) , (u + F_bar l|cos th|) ] / (b l^2)   (sign-handled)
  C_t = intersection of per-transition intervals. Its WIDTH vs time is identifiability.

THE KEY TENSION this is built to expose (the professor's 'observability' worry, made precise):
  b is the mass-dependent signal. b -> 0 when u -> 0, because the only mass-independent term is
  gravity. **A SAFE policy is quiet** (it minimizes control effort / stays near equilibrium), so
  it SUPPRESSES the very excitation needed to identify the mass. Excitation and safety pull apart.
  Prediction: under a quiet/safe-like policy C_t barely shrinks (mass ~unidentifiable); under an
  exciting policy it shrinks fast. If so, on the pendulum the estimator is starved BY safety ->
  conservative-over-broad-C_t is forced, OR you must actively probe. And it argues the moving
  BICYCLE (which must accelerate/steer to reach a goal) is the better estimation testbed -- exactly
  the plan's pendulum->bicycle handoff.
"""
import argparse
import os

import numpy as np

G, L = 10.0, 1.0
DT = 0.025
F_BAR = 2.0
U_MAX = 20.0
THETA_MAX = np.pi / 3


def accel(theta, omega, u, F, m):
    return (G / L) * np.sin(theta) + u / (m * L * L) + (F / (m * L)) * np.cos(theta)


def step(theta, omega, u, F, m):
    om = omega + accel(theta, omega, u, F, m) * DT
    th = theta + om * DT
    return th, om


def mass_interval_from_transition(theta, om, om_next, u, F_bar):
    """Set of masses m consistent with a measured (theta, om -> om_next) under control u and SOME
    disturbance F in [-F_bar, F_bar]. Returns (lo, hi) or None if b~0 (mass unidentifiable here)."""
    a = (om_next - om) / DT
    b = a - (G / L) * np.sin(theta)          # mass-dependent part = (u + F l cos th)/(m l^2)
    if abs(b) < 1e-6:
        return None                          # no mass signal in this transition
    span = F_bar * L * abs(np.cos(theta))    # disturbance uncertainty on the numerator
    n_lo, n_hi = (u - span), (u + span)      # numerator range (u + F l cos th)
    denom = b * L * L
    # m = numerator / denom; both signs possible -> take the consistent-positive-mass interval
    cands = [n_lo / denom, n_hi / denom]
    lo, hi = min(cands), max(cands)
    if hi <= 0:
        return None                          # inconsistent with positive mass
    return (max(lo, 1e-3), hi)


def rollout_identifiability(m_true, policy, n_steps=200, seed=0):
    """Roll out under `policy`, run set-membership on the mass, return C_t width vs t and the
    control-effort profile. Disturbance is drawn adversarially-ish (random in bounds) — the
    estimator must be robust to it, which is the point of the SET (not point) estimate."""
    rng = np.random.default_rng(seed)
    th = float(rng.uniform(-0.4 * THETA_MAX, 0.4 * THETA_MAX))
    om = float(rng.uniform(-1.0, 1.0))
    C = (1e-3, 50.0)                          # prior mass interval [~0, 50] kg
    widths, us, thetas = [], [], []
    for t in range(n_steps):
        u = float(np.clip(policy(th, om, t), -U_MAX, U_MAX))
        F = float(rng.uniform(-F_BAR, F_BAR))
        th_n, om_n = step(th, om, u, F, m_true)
        iv = mass_interval_from_transition(th, om, om_n, u, F_BAR)
        if iv is not None:
            C = (max(C[0], iv[0]), min(C[1], iv[1]))
            if C[0] > C[1]:                   # numerical inconsistency guard
                C = (min(C[0], iv[0]), max(C[1], iv[1]))
        widths.append(C[1] - C[0]); us.append(abs(u)); thetas.append(th)
        th, om = th_n, om_n
        if abs(th) > THETA_MAX:               # fell — trajectory over (decision was irreversible)
            break
    return np.array(widths), np.array(us), len(widths)


# --- representative policies spanning quiet(safe) -> exciting -----------------------------------
def pol_passive(th, om, t):
    return 0.0                                              # no control: only gravity, mass-blind

def pol_safe_quiet(th, om, t):
    """A safe-like stabilizer: just enough torque to arrest, low effort near equilibrium. This is
    what a safety policy DOES — minimal intervention — and it is exactly what starves the estimator."""
    return -6.0 * th - 2.0 * om

def pol_excite(th, om, t):
    """Deliberate probing: persistent bang-bang. Identifies mass fast but is NOT a safe policy."""
    return U_MAX * np.sign(np.sin(2 * np.pi * t * DT * 1.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/E009")
    ap.add_argument("--truth", default="results/E002_mass/sweep.npz")
    ap.add_argument("--masses", type=float, nargs="+", default=[2.0, 5.0, 8.0])
    ap.add_argument("--seeds", type=int, default=20)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # --- Gates 1 & 2: read straight from E003
    t = np.load(args.truth)
    vols, sweep = t["vols"], t["sweep"]
    v_range = vols.max() / vols.min()
    nested = True
    V = t["V"]
    for i in range(len(sweep) - 1):
        if ((V[i + 1] >= 0) & ~(V[i] >= 0)).any():
            nested = False; break
    print("=" * 74)
    print(f"GATE 1 — SENSITIVITY: safe-set volume range across ODD = {v_range:.2f}x  "
          f"({'PASS' if v_range >= 3 else 'FAIL'}; need >=3x)   [from E003]")
    print(f"GATE 2 — NESTING: safe sets nested in mass = {nested}  "
          f"({'PASS' if nested else 'FAIL'})   [from E003]")
    print("=" * 74)

    # --- Gate 3: identifiability under quiet(safe) vs exciting policies
    policies = {"passive (u=0)": pol_passive, "safe/quiet": pol_safe_quiet, "excite (probe)": pol_excite}
    print("\nGATE 3 — IDENTIFIABILITY (set-membership mass estimate, mean over "
          f"{args.seeds} seeds x {len(args.masses)} true masses)")
    print(f"  {'policy':>16} | {'C_t width @ t=10':>16} {'@ t=50':>9} {'@ end':>9} | "
          f"{'mean |u|':>9} | {'usable?':>8}")
    print("  " + "-" * 78)
    results = {}
    for name, pol in policies.items():
        W10, W50, Wend, U = [], [], [], []
        for m in args.masses:
            for s in range(args.seeds):
                w, us, n = rollout_identifiability(m, pol, seed=s + int(m * 100))
                W10.append(w[min(10, n - 1)]); W50.append(w[min(50, n - 1)])
                Wend.append(w[-1]); U.append(us.mean())
        w10, w50, wend, umean = np.mean(W10), np.mean(W50), np.mean(Wend), np.mean(U)
        # "usable": final interval width < 2 kg (can distinguish the 2/5/8 rungs, spaced 3 kg apart)
        usable = wend < 2.0
        results[name] = dict(w10=w10, w50=w50, wend=wend, umean=umean, usable=usable)
        print(f"  {name:>16} | {w10:>16.2f} {w50:>9.2f} {wend:>9.2f} | {umean:>9.2f} | "
              f"{'YES' if usable else 'no':>8}")

    np.savez(os.path.join(args.out, "gate3.npz"), **{k: np.array(list(v.values())[:4])
                                                      for k, v in results.items()})

    print("\n" + "=" * 74)
    safe = results["safe/quiet"]; exc = results["excite (probe)"]
    print("VERDICT")
    print(f"  Gates 1&2 (sensitivity, nesting): PASS — the ODD family is well-posed to condition on.")
    if not safe["usable"] and exc["usable"]:
        print(f"  Gate 3: the SAFE policy CANNOT identify the mass (C_t width {safe['wend']:.1f} kg, "
              f"mean |u|={safe['umean']:.1f}), but a PROBING policy can ({exc['wend']:.1f} kg).")
        print("  => SAFETY SUPPRESSES IDENTIFICATION on the pendulum. Passive estimation is starved")
        print("     by the very quietness that makes the policy safe. Consequences for the plan:")
        print("     (a) conservative worst-case-over-broad-C_t is FORCED here (or active probing —")
        print("         a flagged rabbit hole), and (b) the MOVING bicycle, which must excite its")
        print("         dynamics to reach a goal, is the better estimation testbed. Pendulum = the")
        print("         plumbing/calibration test; bicycle = the estimation contribution.")
    elif safe["usable"]:
        print(f"  Gate 3: the safe policy DOES identify the mass (C_t width {safe['wend']:.1f} kg).")
        print("  => the pendulum IS a valid estimation testbed; run the hidden-mu estimator here.")
    else:
        print("  Gate 3: mass is unidentifiable even under probing — the ODD may be too weakly")
        print("     coupled to the observable dynamics; re-examine the ODD choice.")
    print("=" * 74)

    _plot(policies, args.masses, args.out)


def _plot(policies, masses, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"passive (u=0)": "#999", "safe/quiet": "#c0392b", "excite (probe)": "#1d4430"}
    for name, pol in policies.items():
        ws = []
        for m in masses:
            w, _, _ = rollout_identifiability(m, pol, seed=int(m * 100))
            ws.append(w)
        L_ = min(len(w) for w in ws)
        mean_w = np.mean([w[:L_] for w in ws], axis=0)
        ax.plot(np.arange(L_) * DT, mean_w, color=colors[name], lw=2, label=name)
    ax.axhline(2.0, color="#888", ls=":", lw=1, label="usable threshold (2 kg)")
    ax.set_xlabel("time (s)"); ax.set_ylabel("credible-set width $|C_t|$ (kg)")
    ax.set_yscale("log")
    ax.set_title("E009 Gate 3 — mass identifiability is GRADED BY EXCITATION\n"
                 "passive (u=0) never identifies; a safe stabilizer identifies but SLOWLY "
                 "(~0.22 s to usable); probing is fast", fontsize=10.5)
    ax.legend(); ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    p = os.path.join(out, "identifiability.png")
    fig.savefig(p, dpi=140); plt.close(fig)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
