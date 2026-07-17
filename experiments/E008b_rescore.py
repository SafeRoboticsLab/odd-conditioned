"""E008b — re-score E008's checkpoints. NO TRAINING.

E008 reported the conditioned critic over-claiming 1.7-5.0x vs E003 ground truth. The professor
(2026-07-17) showed that verdict scored the wrong object several ways. This isolates how much of
the "over-claim" is real boundary error vs. scoring artifact, using the checkpoints already on
disk (`results/E008/{conditioned,blind}.zip`).

TWO CORRECTIONS, both legitimate (the grid ground truth gets both for free):

1. **g-composition.** The grid V satisfies V <= g pointwise (the backup mins with g every
   iteration). The raw learned Q has no such constraint. The DEPLOYED filter is
   V_filter = min(g, V_hat) -- g is analytic and known at runtime. Scoring raw V_hat grades the
   network on a quantity no filter ever uses. Report BOTH raw and composed, so this is not
   grading on a curve.

2. **Region decomposition of the optimism.** Split the false-positive (learned-safe-but-truly-
   unsafe) mass into three regions, because they mean completely different things:
     - FAILURE BAND   |theta| in [THETA_MAX, grid_edge]: g < 0 by definition. Any claim here is
       pure artifact -- min(g,.) deletes it at zero cost.
     - OUT-OF-SUPPORT beyond the spawn box (|theta|>0.9*THETA_MAX or |omega|>0.5*OMEGA_LIM):
       never supervised; pure extrapolation. Expected to be junk; not evidence about the method.
     - IN-SUPPORT BOUNDARY the rest: THIS is the only optimism that means "RL misplaced the
       boundary where it had data". This is the number the gate actually cares about.

Prediction to test (professor): composition + support-restriction mostly fixes m=2, leaves an
m=8 residual that E009's V* will later explain (the m=8 residual is a preview of V - V*, because
E008 trained with mid-episode mass resampling => its true fixed point is a jump-ODD value, not
the static m=8 slice).
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from odd_conditioned.envs.pendulum_odd import THETA_MAX, OMEGA_LIM  # noqa: E402
from experiments.E008_conditioned_critic import learned_safe_set     # reuse the exact scorer

# spawn support (pendulum_odd.py:218-219) -- where the critic was actually supervised
SUP_TH = 0.9 * THETA_MAX
SUP_OM = 0.5 * OMEGA_LIM


def raw_value(model, th_grid, om_grid, odd_feat, obs_mode, device="cpu"):
    """V_hat(x) WITHOUT the g-composition -- the raw critic. Mirrors learned_safe_set's action
    construction (concat ctrl+dstb actors) but returns the raw Q, not min(g,.)."""
    TH, OM = np.meshgrid(th_grid, om_grid, indexing="ij")
    core = np.stack([np.sin(TH).ravel(), np.cos(TH).ravel(), OM.ravel()], axis=1)
    if obs_mode == "oracle":
        feat = np.tile(np.asarray(odd_feat, np.float32), (core.shape[0], 1))
        obs = np.concatenate([core, feat], axis=1).astype(np.float32)
    else:
        obs = core.astype(np.float32)
    with torch.no_grad():
        t = torch.as_tensor(obs, device=device)
        a = torch.cat([model.policy.actor(t, deterministic=True),
                       model.policy.dstb_actor(t, deterministic=True)], dim=1)
        q = model.critic(t, a)
        q = torch.min(*q) if isinstance(q, (list, tuple)) else q
    return q.cpu().numpy().reshape(TH.shape)


def g_grid(th_grid, om_grid):
    TH, OM = np.meshgrid(th_grid, om_grid, indexing="ij")
    return np.minimum((THETA_MAX - np.abs(TH)) / THETA_MAX, (OMEGA_LIM - np.abs(OM)) / OMEGA_LIM)


def regions(th_grid, om_grid):
    TH, OM = np.meshgrid(th_grid, om_grid, indexing="ij")
    failure = np.abs(TH) > THETA_MAX                                  # g < 0 by definition
    out_sup = (~failure) & ((np.abs(TH) > SUP_TH) | (np.abs(OM) > SUP_OM))
    in_sup = ~(failure | out_sup)
    return failure, out_sup, in_sup


def decompose(V_hat, V_true, g, cell, fail, outsup, insup):
    """Optimism = learned-safe AND truly-unsafe, split by region; plus the composed set."""
    lm_raw = V_hat >= 0
    lm_cmp = np.minimum(g, V_hat) >= 0          # deployed filter set
    tm = V_true >= 0
    opt_raw = lm_raw & ~tm                       # false positives, raw critic
    return dict(
        vol_true=float(tm.sum() * cell),
        vol_raw=float(lm_raw.sum() * cell),
        vol_composed=float(lm_cmp.sum() * cell),
        optimism_raw=float(opt_raw.sum() * cell),
        optimism_in_failure_band=float((opt_raw & fail).sum() * cell),
        optimism_out_of_support=float((opt_raw & outsup).sum() * cell),
        optimism_in_support_boundary=float((opt_raw & insup).sum() * cell),  # THE number that matters
        optimism_composed=float((lm_cmp & ~tm).sum() * cell),
        conservatism_composed=float((~lm_cmp & tm).sum() * cell),
        iou_raw=float((lm_raw & tm).sum() / max((lm_raw | tm).sum(), 1)),
        iou_composed=float((lm_cmp & tm).sum() / max((lm_cmp | tm).sum(), 1)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/E008")
    ap.add_argument("--truth", default="results/E002_mass/sweep.npz")
    ap.add_argument("--rungs", type=float, nargs="+", default=[2.0, 3.5, 5.0, 6.5, 8.0])
    ap.add_argument("--out", default="results/E008b")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from safety_sb3 import IsaacsSAC
    truth = np.load(args.truth)
    th, om, sweep, Vtrue = truth["theta"], truth["omega"], truth["sweep"], truth["V"]
    cell = (th[1] - th[0]) * (om[1] - om[0])
    g = g_grid(th, om)
    fail, outsup, insup = regions(th, om)
    print(f"grid {len(th)}x{len(om)}  cell {cell:.3e}")
    print(f"region areas: failure-band={fail.sum()*cell:.2f}  out-of-support={outsup.sum()*cell:.2f}  "
          f"in-support={insup.sum()*cell:.2f}\n")

    def truth_at(m):
        return Vtrue[int(np.argmin(np.abs(sweep - m)))]

    out = {}
    for arm in ("conditioned", "blind"):
        p = os.path.join(args.dir, f"{arm}.zip")
        if not os.path.exists(p):
            print(f"skip {arm}: no checkpoint"); continue
        model = IsaacsSAC.load(p, device="cpu")   # score on CPU; checkpoint may be CUDA-saved
        obs_mode = "blind" if arm == "blind" else "oracle"
        print(f"=== {arm} ===")
        print(f"{'rung':>5} | {'true':>6} {'raw':>6} {'comp':>6} | {'IoU raw':>7} {'IoU cmp':>7} | "
              f"{'opt:fail':>8} {'opt:oos':>7} {'opt:BND':>7} | {'opt cmp':>7} {'cons cmp':>8}")
        print("-" * 104)
        for m in args.rungs:
            V_hat = raw_value(model, th, om, [m], obs_mode)
            d = decompose(V_hat, truth_at(m), g, cell, fail, outsup, insup)
            out[f"{arm}_m{m:g}"] = d
            print(f"{m:>5} | {d['vol_true']:>6.2f} {d['vol_raw']:>6.2f} {d['vol_composed']:>6.2f} | "
                  f"{d['iou_raw']:>7.2f} {d['iou_composed']:>7.2f} | "
                  f"{d['optimism_in_failure_band']:>8.2f} {d['optimism_out_of_support']:>7.2f} "
                  f"{d['optimism_in_support_boundary']:>7.2f} | "
                  f"{d['optimism_composed']:>7.2f} {d['conservatism_composed']:>8.2f}")
        print()

    json.dump({"results": out, "rungs": args.rungs}, open(os.path.join(args.out, "rescore.json"), "w"), indent=2)

    # the verdict hinges on ONE column: composed optimism inside the spawn support.
    print("=" * 104)
    print("VERDICT — the only optimism that means 'RL misplaced the boundary where it had data'")
    print("is `opt cmp` (composed, and note composition removes the failure band entirely):")
    for m in args.rungs:
        c = out.get(f"conditioned_m{m:g}")
        if c:
            raw_ratio = c["vol_raw"] / c["vol_true"]
            cmp_ratio = c["vol_composed"] / c["vol_true"]
            print(f"  m={m:g}: raw ratio {raw_ratio:.2f}x  ->  composed ratio {cmp_ratio:.2f}x   "
                  f"(in-support boundary optimism = {c['optimism_in_support_boundary']:.2f})")
    print(f"\nwrote {args.out}/rescore.json")


if __name__ == "__main__":
    main()
