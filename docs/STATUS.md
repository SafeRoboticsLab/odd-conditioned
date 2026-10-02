# Status — where the project stands and what to pick up

*Snapshot 2026-10-02. Experimental work stopped on 2026-08-24; the repository was cleaned up for
collaboration on 2026-10-02. Keep this page current: it is the entry point for anyone picking the work up.*

## Where it stands

- **Thesis settled** (2026-08-22): specification-level ODD conditioning with a family of certified modes and
  certified transitions ([FINDINGS.md](FINDINGS.md)). Policy-level θ-conditioning was tested at length and
  rejected (E043, E055, E060 → E064, E066–E068).
- **All paper experiments are done**, including 4 evaluation repetitions of the headline tables (E094).
- **Paper draft v3** exists (with Buzi, not in the repo; written 2026-08-24, figures signed off). It
  is a results package with figure walkthroughs, not yet a manuscript: no venue chosen, no related-work or
  method sections in paper form. An older prose draft (v2, built on E069–E079) also exists outside the repo.
- **Reproducibility**: everything in the draft regenerates from this repo + the weights bundle
  ([REPRODUCE.md](REPRODUCE.md)); verified from a fresh clone on 2026-10-02 (smoke tests, E091, E092).

## What to pick up next (in rough priority order)

1. **Turn the draft into a manuscript.** Pick a venue; write related work around the positioning in
   [FINDINGS.md §8](FINDINGS.md#8-positioning-from-the-literature-triangulation-2026-07) (cite Gandhi & Mhaskar
   2008 first); convert the figure walkthroughs into captions; state the caveats of FINDINGS §7 up front.
2. **Train more seeds.** Every policy is single-seed. At minimum retrain the four automaton policies (stand_hi
   recal, rest_hi, getup_v2, descend_v4) with 2 more seeds and rerun E092 + standing; recipes in
   [TRAINING.md](TRAINING.md). Runtime thresholds will need recalibration per seed (E089/E091 `--cal`).
3. **Close the loop on the belief.** The walking evaluations read the true W / θ for the return gate and the
   E092 descent trigger. Replace it with an estimator — the set-membership B̂ of E018 for the payload, the
   torque-saturation residual of E091 for actuator faults — and measure what detection latency costs.
4. **Phase-aware braking** (training-free). ~45 % of E092 deaths are in BRAKE, and they are gait-phase
   structured (mid-swing handoffs die). Deferring the walk→stand handoff to the next stance phase should cut
   them; it would likely lift the period/dip numbers well above 0.4.
5. **A walk→rest funnel trained from gait entries**, so the descent no longer needs the brake-to-stance detour.
6. **Compound scenario at full protocol.** E078 is demonstration-level (one run, N=128) and its slam rate stays
   high (0.53); give it the E092-style walking/standing evaluation with repetitions.
7. **Optional cleanups**: regenerate or drop draft figures F1/F4/F5/F6/F8 (no producing script); decide whether
   the old payload line (E014–E038) and bicycle line (E001–E013) appear in the paper at all.

## Open questions

- Is the pulse (maneuver-latency) boundary a limitation to state, or a second result? A forecasting variant
  (trigger on a predicted ODD change) would turn it into one.
- Does the certificate contraction story need a theorem-level statement, or is the F7 "contracts iff
  non-absorbable" evidence plus the negative control enough for the target venue?
- Walking on the leg-fault scenario has only one run (E091); the success gap over the walker (0.28 vs 0.20) is
  modest — rerun with repetitions before leaning on it.

## Where things live

| what | where |
|---|---|
| code | this repo; the project's env/task code is in the sandbox submodule on branch `project/odd-conditioned-go2-payload` |
| checkpoints | `odd-conditioned-weights-v1.tar.gz` from Buzi → `results/` via `scripts/fetch_bundles.sh --from <dir>` |
| paper draft, per-experiment reports, videos | with Buzi (not shared); the docs here are the current summary |
| training logs / wandb | wandb project `odd-conditioned` (entity `buzinguyen`); run ids per policy in [TRAINING.md](TRAINING.md) |
| experiment-by-experiment history | [EXPERIMENTS.md](EXPERIMENTS.md); the full private lab notebook stays with Buzi |

## Code debt to know about

- Experiment scripts are research scripts: constants at module top, `main` blocks that run full tables, and
  helper reuse by import (E089/E091/E092 import E084; E086 patches E084's globals on import, so run it in its
  own process — `E094_seeds.py` already does).
- Scripts before E039 ran on older library versions (safety_sb3 ≤ 0.3, the SAC learners) and their
  checkpoints are not distributed; treat them as a record, not as runnable code.
- `E084b_early.py` still points at getup v1, matching when it ran.
- The sandbox `results/` directory inside the submodule is not git-ignored there; don't commit from it blindly.
