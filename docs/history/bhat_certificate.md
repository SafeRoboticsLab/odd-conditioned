> **Historical (2026-07-21).** Written when the project's contribution was framed as the B̂ detection certificate. The project later moved to specification-level conditioning (docs/FINDINGS.md); B̂ remains the intended trigger/detection layer. Numbers here predate the safety_sb3 0.4.0 soundness fix.

# B̂ as a Certificate: Deductive OOD Detection for the ODD-Conditioned Safety Filter

*Result consolidation, 2026-07-21. Testbed: Go2 quadruped carrying a hidden sloshy/rigid payload; ODD θ = (rigidity ∈ [0,300], total mass ∈ [1.2,7.5] kg).*

## Contribution (reframed)

The defensible contribution of this project is **detection, not rescue**: a deductively-certified belief **set** B̂ over the unobserved ODD θ that **knows when the operating condition has left the certified envelope** — and says so, rather than silently proceeding on a wrong estimate.

- Inside the ODD box, B̂ contracts toward the true θ and **always contains it** (soundness).
- Outside the box (a payload no admissible θ can explain), B̂ becomes the **empty set** — an unambiguous OOD flag.

This is the *honesty layer* for a least-conservative-per-ODD guarantee: the per-θ guarantee is only valid inside the model class, and B̂'s empty-set event is exactly the signal that it is void. A learned point/encoder estimate cannot provide this — it snaps to the nearest wrong θ with no falsification semantics (statistical, not deductive), i.e. it **fails silently** in precisely the regime a safety filter exists for.

## Method (set-membership B̂)

Maintain the **set** of θ consistent with the observed transitions under the known model class plus a bounded-noise tolerance ε — not a point estimate. Concretely (multiple-model, trajectory-matching):

- A grid of candidate θ (15×15 = 225 over rigidity × mass, spanning the training box + a margin), each replicated R=32× in one batched sim.
- All candidates are forced to an identical canonical stand and driven by an identical scripted **excitation** (a fast ±y square-wave base shake, policy-independent). The hidden payload is felt only through its effect on the base.
- Observable y = base attitude (projected gravity), height, linear/angular velocity, and lateral displacement. Denoise GPU contact nondeterminism by averaging over the R replicas of each θ.
- **Consistency:** candidate i ∈ B̂_t iff its cumulative-to-t residual `max_{s≤t} ‖ȳ_pred_s(i) − ȳ_obs_s‖_W ≤ ε`, with W the per-channel ensemble spread and ε = 0.22 (ensemble-std units).
- **δ\*** = detection latency = first t at which |B̂| drops below 10% of the grid (in-dist) or reaches 0 (OOD).

## Results (E018)

![belief-set size vs evidence, and surviving set on the θ-grid](../results/E018/setmembership.png)

| true θ* | |B̂| trajectory | δ\* | outcome |
|---|---|---|---|
| **in-dist mid** (k=183, m=4.33) | 225 → 4 | **0.14 s** | shrinks to a tight set around θ*; **truth ∈ B̂** (residual 0.000 ≤ ε) |
| **in-dist light-rigid** (k=274, m=2.24) | 225 → 9 | **0.10 s** | shrinks; **truth ∈ B̂**; centroid (272, 2.59) ≈ θ* |
| **OOD heavy 12 kg** (k=0, m=12) | 225 → **0** | **0.14 s** | **empty set** — no θ explains the data (closest residual 0.716 ≫ ε=0.22) → **OOD DETECTED** |

Both properties hold: in-distribution the set **shrinks and contains the truth** (sound); out-of-distribution it goes **empty** — the silent-failure alarm — within ~0.14 s.

## Why detection and not rescue (the negative result that motivated the pivot)

We tested the natural next step — "B̂ detects OOD → hand off to a trained soft-descent fallback" (E035) — and it **failed on two counts**, which is why the contribution is scoped to detection:

1. **The standing policy does not need rescuing in-envelope-adjacent.** The blind reactive policy stands robustly at 12 kg + a 20 N pull (0% topple, 0 N non-foot contact) — it is a very strong reactive baseline well outside the 7.5 kg training box.
2. **The fallback is fragile out-of-envelope.** Handing a 12 kg top-heavy load to a descent policy trained only on ≤7.5 kg makes it topple 41% and slam at ~2500 N.

So a naive "θ-OOD → descend" filter is *net harmful*: at mild OOD standing is fine and descending throws the robot; at the extreme where standing would fail, the fallback is not robust either. The lesson: **B̂'s role is the alarm/certificate ("the guarantee is void — proceed with caution"), and the correct action under the alarm is a separate, harder problem** (the right action at 12 kg was "keep standing," not "descend"). The *detection* is sound and self-contained; the *action* is not claimed.

## Contrast: B̂ vs learned estimators

| | deductive set-membership B̂ | learned point / RMA-style encoder |
|---|---|---|
| output | a **set** of consistent θ (or ∅) | a single θ̂ (or latent) |
| OOD behavior | **empty set = explicit flag** | snaps to nearest wrong θ, **silent** |
| soundness | truth never excluded (w.r.t. model class) | statistical/conformal only |
| under distribution shift | holds by construction | breaks — the exact failure a filter must avoid |

The performance layer (a history/introspective policy) *proposes*; the deductive B̂ *disposes/certifies*. They are the same object at two compressions, but only B̂ carries the falsification semantics a guarantee needs.

## Caveats / scope

- **Sound *w.r.t. the model class.*** The POC uses model-vs-model consistency (the in-dist truth is a grid node, so its residual is 0 by construction). Real deployment relies on the **empty-set flag firing for genuine model-class violations** — demonstrated here for mass beyond the box; the analogous test for a *structural* violation (shifting CoM, asymmetric, liquid) is the natural strengthening.
- **Requires active excitation.** The ±y probe is what makes the payload observable; a passive/quiet robot never identifies θ (excitation-graded δ\*).
- **δ\* is a latency, not zero.** ~0.10–0.14 s here; any downstream use must budget for it.

## Fit to the theory & next steps

This grounds the info-constrained value V^info(x, B): B̂ is the belief B, the empty-set event is the boundary of the model class, and δ\* is the identification latency in the guarantee-horizon `w = w_phys − δ\*`. Natural next steps for the *detection* story (not rescue): (1) the model-class-violation OOD test (structural, not just mass); (2) a head-to-head with a learned θ-encoder showing its silent-wrong failure on the same OOD input; (3) tightening ε and the excitation to characterize the sound-detection frontier.
