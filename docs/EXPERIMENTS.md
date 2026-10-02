# Experiment index — E001–E094

One row per experiment ID, distilled from the private lab notebook (registry, daily logs, tickets). Scripts are
in `experiments/` ("—" = no dedicated script: a training run launched from a sandbox config, or never run).
Status: **done**, **superseded** (a later run or fix replaced its numbers or conclusion), **abandoned** (killed
or never run), **negative** (ran cleanly, hypothesis failed). "?" = not recorded.

Runnable today: E039 onward (safety_sb3 / sandbox 0.4.0). Scripts before that ran on older library versions
and checkpoints that are not distributed — read them as a record. The paper rests on E069–E094.

## Superseding events (read before citing any number)

| when | event | what it invalidates |
|---|---|---|
| 07-16 | E002 too slow, killed | E002 → E004 (same sweep, fast implementation) |
| 07-17 | E006 empty safe set = `l_neg` misuse under RA-SAC (E007) | E006 → E007/E008 |
| 07-17 | E008b: E008 scored the wrong object (jump-ODD fixed point) | E008 → E008c |
| 07-18 | E011 v6.2: friction never limited turning (bug) | all E011 v1–v6 results incl. the "7% mode-flip PASS" |
| 07-20 | E015b/E016b: payload was observable (θ redundant with slosh state) | E015/E016 → E015b/E016b |
| 07-21 | **E020 metric fix**: env auto-reset masked falls; use falls/env-sec from `dones & ~timeouts` | **every safe_rate number in E015–E019** (incl. E017-OOD, E019 headline) |
| 07-21 | E022 n=2 video misleading | E022 → E024 |
| 07-21 | E025/E026: no resonance tongue | E023's "parametric resonance" reading |
| 07-25 | E036: `final_model.zip` loaded by E021–E028 was a 0.42 in-dist trough | caveat on all history-arm dynamic-ODD numbers (the result survives) |
| 07-31 | E037 corrected break sweep | E019 headline ("raw-θ breaks first; blind most robust") is only half true |
| 08-02 | **safety_sb3/RSS 0.4.0 migration**: entropy bonus removed from the RA-SAC critic target (certificate precision 0.05→0.98); old class paths don't load | all value-based results on ≤v0.3.x critics (E014–E038, e.g. E038's V-as-detector) and every pre-0.4.0 checkpoint |
| 08-11 | SAC in 0.4.0 also bootstraps timeout terminals | E039–E042 SAC collapse is confounded; switched to PPO (ReachAvoidPPO2P) |
| 08-20 | Buzi's reframing + E043: no strategy split under a sound PPO learner | E014 dodge-vs-brace bifurcation likely an artifact of the v0.3 learner |
| 08-21 | **E064 θ-ablation**: runtime θ isn't load-bearing (frozen 0.2 beats oracle) | E060 "conditioning wins because told θ" (= privileged training) and E062/E063 by inheritance |
| 08-22 | E073 physics fix: flat load = massless base force that stabilizes laterally | E070–E072 kept only as "iteration 1" (high-CoM E073–E075 is the result) |
| 08-22 | E075 recal | E074's noisy stand_hi trigger numbers |
| 08-23 | E081 no-respawn survival | E080's with-respawn aggregate metrics |
| 08-23 | E086 diagnostic (guard bugs) + E087 (V_up miscalibrated) | E084's "physical descent floor" verdict was partly artifacts; re-measured in the E086/E084 RERUN |
| 08-23 | E086c timeout fix (20 s episode < 24 s horizon) | earlier E086 single-excursion table |
| **08-23 evening** | **CRITICAL accounting fix**: the env's `illegal_contact` termination was never counted (93/94 such deaths respawned as "alive") | **ALL E080–E086 survival numbers (incl. E081, E084, E084b, E086/E086c) and E089 run-1.** Only the "CORRECTED FINAL TABLES" (08-23) and later are valid; seeded in E094 |
| 08-23 | E089 → **E089v2**: at 9 m goal, WALK-ONLY looked good because finished robots stood still through the excursion | E089 walking-success claims; weight-excursion walking = boundary finding, so the walking demo moved to E091 (leg fault) and E092 (payload swap) |
| 08-24 | trigger ROC recalibration (−0.15, K=10) | E089 operating point (−0.10, K=5 gave 47.7% false sits / 30 s) |

---

## (a) E001–E013 — pendulum / bicycle ground truth and gates (RL-free HJ + small SAC)

| ID | script(s) | question / delta | outcome | status |
|---|---|---|---|---|
| E001 | E001_compounding_conservatism.py | Does the classical CLF handoff test reject recoverable states vs the maximal RA set? (pendulum, torque ODD) | Yes: Ω 1.61–3.81× smaller than RA; 28–50% of recoverable envelope rejected; maximal test also fails 4/4 (25–33% stranded) | done |
| E002 | E002_odd_sweep_groundtruth.py | Ground-truth safe-set family over disturbance bound F̄ | killed at ~40 min, no results | superseded → E004 |
| E003 | E002_odd_sweep_groundtruth.py (mass axis) | Family over pendulum mass 2→8 kg | 4.0× volume contraction, monotone and nested, smooth | done |
| E004 | E002_odd_sweep_groundtruth.py (F̄ axis) | F̄ 0→6 N on the fast implementation | only 1.33× (adversary out-powered); nested | done |
| E005 | E005_sine_odd.py | Oscillating mass ODD: resonance? Value of observing the ODD phase | no resonance (flat); value of observing the phase collapses 84% as the ODD speeds up; crossover at √(g/l) | done (resonance negative) |
| E006 | E006_conditioned_critic.py | One conditioned IsaacsSAC critic vs blind on the mass family | killed at 75k: empty safe set everywhere (`l_neg` misuse); launched without a registry row | abandoned → E007 |
| E007 | E007_diagnose_value_level.py | Diagnose E006's V≡l | constant l under RA-SAC drives V→l; library correct, usage wrong; avoid ≠ RA instance | done |
| E008 | E008_conditioned_critic.py | The gate on v0.2.0 IsaacsSAC (avoid), cond vs blind, mid-episode resampling | ran; raw 1.7–5.0× "over-claim" (row still says "queued") | superseded → E008b/E008c |
| E008b | E008b_rescore.py | Re-score E008 ckpts with min(g,V̂) + region decomposition | over-claim mostly artifact, but wrong object; cond ≈ blind | done |
| E008c | E008c_gate.py, run_E008c_parallel.sh | Gate attempt 2: static ODD, normalized, wide spawns, γ anneal, specialists | **PASS**: conditioned composed IoU 0.87–0.98 tracks the family; blind frozen at one set; beats m=8 specialist (1 seed) | done |
| E009 | E009_identifiability_gate.py | Estimation gates: sensitivity, nesting, identifiability of mass | gates 1–2 pass; ID graded by excitation (passive never, safe stabilizer ~0.22 s, probing 1 step) | done |
| E010 | E010_bicycle_vanilla_dynamicodd.py, run_E010_parallel.sh | Does vanilla ReachAvoidSAC handle a control-authority ODD (Bicycle5D)? | blind reaches ~86% but ~6× the collisions of spec_hi; oracle-append worse than blind; slope metrics confounded (1 seed) | done |
| E011 | E011_friction_grid_gate.py, E011_render_rollouts.py | RL-free friction-circle bicycle gate (≥3× volume, nesting, mode flip) | v1–v6 void (bug); after the fix 1.51× volume (<3×), brake/thread flip 8.3% following √μ | done (v1–v6 superseded) |
| E012 | E012_guarantee_horizon.py | Kill-test: is local-μ knowledge worth anything given a guaranteed horizon w? | **PASS**: +60% safe set over blind; 90% of the gap at w≈0.6–1.0 s | done |
| E013 | E013_rung1_conditioned_critic.py, E013_score.py, E013_diag.py, run_E013_parallel.sh | Rung-1 conditioned RA-SAC critic on the friction toy | critic tracks the family, blind frozen; behavioral parity = eval artifacts | superseded → E013b |
| E013b | same (safety_sb3 main@f128330, γ anneal) | Annealed γ: does the critic match the undiscounted grid? | IoU doubled to ~0.50; blind mis-claims; actor loiters (reach 66→20%) | superseded → E013c |
| E013c | same (GOAL_VALUE 0.3→1.0) | Raise reach reward to break the loiter | IoU 0.53–0.70 (target 0.85); loiter is γ-driven; bicycle line paused, Go2 becomes the flagship | done (partial) |

## (b) E014–E038 — Go2 sloshy payload + B̂ set-membership (safety_sb3 ≤ v0.3.x, SAC; all superseded by the 08-02 soundness reset for value-based claims)

| ID | script(s) | question / delta | outcome | status |
|---|---|---|---|---|
| E014 | E014_bifurcation_eval.py, E014_bifurcation_video.py | Two fixed-payload GameplaySAC specialists: does the strategy flip? | DODGE 4.62 m vs BRACE 0.34 m (14×), 0 falls | superseded (08-20: likely a v0.3 learner artifact; no split under PPO in E043) |
| E015 | E015_readout.py | Conditioned (oracle θ) vs blind, 20M | cond 2.52/0.40 vs blind 1.14/0.37 m drift: modest | superseded → E015b |
| E016 | E016_counterfactual.py | Inject wrong θ | θ causal but backwards off-distribution (θ ≡ physics) | superseded → E016b |
| E015b | E015_readout.py | Hidden-payload re-run, 50M | both adapt; at 50 N cond resists heavy-sloshy 2.02 m vs blind dragged 5.33 m | done (numbers pre-E020) |
| E016b | E016_counterfactual.py | Counterfactual on hidden-payload policies | direction now correct (+0.99/+0.39 m @0.35); within noise at full force | done |
| E017-OOD | E017_ood.py | OOD payloads (12 kg, k=400) | conditioned falls on heavy-rigid OOD (safe 0.18–0.29), blind 1.00 | superseded (E020 metric, E037) |
| E017 | E017_ood.py, E017_strategy.py | K=16 frame-stacked history arm | history OOD-robust and strategy-correct OOD; conservative in-dist; ≈ blind | done (safe_rate pre-E020) |
| E018 | E018_setmembership.py, E018_precheck.py | RL-free set-membership belief B̂ over payload θ | in-dist 225→1–8 candidates, sound, δ*≈0.10–0.20 s; OOD 12 kg → empty at ~0.14–0.20 s (needs a 50 N shake probe) | done (scoped by E038 to a commissioning-time probe) |
| E019 | E019_breakboundary.py, E019_blind_video.py | Mass break-boundary sweep | "raw-θ breaks first; blind most robust to 25 kg" | superseded (E020, E037) |
| E019b | E019b_forcesweep.py | Force sweep 50–250 N: any cell where blind falls but conditioned survives? | ? (not in registry) | ? |
| E020 | E020_corrected.py | Fix the metric (auto-reset masked falls) | all E015–E019 safe_rates wrong; raw-θ hurts under stress; history ≈ blind on static ODD | done |
| E021 | E021_dynamic_odd.py | Mid-episode rigidity jump | history 0.28 vs blind 0.57 falls/env-s at the transient: first positive | done |
| E022 | E022_dynamic_video.py | n=2 dynamic-ODD video | misleading (opposite of the aggregate) | superseded → E024 |
| E023 | E023_dynamic_sweep.py | Shape of change (step/ramp/oscillation) | oscillation worst for blind (~0.5); history flat ~0.2 | done (resonance reading retracted) |
| E024 | E024_dynamic_compare_video.py | 16-robot grids + 128-env curve | picture matches statistic | done |
| E025 | E025_chirp.py | θ-chirp 0.2–7 Hz | broadband plateau, no narrow tongue | done |
| E026 | E026_tongue.py | Frequency × depth map | no resonance tongue; amplitude is the knob; history invariant | done (negative for resonance) |
| E027 | E027_worst_video.py | Worst-modulation video | fall rate 0.50 vs 0.20 (cumulative saturates) | done |
| E028 | E028_worst_recovery_video.py | Recovery-aware video | instantaneous rate: blind high, history low | done |
| E029 | E029_fallback.py | Which θ-agnostic fallback when B̂ is empty? | hard-coded stance-hold 3–10× worse; history the best fallback overall | done |
| E029b/c | E029b_stance_check.py, E029c_stance_mode.py | Stance-hold failure mode | a=0 sinks (can't hold height); lie-down lies outside the standing safe set | done |
| E030 | E030_fallback_contactforce.py | Fallback under a contact-force margin | learned policies stand (0 N) or slam (~4700 N); settle vs slam separable at ~100–200 N | done |
| E031 | E031_handcraft_liedown.py | Handcrafted soft lie-down | can't be faked; topples below ~0.25 m; needs training | negative |
| E032 | — (RSS `go2_payload_descent`, ReachAvoidSAC 1P) | Train a soft-descent fallback | 15M: crouch only; 100M: reaches the target via a fast feet-supported drop (~1.1 m/s) | done (accepted as-is) |
| E033 | E033_descent_eval.py | Eval of the 15M descent | crouches, reach 0.00 | done |
| E034 | E034_descent_force_profile.py | Force profile of the 75M descent | 0 N non-foot contact; the gap is unbounded descent speed | done |
| E035 | E035_filter_handoff.py | B̂ → descent handoff on OOD 12 kg + 20 N | filter made it worse (topple 0.41 vs 0.00) | negative |
| E036 | E036_ckpt_diag.py | Is history's ~50% undertraining? | No: the final ckpt was a trough; in-dist safe_rate anti-predicts dynamic robustness | done |
| E037 | E037_break_sweep.py | Where do the arms really break (corrected metric)? Lead time? | break at ≥12 kg; lead time 1.3–3.4 s; force-dependent (cond best at 0 N, worst at 50 N) | done |
| E038 | E038_three_timestamps.py | V vs B̂ as failure detector | V leads failure by +0.36–0.50 s, no false positives; windowed B̂ invalid (chaos ≫ θ signal) | done (B̂ negative; V on a pre-0.4.0 critic) |

## (c) E039–E045 — safety_sb3 / RSS 0.4.0 soundness reset, SAC → PPO

| ID | script(s) | question / delta | outcome | status |
|---|---|---|---|---|
| E039 | — (RSS `go2_payload_{blind,history}`, ReachAvoidSAC2P, ws3) | Retrain blind and history under the sound 0.4.0 critic | collapse: blind peak 0.63, history ≈0 | negative (confounded by the SAC timeout bug) |
| E040 | E040_attribution.py | Is the RSS 0.4.0 port clean? (graft v0.3 actor) | every cell matches E037: port clean | done |
| E041 | E041_warmstart.py | Warm-start a competent policy under the hard target | 0.98 → 0.13–0.65 oscillation | negative (confounded) |
| E042 | — (`go2_payload_{light_rigid,heavy_sloshy}.yaml`, ws4) | Fixed-θ specialists under sound SAC | neither converges (0–0.4; one transient 0.898) | superseded (SAC timeout bug → PPO) |
| E043 | — (`go2_payload_*_ppo.yaml`, ReachAvoidPPO2P, ws4) | PPO specialists, clean 4-corner 2×2 | all train stably; robustness smooth/monotone (~23–46 N), no strategy split; checkpoints lost with ws4 | done (ckpts lost) |
| E044 | — (constant 50 N, curricula off) | Does a split appear without the curriculum? | cold-start collapse (failure ~1.0), killed | abandoned |
| E045 | — (curriculum 2×2, buzi-pc) | Regenerate the specialists for the swap matrix | trained (`results/go2_payload_runs/go2_payload_*_adv`); no dedicated swap-matrix result, though E049 calls the payload matrix "flat" | ? |

## (d) E046–E065 — leg-degradation ODD: negative under "stand", E060 positive under graceful degradation, E064 kills runtime θ

| ID | script(s) | question / delta | outcome | status |
|---|---|---|---|---|
| E046 | — (`go2_stabilize_ppo`, `go2_broken_leg_ppo`) | Normal vs dead-FR-leg specialists | dead leg infeasible by construction (corner sag 39% at rest); normal trains | negative → E047 |
| E047 | — (`go2_weak_leg_ppo`, 25 N) | 50% FR torque + smaller adversary | weak50 and normal@25N trained (used in E049/E069) | done |
| E048 | — (`go2_weak_leg_20_ppo` @30 N) | 20% FR torque | never converged (0.97 failure) | abandoned → E050 |
| E049 | E049_torque_sweep.py | Survival swap matrix over test θ | normal vs weak50 curves cross at θ≈15–20%: a moderate bifurcation signature | done |
| E050 | — (`go2_weak_leg_20_ppo` @25 N) | Retrain the 0.2 specialist | still struggles (failure 0.771) | done (weak reference) |
| E051–E053 | — (`go2_weak_leg_{blind,conditioned,history}_ppo`) | Randomized-θ arms (θ hidden / exposed / K=16 history) | training failure 0.294 / 0.125 / 0.262 | done |
| E054 | E054_conditioned_eval.py, E054_plot.py | Payoff eval under a common scripted pull | conditioned ≈ blind; history worse (overfit to its adversary) | negative |
| E055 | E055_dynamic_ramp.py | Sudden mid-episode leg failure | no commitment window; blind absorbs it reactively | negative |
| E056 | — (`go2_weak_leg_20_soft`) | Soft-rest objective (no-slam < 80 N + level, then + settled) | level-only tipped; adding "settled" fixed it (0.52 failure, no tipping) | done |
| E057–E059 | — (`go2_weak_leg_{blind,conditioned,history}_soft`) | Soft-rest arms | trained; used by E060–E063 | done |
| E060 | E060_soft_ramp.py, E060_value_fig.py | Dynamic leg death scored by soft-rest | conditioned 0.93 > history 0.82 > blind 0.51 @15 N (7× @25 N) | superseded (E064 confound) |
| E061 | E061_soft_ramp_video.py | Red-leg videos at 20/10/0% | videos; θ=0 all collapse | done |
| E062 | E062_odd_dynamics.py, E062_plot.py | θ(t) shape sweep (step/ramp/sine/recover) | conditioned shape-invariant 0.91–0.94; blind worst on step | done (inherits E064 confound) |
| E063 | E063_fl_break.py | Break FL instead of FR | conditioned's edge shrinks +0.41 → +0.16; structured θ is narrow | done |
| E064 | E064_eval.py, E064_rma.py, E064_theta_ablation.py | Fair fight: dynamic-θ training, RMA baseline, 3 seeds (`E064_s{0,1,2}`, the only multi-seed training), θ-ablation | runtime θ is not load-bearing (frozen 0.2 beats oracle, inverted beats oracle); the E060 win = privileged training | done (kills runtime-θ) |
| E065 | — | Close the loop with per-leg B̂ from torque residuals | planned 08-21, never run | abandoned (pivot to spec family) |

## (e) E066–E068 — is there a commitment window anywhere on the quadruped? (identifiability gates)

| ID | script(s) | question / delta | outcome | status |
|---|---|---|---|---|
| E066 | E066_alias.py | Shove vs slip aliasing gate | identified in 2 steps (40 ms) ≪ commitment deadline: absorbable; pre-registered kill | negative |
| E067 | E067_tconflict.py | Left vs right leg death: t_conflict | a symmetric hedge is universally safe; linear ID looks slow (AUC 0.73); underpowered | negative |
| E068 | E068_tight_identify.py | t_identify with a latent (1D-conv) encoder | 3 steps (60 ms): the standing quadruped senses fast; caveat that ID is load-gated (locomotion differs) | negative |

## (f) E069–E079 — weight ladder + leg/compound spec families: the spec-family thesis

| ID | script(s) | question / delta | outcome | status |
|---|---|---|---|---|
| E069 | E069_weight_gate.py | G1 physics gate: zero-shot stand vs rest under constant load W | standing degrades with W (balance, not saturation); rest probe invalid | done |
| E069b | E069b_descent_probe.py | Scripted-fold descent probe | low crouch holds at W 90–150 where standing fails: handoff window real (G1 PASSED) | done |
| E070 | E070_matrix.py (+ trains `go2_weight_{stand,rest}`) | Train V_stand(x,W), V_rest(x,W) with a flat load | stand 0.19 / rest 0.000 training failure; clean behavioral bifurcation; two-sided envelope | superseded (iteration 1; flat load stabilizes) |
| E071 | E071_handoff.py, _value_util.py | Certified handoff on a W ramp | V_stand contracts 0.044→−0.012; disturbance-aware trigger (W≈204 @10 N vs 141 @25 N) beats the fixed-W oracle on slam | superseded (iteration 1) |
| E072 | E072_video.py | Video + timeline of the handoff | switched 91% at W≈209 | superseded (iteration 1) |
| E073 | E073_hicom_probe.py (+ trains stand_hi, rest_hi, unified_hi, unified_disc_hi) | High-CoM load (h=0.25) + unified single-spec baselines | pendulum validated; training failure 0.779 / 0.000 / 0.000 (lazy) / 0.049 | done |
| E074 | E074_matrix.py, E074_value.py, E074_ramp.py, E074_video.py, E074_tune.py | High-CoM eval battery + gust ramp | stand_hi feasible W≲150; unified descends lazily; **certification deficit** (V_unified rises as standing dies); handoff 2.3× safer than stand-only; V noisy | done (trigger numbers superseded by E075) |
| E075 | E075_ramp.py, E075_value.py (+ trains stand_hi `hi=120`) | **Recalibrated** stand_hi on the comfortably-feasible band W∈[0,120] | discrimination 0.97→1.16, tip 0.21→0.13, still above oracle 0.03–0.05 | done (headline STAND expert) |
| E076 | E076_matrix.py, E076_value.py, E076_video.py (+ trains `go2_leg_{stand,rest}`) | Leg spec family at stance | V_leg_stand flat (0.18), handoff skipped: matched **negative control**; leg_rest at θ=0 tip 0.00 | done |
| E077 | E077_figure.py | F7 cross-demo contraction figure (weight vs leg) | weight contracts, leg flat | superseded (3-curve F7 in E078, polished in E093) |
| E078 | E078_gate.py, E078_matrix.py, E078_value.py, E078_ramp.py, E078_video.py, E078_figure.py (+ trains `go2_compound_{stand,rest}`) | Compound: leg death while carrying W=80 N at h=0.25 | gate θ_c≈0.3; V contracts (discrimination 1.54); handoff tip 0.20 vs stand-only 0.63 (3×), switch at θ≈0.24 | done (demo 2) |
| E079 | E079_region_sweep.py | 252-cell certifiable-region sweep, F1/F2/F4 | value-triggered switches land on the measured feasibility boundaries (within 7% / Δθ 0.02) | done |

## (g) E080–E088 — bidirectional switching and certified transitions / automaton

All survival numbers in this block from before the 08-23 accounting fix are **superseded** (see the top table).

| ID | script(s) | question / delta | outcome | status |
|---|---|---|---|---|
| E080 | E080_bidirectional.py, E080_video.py | Bidirectional handoff (stand back up when the ODD recovers) | "works"; ~0.026 tip per transition (with-respawn metric); zero-shot get-up from prone 96–100% | superseded (E081, accounting fix) |
| E081 | E081_survival.py | No-respawn survival curves | inverts E080: ONE-WAY > BIDIR in all 4 conditions; REST-ONLY best survival | superseded numbers (accounting fix); the qualitative lesson stands |
| E082 | — | — | no record anywhere (ID apparently skipped) | ? |
| E083 | — (trains `go2_getup`, `go2_descend` v1/v3/v4) | Transitions as reach-avoid funnels | getup v1 0.268 failure; descend v1/v3 dead (1.000, spawn bug); v4 0.000 after the spawn fix + warm-start from rest_hi | done |
| E084 | E084_automaton.py | 4-state automaton vs batch-1 arms (survival + affordance) | first pass: V2-REUSE beat V1 2–3×; return edge fixed; "descent has a physical floor" | superseded (guard bugs, V_up miscalibration, accounting); corrected tables 08-23 evening, seeded in E094 |
| E084b | E084b_early.py | Earlier or forced descent triggers | null: descending deaths unchanged (uses getup v1) | superseded numbers |
| E085 | E085_video.py | Batch-2 videos | 3-panel videos; re-rendered with corrected accounting via E090 | done |
| E086 | E086_single_pulse.py | Single excursion (one pulse / one period, 24 s) | diagnostic found guard bugs (EMA-from-0, EPS_UP too permissive) → fixes; rerun with getup_v2: the return works | superseded → E086c |
| E086c | E086_single_pulse.py | Timeout fix (20 s episode < 24 s horizon) | pulse/benign V2 0.52/0.71 (survival/affordance) vs ONE-WAY 0.78/0.46 | superseded (accounting fix) |
| E087 | — (trains getup_v2, warm-started from v1) | Retrain the get-up on measured settled-rest spawns | failure 0.424 @ force 0.71; V_up calibrated; return fires, stuck-in-REST gone | done (headline get-up) |
| E088 | — | Weight-conditioned walker | superseded before launch: use the existing weight-naive walker | abandoned |
| (corrected) | E084/E086 reruns | CORRECTED FINAL TABLES (08-23 evening) | REST-ONLY 1.00 everywhere; STAND-ONLY 0.00–0.07; V2 best switching arm on ramps (0.75 vs ONE-WAY 0.61); ONE-WAY edges square steps; gusty: everyone except REST ≤0.25 | done (valid) |

## (h) E089–E094 — walking evaluations, figures, seeds

| ID | script(s) | question / delta | outcome | status |
|---|---|---|---|---|
| E089 | E089_goal_walk.py | Goal walk (9 m) with the naive walker under a single pulse/period load + value filter | run-1 artifacts fixed; illegal_contact leak found; corrected: the filter protects walking on ramps (walking deaths 1–3 vs 164–249), steps outrun any reactive filter (safe ≤0.10) | superseded for walking-success claims (→ E089v2); ramp-vs-step finding stands |
| E089v2 | E089_goal_walk.py | 12 m goal, excursion at [5,13) s, 30 s horizon, + medium condition | walking under weight excursions is infeasible for ALL arms (safe ≤0.11): a boundary finding | done |
| E089-loose | E089_goal_walk.py | Death = tilt > 80° only | numbers barely move; deaths are genuine flips; 5–9% false descents while walking | done |
| trigger recal | ? (no script kept; log only) | ROC sweep over (ε, K) on healthy walking | (−0.15, K=10): 1.6% false sits / 30 s (was 47.7%) | done |
| E090 | E090_walk_viz.py | Walking visualization suite (corrected) | top-downs, distance timelines, videos | done |
| E091 | E091_leg_walk.py | Walking with an FR thermal derate (leg fault) | V̄_stand is blind to actuator faults → torque-saturation residual detector; V2 success 0.02→0.26 (top of the ladder) | done |
| E092 | E092_payload_walk.py | Payload-swap walking (W 0 ↔ 220 N at h 0.35; pulse/period/dip) | period: ODD-cond 0.38/0.39 vs ONE-WAY 0/0.42, WALK-ONLY 0/0; dip: full 0.40/0.40 > direct 0.33/0.36; pulse ≈0.02 for all (maneuver latency); ~45–48% brake deaths, gait-phase structured | done (**headline**) |
| E093 | E093_figpolish.py | Figure polish + paper naming | F2/F7/F3 reworked; weight 1.21σ, compound 1.54σ, negative control 0.48σ | done |
| E094 | E094_seeds.py | 3 extra eval reps (N=256) on E092, E084 waves, E086 single | everything stable (sd ≤0.04); single period/benign V2 0.74±0.01 vs ONE-WAY 0.62±0.01; pulse ONE-WAY 0.54±0.04 vs V2 0.49±0.01 | done (eval reps only; policies are single-seed) |

Naming used in the paper (E093): system = "ODD-conditioned" = arm **V2** (dedicated descend_v4 descent); "ODD-conditioned
(direct)" = **V1**; V2-REUSE (rest_hi descent) is not shown separately. Project phases ↔ the tickets they answered:
E042–E045 sound-specialists question (left open, superseded by the spec-family pivot); E046–E064 leg-degradation
bifurcation; E069–E075 weight ladder; E076/E078 leg family + compound; E083–E087 certified transitions; E092–E094
paper results.
