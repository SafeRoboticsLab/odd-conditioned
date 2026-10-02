# Findings

What this project established, as of the paper draft v3 (2026-08-24). The draft itself, with detailed figure
walkthroughs, is `reference/PAPER-draft/REPORT.md` (after `scripts/fetch_bundles.sh --reference`).
Experiment IDs (E0XX) point to `experiments/E0XX_*.py` and [EXPERIMENTS.md](EXPERIMENTS.md).

## 1. The claim

When a robot's operating design domain (ODD) changes at runtime — a payload is loaded, a motor derates — the
right place to condition a safety filter is the **specification**, not the policy. A family of reach-avoid
specifications, each with a certified expert for the strongest spec feasible in its mode, joined by
**certified transitions** and switched by **belief-primary triggers checked by value certificates**, is

1. **safer** than any fixed policy across ODD dynamics, and
2. **permissive**: it withdraws to the safe mode only while needed and resumes the task when the ODD clears.

Baselines are single fixed-ODD filters — the classical ISAACS / gameplay-filter construction. Off-design such a
filter is either unsafe (it keeps the affordance the ODD no longer supports) or, if built for the worst case,
useless (no affordance at all).

## 2. Why the specification and not the policy (the negative results that shaped the claim)

Most of the project's first two months tested **policy-level** ODD conditioning (give the policy θ, or a history
to infer θ). It does not hold up:

- **A reactive policy absorbs a single-axis ODD change.** With a sound learner (safety_sb3 0.4.0, PPO), the
  dodge-vs-brace payload "bifurcation" seen earlier disappeared — robustness was smooth across the payload 2×2
  (E043). A sudden leg death is absorbed reactively by a blind policy (E055). The standing quadruped identifies
  a shove vs a slip, or which leg died, within 40–60 ms (E066–E068), so there is no window in which knowing θ
  would matter.
- **The one positive instance did not survive a fair fight.** Under a graceful-degradation objective, a θ-conditioned
  policy beat blind 0.93 vs 0.51 on a dynamic leg death (E060). E064 then showed that feeding the *same* trained
  network a frozen or even inverted θ works as well as the true θ: the gain came from privileged training (as in
  RMA), not from runtime θ.

What θ **does** change is **what can be certified**: at low load the robot can certifiably stand; past a boundary,
only lying down is certifiable. That geometry — and a runtime object that tracks it — is the contribution.

## 3. The certificate tracks certifiability (paper §1: F2, F7, F3)

- **The STAND-certifiable region is an island** in (ODD × disturbance) space: at 35 N of pull it ends near
  W≈90 N, at 5 N it reaches past W≈130 N (E079). The value-triggered handoff fires essentially on that measured
  boundary (W ≈ 122 ± 64 N across the fleet, the spread from gust timing; θ ≈ 0.24 on the leg axis), within 7 % /
  Δθ 0.02 of the sweep — **without the trigger ever being told the boundary**.
- **The certificate contracts if and only if the ODD change removes certifiability** (F7, contraction in units of
  the value's own state spread σ): carried load with a raised CoM contracts **1.21σ** (E075), a leg dying while
  loaded **1.54σ** (E078), and the unloaded leg — which the reactive policy absorbs — only **0.48σ** (E076, the
  matched negative control: the filter correctly never fires).
- **A single merged specification hides the failure** (F3, E074). A "unified" stand-or-rest spec trained as one
  policy has a value that *rises* (to ~0.87) while the robot loses the stance, because resting is ever easier to
  satisfy. A monitor on it would report improvement. Only the family's per-mode value signals the switch. The
  unified policy also descends early, giving up standing long before necessary.

## 4. The system (paper §2)

Two certified modes and two certified transitions, each a reach-avoid problem with a learned value:

| object | role | checkpoint |
|---|---|---|
| STAND / WALK | task affordance (stance expert; walking uses a pretrained nominal walker) | stand_hi (E075) / go2_atomic_skills walker |
| REST | the anchor safe set — survives every scenario tested at 1.00 | rest_hi (E073) |
| descent | RA(stand → rest) funnel, entered quasi-statically | descend_v4 (E083) |
| return | RA(rest → stand) get-up funnel + its certificate `V_up` + certified abort | getup_v2 (E087) |

Design rules distilled from the experimental arc: **(1)** belief-primary triggering — the value is a certificate,
not a fault detector (on walking gaits it false-fires 20–25 %; it is blind to actuator faults); **(2)**
per-regime calibration of certificate thresholds; **(3)** enter every transition inside its funnel's trained
initial set (brake before descending, get up only after unloading); **(4)** debounce the return.
Full guard-by-guard specification: [SWITCHING_LOGIC.md](SWITCHING_LOGIC.md).

## 5. Results

All rates over N=256 robots, no respawn; ± = sd over 4 evaluation repetitions.

**Walking under a payload swap (E092, paper §3).** A tall 220 N crate is loaded mid-route for 8 s; the goal
is unreachable before it arrives, so success = resuming.

| success / safe | pulse (instant) | period (ramp) | dip (deceptive ramp) |
|---|---|---|---|
| WALK-ONLY | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| REST-ONLY | 0.00 / 1.00 | 0.00 / 1.00 | 0.00 / 1.00 |
| ONE-WAY | 0.00 / 0.02 | 0.00 / 0.42 | 0.00 / 0.42 |
| ODD-conditioned (direct) | 0.03 / 0.03 | 0.37 / 0.37 | 0.33 / 0.36 |
| **ODD-conditioned** | 0.03 / 0.03 | **0.38±.02 / 0.39±.02** | **0.40±.02 / 0.40±.02** |

The full system keeps ONE-WAY's safety on ramps while turning 0 % success into ~40 %. On the honest ramp the
direct ablation ties it; the deceptive dip separates them — +0.07 success, +0.04 safety on average over 4 reps
(the blind
return stands into the returning crate). **The pulse is a boundary, not a detection failure**: an instantly applied 220 N tall load
flips a mid-stride robot in ~0.3 s, faster than any detect + brake + descend (~2.5 s). Reactive filtering needs
the ODD to change slower than the maneuver; faster changes need forecasting.

**Standing under load changes (E084/E086, paper §4).** ODD-conditioned is the best switching arm on smooth
changes (single benign/period **0.74±.01** vs ONE-WAY 0.62±.01, direct 0.21±.02); ONE-WAY edges it on steps and
square waves (benign/pulse 0.54±.04 vs 0.49±.01) because every return costs another descent. REST-ONLY is 1.00
in all eight scenarios; STAND-ONLY ≤ 0.06. Under gusts every non-rest arm bleeds while still standing (≤ 0.25).

**Leg fault while loaded (E078, paper §5).** On a second ODD axis the same structure appears: the stance expert
holds ≥ 0.95 stance fraction down to θ=0.3, then tips 49 % (θ=0.2) and 58 % (θ=0.1); the rest expert is tip-free
across θ. The handoff switches at θ≈0.24 and cuts tips ~3× (0.20 vs 0.63 for stand-only) while keeping 0.57
affordance — but **slams (non-foot contact > 184 N) barely improve (0.53 vs 0.57)**: the descent on a dying
leg under load is the weak point. Demonstration-level evidence only (one run, N=128; `E078-compound-demo/`).

**Leg fault while walking (E091, paper §6).** A motor derate is invisible to every observation channel and to
`V_stand` (a hobbling robot reads +0.07), but a torque-saturation residual — demanded PD torque vs achieved —
reads exactly 0 on a healthy gait and fires during the derate. With it, the certified automaton has the best
success of any arm (0.28 vs walker 0.20; one run).

## 6. Findings worth knowing beyond the headline

- **Transitions near the failure boundary are where robots die.** Benign standing deaths concentrate in
  DESCENDING; walking deaths in BRAKE (~45 %, the walk→stand handoff under a rising load).
- **Brake mortality is gait-phase structured**: braking mid-swing dies, from stance survives. Identical gait clocks
  made solo-robot deaths deterministic; the fleet's ~45 % is the phase average. A trigger that defers the handoff
  to the next stance phase is an obvious training-free improvement (E092 iteration-5).
- **A load-naive walker cannot carry these loads** (E089): with the 12 m protocol every walking arm is ≤ 0.05
  safe; settling under peak load from a gait flips robots. The criterion is robust (flip-over-only accounting
  gives the same picture).
- **Count deaths with the environment's own termination flags.** Recomputing failure outside the env missed
  resets whose sensor buffers were already cleared (93 of 94 in the worst probe) and inflated every survival
  number until the 2026-08-23 correction.
- **Certificates are ordinal off their training distribution.** Thresholds had to be recalibrated per regime
  (walking ROC: −0.15 / K=10 gives 1.6 % false sits per 30 s, vs 47.7 % at the standing default).
- **The set-membership belief B̂ (E018)** — RL-free trajectory matching over (rigidity, mass) — shrinks to the
  true payload in 0.10–0.20 s and becomes *empty* for an out-of-distribution 12 kg payload: an OOD flag a point
  estimate or learned encoder cannot produce. Using it to trigger a naive rescue made things worse (E035), which is
  why it is positioned as the trigger/detection layer, not a rescue. Write-up: `docs/history/bhat_certificate.md`.

## 7. Caveats — state these in any paper

- **Single training seed.** Every policy was trained once (seed 0). The "4 reps" are evaluation repetitions.
- **The belief is an oracle in the walking evaluations.** The return gate and the E092 descent trigger read the
  true scheduled W or θ, standing in for an estimator such as B̂. Only E091's residual detector is a real
  estimator. See [SWITCHING_LOGIC.md §5](SWITCHING_LOGIC.md#5-what-belief-means-in-these-evaluations--read-this-before-extending).
- **Learned certificates, not theorems.** `V ≥ 0` is a learned reach-avoid value; there is no formal verification.
  Trigger precision is bounded by value noise near the feasibility edge (handoff tip 0.13 vs oracle 0.03–0.05).
- **Designed ODD processes.** Schedules (pulse / ramp / dip / waves) are hand-designed; E092's heavy phase
  (h=0.35) is outside the twins' training (h=0.25) on purpose.
- **Reproducibility gaps.** No training launch command was recorded (reconstructed from each run's
  `config.yaml`, see [TRAINING.md](TRAINING.md)); the sandbox env code was committed only on 2026-10-02, after
  training, so HEAD reproduces the final env state (getup v1 / descend v1–v3 spawns are not recoverable);
  draft figures F1, F4, F5, F6, F8 have no producing script.

## 8. Positioning (from the literature triangulation, 2026-07)

Not new: parameter-conditioned reachability (Borquez, Nakamura, Bansal), RMA-style adaptation, the classical
safe-parking / fault-tolerant handoff condition (Gandhi & Mhaskar 2008 — cite in the first paragraph), drift
re-inflation of set estimates (Lorenzen). The contribution is the **instantiation and scaling**: maximal
(adversarially learned) reach-avoid certificates per specification mode at 36-D model-free scale, the
certified transition funnels, and the empirical demonstration that the certificate — not the policy input — is
where ODD information is load-bearing.

## 9. Naming map (code ↔ paper)

| in code / videos | in the paper |
|---|---|
| `V2` (dedicated descend_v4 funnel) | **ODD-conditioned** |
| `V1` (batch-1 bidirectional) | ODD-conditioned (direct) |
| `V2-REUSE` (rest_hi descent) | not shown separately; used in E089 / E091 |
| `stand_hi` recal (E075) | the STAND expert |
| `rest_hi` (E073) | the REST expert |
| `getup_v2` (E087) / `descend_v4` (E083) | the return / descent funnels |
| `unified_hi`, `unified_disc_hi` (E073) | unified-spec baselines |
