# Findings

What the results establish. The results of §3 and §5 are produced by `scripts/reproduce.sh` with the published
checkpoints ([REPRODUCE.md](REPRODUCE.md) maps each to its target; figures land in `outputs/figures/`). The
rationale in §2, §4 and §6 summarizes development experiments that are not part of this code release.

## 1. The claim

When a robot's operating design domain (ODD) changes at runtime — a payload is loaded, a motor derates — the
right place to condition a safety filter is the **specification**, not the policy. A family of reach-avoid
specifications, each with a certified expert for the strongest specification feasible in its mode, joined by
**certified transitions** and switched by **belief-primary triggers checked by value certificates**, is

1. **safer** than any fixed policy across ODD dynamics, and
2. **permissive**: it withdraws to the safe mode only while needed and resumes the task when the ODD clears.

The baselines are single fixed-ODD filters — the classical construction of one safety filter for one
specification. Off-design such a filter is either unsafe (it keeps an affordance the ODD no longer supports)
or, built for the worst case, useless (no affordance at all).

## 2. Why the specification and not the policy

Conditioning the **policy** on the ODD parameter θ (giving it θ, or a history to infer θ) does not hold up:

- **A reactive policy absorbs a single-axis ODD change.** With a sound two-player learner, robustness is smooth
  across payload mass and stiffness; a sudden leg death is absorbed reactively by a policy that never sees θ;
  and the standing quadruped tells a shove from a slip, or which leg died, within 40–60 ms — there is no window
  in which knowing θ matters.
- **The one positive instance did not survive a fair test.** Under a graceful-degradation objective a
  θ-conditioned policy beat a blind one on a dynamic leg death — but feeding the same network a frozen or even
  inverted θ worked as well as the true θ: the gain came from privileged training, not from runtime θ.

What θ **does** change is **what can be certified**: at low load the robot can certifiably stand; past a
boundary, only lying down is certifiable. That geometry — and a runtime object that tracks it — is the
contribution.

## 3. The certificate tracks certifiability

`python scripts/evaluate.py certificates` — figures `certifiable_regions.png`, `certificate_contraction.png`,
`certification_deficit.png`, `compound_timeline.png`.

- **The STAND-certifiable region is an island** in (ODD × disturbance) space. Under a 5–10 N push the stand
  expert holds up to W ≈ 140–150 N; under 20 N only between W ≈ 30 and 110 N; above ~25 N nowhere. The REST
  expert fails ≤ 4 % over the whole plane. On the demo ramp the value-triggered handoff fires at W ≈ 121 ± 62 N
  (the spread is gust timing), at or before the boundary at the ambient push; on the compound axis it fires at
  θ ≈ 0.28, on the boundary — **without the trigger ever being told where the boundary is**.
- **The certificate contracts if and only if the ODD change removes certifiability.** In units of the value's
  own spread over states: a load with a raised centre of mass contracts the stand certificate by **1.30σ**, a leg
  dying while loaded by **1.57σ**, and the unloaded leg — a fault the stance policy absorbs reactively — by only
  0.52σ, with a band discrimination of 0.02: the matched negative control, where the filter correctly never fires.
- **A single merged specification hides the failure** (`certification_deficit.png`). A "unified" stand-or-rest
  specification trained as one policy has a value that *rises* (to ~0.85) while standing becomes impossible,
  because resting is ever easier to satisfy: a monitor on it would report improvement. Only the family's
  per-mode value signals the switch. The unified policy also gives up standing early (it stands 13 % of the time
  vs 47 % for the handoff).
- **Train the stand expert on the feasible band.** Trained on W ∈ [0, 150] N, most of it infeasible, the
  certificate is noisier (discrimination 0.96 vs 1.17 for W ∈ [0, 120]) and the handoff tips more (0.19 vs 0.14).
- **The handoff has a mechanically safe window.** Forcing the switch at a fixed load shows tips ≤ 6 % for
  switches between 40 and 120 N and ≥ 20 % from 160 N; the value trigger lands in that window.

## 4. The system

Two certified modes and two certified transitions, each a reach-avoid problem with a learned value:

| object | role | checkpoint |
|---|---|---|
| STAND / WALK | the task affordance (stance expert; walking uses a pretrained nominal walker) | `stand` / go2_atomic_skills walker |
| REST | the anchor safe set — safe in every scenario tested | `rest` |
| descent | reach-avoid STAND → REST funnel, entered quasi-statically | `descend` |
| return | reach-avoid REST → STAND get-up funnel, its certificate `V_up`, and a certified abort | `getup` |

Design rules distilled from the experiments: **(1)** belief-primary triggering — the value is a certificate,
not a fault detector (on walking gaits it false-fires on 20–25 % of healthy gaits; it is blind to actuator
faults); **(2)** per-regime calibration of certificate thresholds; **(3)** enter every transition inside its
funnel's trained start set (brake before descending; get up only after unloading); **(4)** debounce the return.
Guard by guard: [SWITCHING_LOGIC.md](SWITCHING_LOGIC.md).

## 5. Results

All rates over N = 256 robots, no respawn; ± = sd over evaluation seeds 0–3 (tables and protocols:
[REPRODUCE.md](REPRODUCE.md)).

**Walking under a payload swap.** A tall 220 N crate is loaded mid-route for 8 s; the goal is unreachable before
it arrives, so success means resuming.

| success / safe | pulse (instant) | period (ramp) | dip (deceptive ramp) |
|---|---|---|---|
| WALK-ONLY | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| REST-ONLY | 0.00 / 1.00 | 0.00 / 1.00 | 0.00 / 1.00 |
| ONE-WAY | 0.00 / 0.02 | 0.00 / 0.42±.03 | 0.00 / 0.42±.02 |
| ODD-conditioned (direct) | 0.02 / 0.02 | 0.39±.03 / 0.40±.03 | 0.38±.03 / 0.41±.03 |
| **ODD-conditioned** | 0.02 / 0.02 | **0.43±.02 / 0.43±.02** | **0.43±.03 / 0.43±.03** |

The full system keeps ONE-WAY's safety on the ramps while turning zero success into ~0.43, and every robot it
keeps alive reaches the goal. Paired by seed it beats the direct ablation every time: +0.03–0.04 success on the
honest ramp, +0.04–0.07 success and +0.02–0.03 safety on the dip, where the blind return takes the bait and
stands into the returning crate. **The pulse is a boundary, not a detection failure**: an instantly applied 220 N
tall load flips a mid-stride robot in ~0.3 s, faster than any detect + brake + descend (~2.5 s). Reactive
filtering needs the ODD to change more slowly than the maneuver; faster changes need forecasting.

**Standing under load changes.** REST-ONLY is safe in all eight scenarios and STAND-ONLY ≤ 0.07. The full system
is the best switching method on the smooth single excursion (benign/period **0.74 ± .03** vs ONE-WAY 0.62, direct
0.25) and beats its direct ablation in every scenario and seed. Both certified pieces pay: with the same descent,
the certified return lifts benign/period from 0.25 (direct) to 0.54 (rest descent), and the dedicated descent
funnel lifts it further to 0.74. ONE-WAY edges the full system on steps and square waves (benign/pulse 0.56 vs
0.51, square waves 0.57 vs 0.47) because every return costs another descent, and descents are where
benign-push deaths happen. Under 35 N gusts every method except REST loses robots while still standing (≤ 0.25).

**Leg fault while walking.** A motor derate is invisible to the observations and to `V_stand` (a hobbling robot
reads +0.07), but a torque-saturation residual — demanded PD torque vs achieved — reads ≈ 0 on a healthy gait and
fires during the derate (median 7.9 s, never before the fault). With it the full system has the best success of
any method: **0.28 / 0.28** vs WALK-ONLY 0.19 / 0.19, direct 0.16 / 0.19, ONE-WAY 0.07 / 0.27 (seed 0).

**Leg fault while loaded (compound).** On a second ODD axis the same structure appears: the compound stance
expert holds ≥ 0.97 stance down to θ = 0.3, then tips 51 % (θ = 0.2) and 63 % (θ = 0.1); the rest expert is
tip-free at every θ. On a ramp where the leg dies while carrying 80 N, the handoff switches at θ ≈ 0.28 and
halves the tips (0.28 vs 0.63 for STAND-ONLY) while keeping the stance 55 % of the time — but **slams (non-foot
contact above 184 N) improve only from 0.52 to 0.40**: descending on a dying leg under load is the weak point.
Demonstration-level evidence (one ramp profile, one seed).

**Weight walking — the boundary.** The load-naive walker carrying 220 N at h = 0.25 m leaves every walking method
at safe ≤ 0.03 in every push condition (REST-ONLY 1.00): walking under this load is infeasible, and settling from a
gait under peak load flips robots. This is why the payload walk has an unloaded light phase.

## 6. Findings beyond the headline

- **Transitions near the failure boundary are where robots die.** Standing deaths under benign pushes
  concentrate in DESCENDING; ~80 % of the payload walk's deaths happen in BRAKE (the walk → stand handoff under a
  rising load).
- **Brake mortality depends on the gait phase**: braking mid-swing dies, from stance survives. Identical gait
  clocks made the solo-robot deaths deterministic, so walking starts are staggered; a trigger that defers the
  handoff to the next stance phase is an obvious training-free improvement.
- **A load-naive walker cannot carry these loads**: walking under a 220 N excursion at h = 0.25 m leaves every
  walking method unsafe (the weight walk, §5). The payload walk therefore uses an unloaded light phase.
- **Count deaths with the environment's own termination flags.** Recomputing failure outside the environment
  misses resets whose sensor buffers were already cleared and inflates survival.
- **Certificates are ordinal off their training distribution.** Thresholds are calibrated per regime: the
  standing descent threshold read on a walking gait gives ~48 % false descents per 30 s; the walking threshold
  (−0.15, 10 steps) gives 1.6 %.

## 7. Caveats

- **Each policy is a single training run**, and the stance-type policies vary a lot from run to run: retrained
  with other seeds, most stand experts fail the acceptance test that the published one passes
  ([TRAINING.md §5](TRAINING.md#5-training-variance-and-acceptance)). The seeds in the tables are evaluation
  seeds. The structural results (anchors, the pulse boundary, the weight-walk boundary, the leg-fault ordering,
  the certification deficit) also hold for retrained policies; the walking and standing *levels* depend on the
  stand expert's braking and loaded-stance quality.
- **The belief is an oracle in the walking evaluations.** The return gate and the payload descent trigger read
  the true scheduled W or θ, standing in for an estimator. Only the leg-fault residual is an estimated signal.
  See [SWITCHING_LOGIC.md §5](SWITCHING_LOGIC.md#5-what-belief-means-in-these-evaluations--read-this-before-extending).
- **Learned certificates, not theorems.** `V ≥ 0` is a learned reach-avoid value; there is no formal
  verification. Trigger precision is bounded by value noise near the feasibility edge.
- **Designed ODD processes.** The profiles (pulse / ramp / dip / waves) are hand-designed; the payload walk's
  heavy phase (h = 0.35 m) is deliberately outside the twins' training (h = 0.25 m).
- **Evaluation seeds, not training seeds.** Rollouts are seeded; methods share initial conditions within a
  seed, so compare them paired by seed. (Loading a stable-baselines3 checkpoint reseeds torch with its training
  seed; the evaluation reseeds after loading, so its seeds are independent.)
- **The evaluation push is scripted.** A constant 10 N lateral push (gusts to 35 N where stated) replaces the
  learned adversary at evaluation; without it every method that stands survives more, and the full system's
  margin over the baselines grows.

## 8. Positioning

Not new: parameter-conditioned reachability, RMA-style adaptation, the classical safe-parking / fault-tolerant
handoff condition (Gandhi & Mhaskar, 2008), re-inflation of set estimates under drift. The contribution is the
**instantiation and scaling**: maximal (adversarially learned) reach-avoid certificates per specification mode
at 36-D, model-free; the certified transition funnels; and the empirical demonstration that the certificate —
not the policy input — is where ODD information is load-bearing.

## 9. Naming (code ↔ text)

| code | text |
|---|---|
| method `odd` | **ODD-conditioned** |
| method `direct` | ODD-conditioned (direct) |
| method `odd-rest-descent` | ODD-conditioned with the REST expert as descent (ablation) |
| `one-way`, `task-only`, `rest-only` | ONE-WAY, STAND-ONLY / WALK-ONLY, REST-ONLY |
| checkpoints `stand`, `rest` | the STAND and REST experts |
| checkpoints `getup`, `descend` | the return and descent funnels |
| `unified`, `unified_discounted` | single-specification baselines |
