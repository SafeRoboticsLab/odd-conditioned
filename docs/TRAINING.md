# Training the policies

Every safety policy is a **two-player reach-avoid PPO twin** (`ReachAvoidPPO2P`, safety-stable-baselines
v0.4.0): a control player against a learned adversarial base force, with a state-value net whose sign is the
mode's certificate. All of them share one recipe and differ only in task, warm-start and env overrides. The
tasks, margins and training configs live on the robot-safety-sandbox branch `project/odd-conditioned` (the
`external/robot-safety-sandbox` submodule). The nominal walker was not trained in this project (§3).

## 0. Train from scratch

```bash
source activate.sh
bash scripts/train.sh core            # rest, stand, getup_stage1, getup, descend — the automaton     ~2 h
bash scripts/train.sh certificates    # stand_wide, unified, unified_discounted, leg_stand,
                                      # compound_stand, compound_rest — the certificate figures       ~2.5 h
```

`train.sh` runs the policies one after another in dependency order and trains each into `runs/<name>/`
(log: `runs/logs/<name>.log`). When a run finishes it **installs** the final-step checkpoint into
`checkpoints/<name>/` (`model.zip`, `tensornormalize.pt`, `config.yaml`), which is where every evaluation
looks. It never replaces an installed checkpoint or an existing run directory unless `FORCE=1`. Keep a
retrain apart from the published weights by pointing it elsewhere:

```bash
ODD_CHECKPOINTS=checkpoints_retrained bash scripts/train.sh core certificates
export ODD_CHECKPOINTS=checkpoints_retrained       # then every script below uses the retrained set
```

Knobs: `SEED=<k>` (default 0, the published runs), `ODD_WANDB=1` (log to wandb; default off), `SMOKE=1`
(300k steps per policy in a scratch directory — checks the commands, not a usable policy).

After training:

1. **Accept** the stance-type policies (§5): `python scripts/check_policies.py stand checkpoints_retrained/stand/model.zip`
   and `... compound checkpoints_retrained/compound_stand/model.zip`.
2. **Calibrate** the switching thresholds (§6): `python scripts/calibrate.py`, then set the recommended values.
3. **Reproduce**: `bash scripts/reproduce.sh all` ([REPRODUCE.md](REPRODUCE.md)).

## 1. Command and recipe

`train.sh` wraps the sandbox trainer:

```bash
python external/robot-safety-sandbox/examples/train.py \
    --config external/robot-safety-sandbox/configs/<CONFIG>.yaml --out runs/<name> [--env-override K=V] [--load ...]
```

Shared recipe (every config): `num_envs 1024`, `steps 50M` (actual 50,036,736 = 1018 iterations × 1024 envs ×
48 steps), `seed 0`, fixed `lr 1e-4`, `target_kl 0.01`, `ent_coef 1e-3`, `max_std 0.4`, `net 128,128,128`,
`vf_coef 0.5`, `gamma 0.99`, `terminal_type all` (terminal valued `min(l, g)`), adversary `force_max 25 N` ramped
up over the first 55 % of training with a per-env survival force scale (floor 0.3, init 0.5), `dstb_pretrain 20`,
`norm_freeze_steps 5M` (active only on warm-started runs). Checkpoints every 25M steps
(`checkpoints/model_{24999936,49999872}_steps.zip` + step-matched `tensornorm_*.pt`), plus `final_model.zip` +
`tensornormalize.pt`. The evaluations use the **49,999,872-step checkpoint**; warm starts load the source's
**final** model. Each installed `config.yaml` records every flag of the run.

## 2. The policies

Wall-clock on one RTX 4070 (shared GPU for some). "Final" = last training-log failure rate (each env against
its own adversary — not comparable across runs) / adversary force scale / value explained variance.

| name | role | task | config | extra flags | wall-clock | final failure / force / EV |
|---|---|---|---|---|---|---|
| **stand** | STAND expert; `V_stand` | `go2_weight_stand_hi` | `go2_weight_stand_hi_ppo.yaml` | `--env-override hi=120` | 22 min | 0.524 / 0.515 / 0.586 |
| **rest** | REST expert; descent in `odd-rest-descent` / `direct` / `one-way` | `go2_weight_rest_hi` | `go2_weight_rest_hi_ppo.yaml` | — | 56 min | 0.000 / 1.00 / 0.606 |
| getup_stage1 | warm-start source of `getup` (final model only) | `go2_getup` | `go2_getup_ppo.yaml` | — | 37 min | 0.268 / 0.936 / 0.62 |
| **getup** | get-up funnel; `V_up` | `go2_getup` | `go2_getup_ppo.yaml` | `--load getup_stage1/final` | 17 min | 0.424 / 0.714 / 0.48 |
| **descend** | descent funnel (`odd`) | `go2_descend` | `go2_descend_ppo.yaml` | `--load rest/final` | 14 min | 0.000 / 0.998 / −0.805 |
| stand_wide | STAND expert on W ∈ [0, 150] (wide-stand demo, certification deficit) | `go2_weight_stand_hi` | `go2_weight_stand_hi_ppo.yaml` | — | 70 min | 0.779 / 0.538 / 0.686 |
| unified | single-spec baseline | `go2_weight_unified_hi` | `go2_weight_unified_hi_ppo.yaml` | — | 66 min | 0.000 / 1.00 / 0.865 |
| unified_discounted | single-spec baseline, prefers standing | `go2_weight_unified_disc_hi` | `go2_weight_unified_disc_hi_ppo.yaml` | — | 67 min | 0.049 / 0.995 / 0.684 |
| leg_stand | STAND expert, derated leg (negative control) | `go2_leg_stand` | `go2_leg_stand_ppo.yaml` | — | 37 min | 0.368 / 0.755 / 0.65 |
| compound_stand | STAND expert, derated leg while carrying 80 N | `go2_compound_stand` | `go2_compound_stand_ppo.yaml` | — | 36 min | 0.149 / 0.694 / 0.756 |
| compound_rest | REST expert, compound case | `go2_compound_rest` | `go2_compound_rest_ppo.yaml` | — | 33 min | 0.000 / 1.00 / 0.685 |

Notes:

- **stand vs stand_wide.** The same task trained on W ∈ [0, 150] N (stand_wide) — most of which is infeasible
  with the raised-CoM load (0.78 training failure) — gives a value net too noisy for a precise trigger.
  Training on the comfortably feasible band W ∈ [0, 120] (stand) gives a calibrated certificate near the edge
  (discrimination 0.97 → 1.16). The same feasible-band principle sets the training ranges of leg_stand
  (θ ∈ [0.5, 1]), compound_stand (θ ∈ [0.4, 1]) and getup (W ∈ [0, 160]).
- **getup is two-stage.** Stage 1 trains from prone spawns; stage 2 continues from its final model on the
  widened prone-pose ranges measured on the rest poses the automaton actually produces, so that `V_up` reads
  correctly where it is used.
- **descend** warm-starts from the REST expert and is trained from disturbed-standing spawns under heavy loads.
  Its explained variance (−0.805) is uninformative — with ~0 failures the returns are nearly constant — so its
  value is not used as a gate; only its policy is.

## 3. The nominal walker

The walking scenarios use a **load-naive joystick walker that was not trained in this project**: sandbox task
`go2_walker_flat` (plain reward PPO, no adversary, no margins), 4096 envs, 150M steps, `net 512,256,128`,
seed 0, trained at W = 0 (`external/robot-safety-sandbox/configs/go2_walker_flat.yaml`). Its actor and
observation normaliser were extracted into `go2_atomic_skills` (`walker_actor.pt`, `walker_norm.npz`), which
`policies.Walker` loads and drives with a rebuilt 47-d observation. It walks ~0.84 m/s and is reliable only
unloaded.

## 4. What the policies are trained on

Base env (every mode): mjlab `unitree_go2_flat` with the velocity command pinned to zero, 50 Hz control, 20 s
episodes (evaluations remove the timeout), terminations `time_out`, `fell_over` (tilt > 70°) and
`illegal_contact`. The adversary is a learned 3-D base force up to 25 N. Actor observation = 47-d
proprioception + 1 ODD scalar = **48-d**; the value is read on the same observation.

**Carried load (weight ladder).** W newtons applied as an extra base wrench `[0, 0, −W]`, added to the
adversary's force. For every task here the load sits at body-frame height `h = 0.25 m`, adding the
inverted-pendulum torque `τ = R(q)[0,0,h] × [0,0,−W]` (|τ| = W·h·sin(tilt)), which pushes further into any tilt.
Training samples W per episode; evaluations drive it per step (`sim.set_load`). ODD observation `W/150`.

**Leg degradation.** θ scales the torque limits of the three front-right actuators (`actuator_forcerange`,
written absolutely from the nominal 23.5 / 23.5 / 45 Nm). Observation: θ.

**Margins** (g = safety, fail if < 0; l = target, reached if ≥ 0):

| mode / task | g | l | ODD range in training | illegal-contact termination |
|---|---|---|---|---|
| STAND (`go2_weight_stand_hi`, `go2_leg_stand`, `go2_compound_stand`) | lowest trunk-corner height − 0.10 m | corners in [0.10, 0.40] m, \|v\| < 0.20, \|ω\| < 0.174 | W ∈ [0,150] (stand: [0,120]); leg θ ∈ [0.5,1]; compound θ ∈ [0.4,1] at W=80 | 10 N |
| REST (`go2_weight_rest_hi`, `go2_compound_rest`) | (80 + 1.3·W − max non-foot contact force) / 80 — a load-scaled no-slam cap | low (z < 0.15), level (tilt < 0.25), settled (\|v\| < 0.30, \|ω\| < 0.50) | W ∈ [0,300]; compound θ ∈ [0,1] at W=80 | 500 N |
| REST, leg (`go2_leg_rest`, evaluation surface of the leg sweep) | (80 − max non-foot contact) / 80 | same | θ ∈ [0,1] incl. dead leg | 500 N |
| unified baselines | = REST g | max(l_stand, l_rest − δ), δ = 0 / 0.5 | W ∈ [0,300] | 500 N |
| get-up funnel (`go2_getup`) | = REST g | = STAND l (reach and hold a stand) | W ∈ [0,160]; prone spawns | 500 N |
| descent funnel (`go2_descend`) | = REST g | = REST l | W ∈ [80,260]; disturbed standing spawns | 500 N |

Every automaton evaluation runs in `go2_weight_rest_hi_at_0` with W driven per step, so a death there is
`fell_over` (70°, or 80° under the payload walk's flip-over criterion) or a non-foot contact above 500 N.

### The two transition funnels

The automaton's transitions are policies in their own right — the robot does not switch from the stand expert
to the rest expert and hope. Each funnel is the same kind of reach-avoid twin as the mode experts (same recipe,
same adversary, observes W); it differs only in **where its training episodes start** and **which target it
must reach**. The safety condition g is the rest mode's load-scaled no-slam rule for both. Code: sandbox
`robot_safety_sandbox/envs/go2_transitions/env_cfg.py`.

| | get-up funnel (`go2_getup` → getup) | descent funnel (`go2_descend` → descend) |
|---|---|---|
| job | lying down → stable stand | standing → settled rest |
| episodes start | lying down: base 0.09–0.13 m, roll/pitch ±0.15 rad, any heading, legs folded (thigh 0.3–1.8, calf −2.75 to −1.85, hip ±0.9 rad — the ranges measured on the rest poses the automaton produces) | standing but disturbed: roll/pitch ±0.15 rad, lateral velocity ±0.3 m/s |
| loads | W ∈ [0, 160] N, CoM 0.25 m up | W ∈ [80, 260] N (heavy only), CoM 0.25 m up |
| target l | the stance target, which it must **reach and hold** (episodes end only on failure) | the rest target (low, level, settled) |
| warm-start | getup_stage1 | the REST expert |
| its value in the automaton | `V_up`: the return certificate and the abort signal | not used |

Two consequences shape the automaton:

- **A funnel is trustworthy only from inside its training start set.** The descent funnel never saw a walking
  gait, so walking first BRAKEs and only then descends. The get-up funnel never saw loads above 160 N, so the
  return waits for the load to clear.
- **The descent funnel pays for itself** standing (compare `odd` with `odd-rest-descent` in
  [FINDINGS.md](FINDINGS.md#5-results)).

## 5. Training variance and acceptance

**The stance-type policies (stand, compound_stand) are high-variance across training runs.** Each published
policy is a single training run (seed 0). Retraining with the published recipe and other seeds gave much weaker
stance experts: of five runs of the `stand` recipe, only the published one passes the acceptance test below
(the others reach 0.09–0.48 on its walking check, against 0.74 for the published one), and one retrain of
compound_stand collapsed (adversary pinned at its floor, training failure 1.0). The rest, get-up and descent
funnels and the baselines retrain reliably. A weak stand expert shows up mostly as deaths in the BRAKE mode of
the payload walk, which it flies: braking a walking robot is not part of its training distribution.

So, when retraining: train a few seeds of `stand` (and `compound_stand`) and keep one that passes —

```bash
for s in 1 2 3; do SEED=$s ODD_CHECKPOINTS=ck_seed$s bash scripts/train.sh stand; done
python scripts/check_policies.py stand ck_seed*/stand/model.zip
python scripts/check_policies.py compound checkpoints_retrained/compound_stand/model.zip
```

| test | what it runs | pass | published policy |
|---|---|---|---|
| `stand` T1 | the candidate alone, standing under constant 40 N and 100 N (CoM 0.25 m), 20 s, no push | survival at 40 N ≥ 0.90 | 1.00 / 0.67 |
| `stand` T2 | payload walk (period, no push) with the candidate as STAND expert, the installed rest / getup / descend | ODD-conditioned success ≥ 0.60 | 0.74 (28 brake deaths) |
| `compound` | compound_stand at fixed θ ∈ {1.0, 0.6, 0.4} while carrying 80 N, 10 N push + one 35 N gust | stance ≥ 0.90 and tip ≤ 0.10 at each θ | stance 0.98–0.99, tip 0.02–0.05 |

`scripts/train_curves.py runs/logs/stand.log ...` prints training curves side by side to see where runs diverge.

## 6. Calibrating the switching thresholds

The automaton compares learned values (and one torque residual) against constants in `scenarios.py`,
`automaton.py` and `certificates.py`. They were placed on the published value networks at specific operating
points; a retrained network — or a changed margin function — moves the value scale, so the thresholds must be
re-placed at the same operating points. One script does it:

```bash
python scripts/calibrate.py                                  # all thresholds, 3 pooled seeds, ~7 min
python scripts/calibrate.py --only standing payload --reps 1 # a subset / a quick look
```

It runs every probe on the installed checkpoints and prints, per constant, where it lives, its current value,
the recommended value and the evidence (percentiles of the measured distribution, and how often the current
value fires or passes). It edits nothing. Pool seeds (`--reps 3`): the operating points sit in distribution
tails, and one run moves them by a few hundredths.

**Check of the rules:** on the published checkpoints the script reproduces the published constants (3 pooled seeds) —

| constant | published | recommended on published nets | rule (operating point) |
|---|---|---|---|
| standing `trigger_eps` | −0.05 | −0.036 | 7th percentile of V̄_stand, robots standing under 40 N |
| standing `eps_up` | +0.10 | +0.102 | 94th percentile of V̄_up on settled REST after the load clears |
| standing `eps_abort` | −0.02 | −0.018 | `eps_up` − 0.12 |
| `DIRECT_EPS_UP` | 0.15 | 0.150 | 21st percentile of V̄_stand on the same settled-REST robots |
| weight-walk `trigger_eps` | −0.15 | −0.125 | highest value with ≤ 1.6 % false descents per 30 s of healthy walking; the published value is more conservative (0.6 % false descents) |
| leg-fault `trigger_eps` | 0.03 | 0.03 | keep while between the healthy band (p99 ≈ 0.000) and the derated band (median ≈ 0.014); depends on the walker and physics, not on the safety networks |
| leg-fault `eps_up` / `eps_abort` | −0.35 / −0.45 | −0.339 / −0.439 | 12th percentile of V̄_up on settled REST after the leg heals; abort 0.10 lower |
| payload `eps_up` / `eps_abort` | −0.30 / −0.45 | −0.297 / −0.447 | 35th percentile of V̄_up on settled REST after the load clears; abort 0.15 lower |
| compound ramp `eps` | −0.14 | −0.141 | midpoint of the certifiable- and failed-band value means (compound value sweep) |
| weight ramps `eps` | −0.04 / −0.05 | reported only | inside the window where a forced switch is mechanically safe (`certificates/forced_switch.json`); the script reports the value-sweep midpoint and discrimination — a discrimination below ~0.5 means the certificate is flat (the unloaded-leg control reads ~0.0) and the policy should be retrained |

Not tied to the networks (no recalibration needed): the payload trigger W ≥ 60 N (the edge of the walker's
comfortable load), the return gate W < 130 N (the measured stand-feasibility boundary), and the timing
constants (EMA α, arming, refractory, sustain counts).

After calibrating, rerun `bash scripts/reproduce.sh payload leg standing` and compare with
[REPRODUCE.md](REPRODUCE.md): the structural results (REST-ONLY 1.00, WALK-ONLY 0.00, ODD-conditioned ≈ ONE-WAY
safety with clearly positive success on the payload ramp) should hold; the exact rates will move.
