# Training the policies

Every safety policy in the paper is a **two-player reach-avoid PPO twin** (`ReachAvoidPPO2P`, safety_sb3
v0.4.0): a control player against a learned adversarial base force, with a state-value net whose sign is the
mode's certificate. All 16 share one recipe and differ only in task, output directory, warm-start and env
overrides. The nominal walker was not trained in this project (§3).

> **Provenance.** No training launch command was written down at the time. The commands below are
> reconstructed from each run's training-log header plus its resolved `config.yaml` (shipped in the weights
> bundle next to every checkpoint), which records every flag and is the authoritative record. The sandbox
> env code was committed only after training (2026-10-02, sandbox `a3c14f2`), at its final state: the
> abandoned getup v1 and descend v1/v3 spawn distributions cannot be rebuilt from it.

## 1. Command

From the repo root, after `source activate.sh`:

```bash
python external/robot-safety-sandbox/examples/train.py \
    --config external/robot-safety-sandbox/configs/<CONFIG>.yaml [EXTRA FLAGS]
```

Extra flags used: `--out <dir>`, `--load <warm-start .zip>`, `--env-override KEY=VAL`, `--no-wandb`. The
config's `out:` is relative to the working directory, so runs land in `results/...` at the repo root.
Logging goes to wandb project `odd-conditioned` unless `--no-wandb`; checkpoints never leave the machine.

Shared recipe (from every `config.yaml`): `num_envs 1024`, `steps 50M` (actual 50,036,736 = 1018 iterations ×
1024 envs × 48 steps), **`seed 0` for every policy**, fixed `lr 1e-4`, `target_kl 0.01`, `ent_coef 1e-3`,
`max_std 0.4`, `net 128,128,128`, `vf_coef 0.5`, `gamma 0.99`, `terminal_type all` (terminal valued
`min(l, g)`), adversary `force_max 25 N` ramped up over the first 55 % of training with a per-env survival
force scale (floor 0.3, init 0.5), `dstb_pretrain 20`, `norm_freeze_steps 5M` (only active on warm-started
runs). Checkpoints every 25M: `checkpoints/model_{24999936,49999872}_steps.zip` + `tensornorm_*.pt`, plus
`final_model.zip` + `tensornormalize.pt`.

**The evaluations load `checkpoints/model_49999872_steps.zip`** (with its step-matched
`tensornorm_50001920.pt`, found automatically by `load_twin`), not `final_model.zip`. The two warm-started
runs loaded their source's `final_model.zip` + `tensornormalize.pt`; both are in the weights bundle.

## 2. The 16 policies

Wall-clock depends on how many runs shared the GPU (RTX 4070; up to 4 concurrent). "Final" = the last
training-log values of failure rate (each env against its own adversary — not comparable across runs),
adversary force scale, and value explained variance.

| name in scripts | run dir (`results/...`) | task | config | extra flags | trained in | wall-clock | final failure / force / EV | wandb |
|---|---|---|---|---|---|---|---|---|
| **stand** (stand_hi recal) — the STAND expert, `V_stand` | `go2_weight_runs/E075_recal/go2_weight_stand_hi_adv` | `go2_weight_stand_hi` | `go2_weight_stand_hi_ppo.yaml` | `--env-override hi=120 --out results/go2_weight_runs/E075_recal` | E075 | 22 min | 0.524 / 0.515 / 0.586 | 6ridkfd4 |
| **rest** (rest_hi) — the REST expert; V2-REUSE / V1 descent | `go2_weight_runs/go2_weight_rest_hi_adv` | `go2_weight_rest_hi` | `go2_weight_rest_hi_ppo.yaml` | — | E073 | 56 min | 0.000 / 1.00 / 0.606 | 6bxvi2ia |
| **getup** (getup_v2) — return funnel, `V_up` | `go2_transition_runs/getup_v2/go2_getup_adv` | `go2_getup` | `go2_getup_ppo.yaml` | `--load results/go2_transition_runs/go2_getup_adv/final_model.zip --out results/go2_transition_runs/getup_v2` | E087 | 17 min | 0.424 / 0.714 / 0.48 | eowlotzi |
| **descend** (descend_v4) — descent funnel (arm V2) | `go2_transition_runs/descend_v4/go2_descend_adv` | `go2_descend` | `go2_descend_ppo.yaml` | `--load results/go2_weight_runs/go2_weight_rest_hi_adv/final_model.zip --out results/go2_transition_runs/descend_v4` | E083 | 14 min | 0.000 / 0.998 / −0.805 | eofapm4v |
| stand_hi (original, W∈[0,150]) | `go2_weight_runs/go2_weight_stand_hi_adv` | `go2_weight_stand_hi` | `go2_weight_stand_hi_ppo.yaml` | — | E073 | 70 min | 0.779 / 0.538 / 0.686 | xkbqu7w1 |
| unified_hi (baseline) | `go2_weight_runs/go2_weight_unified_hi_adv` | `go2_weight_unified_hi` | `go2_weight_unified_hi_ppo.yaml` | — | E073 | 66 min | 0.000 / 1.00 / 0.865 | k5pzvk7t |
| unified_disc_hi (baseline) | `go2_weight_runs/go2_weight_unified_disc_hi_adv` | `go2_weight_unified_disc_hi` | `go2_weight_unified_disc_hi_ppo.yaml` | — | E073 | 67 min | 0.049 / 0.995 / 0.684 | nwak8k8m |
| stand (flat load, iteration 1) | `go2_weight_runs/go2_weight_stand_adv` | `go2_weight_stand` | `go2_weight_stand_ppo.yaml` | — | E070 | 32 min | 0.191 / 0.953 / 0.211 | t9vceryg |
| rest (flat load, iteration 1) | `go2_weight_runs/go2_weight_rest_adv` | `go2_weight_rest` | `go2_weight_rest_ppo.yaml` | — | E070 | 26 min | 0.000 / 1.000 / 0.601 | ku50xklf |
| getup v1 (warm-start source) | `go2_transition_runs/go2_getup_adv` | `go2_getup` (original narrow prone spawns) | `go2_getup_ppo.yaml` | — | E083 | 37 min | 0.268 / 0.936 / 0.62 | sa41tzib |
| descend v1 (dead, spawn bug) | `go2_transition_runs/go2_descend_adv` | `go2_descend` | `go2_descend_ppo.yaml` | — | E083 | 40 min | 1.000 / 0.30 / 0.544 | yp3wwh3g |
| descend v3 (dead, spawn bug) | `go2_transition_runs/descend_v3/go2_descend_adv` | `go2_descend` | `go2_descend_ppo.yaml` | `--env-override lo=0 --env-override hi=300 --out results/go2_transition_runs/descend_v3` | E083 | 17 min | 1.000 / 0.30 / 0.555 | gpao0r5f |
| compound_stand | `go2_compound_runs/go2_compound_stand_adv` | `go2_compound_stand` | `go2_compound_stand_ppo.yaml` | — | E078 | 36 min | 0.149 / 0.694 / 0.756 | jy2yw54y |
| compound_rest | `go2_compound_runs/go2_compound_rest_adv` | `go2_compound_rest` | `go2_compound_rest_ppo.yaml` | — | E078 | 33 min | 0.000 / 1.00 / 0.685 | fqklq89q |
| leg_stand | `go2_leg_family_runs/go2_leg_stand_adv` | `go2_leg_stand` | `go2_leg_stand_ppo.yaml` | — | E076 | 37 min | 0.368 / 0.755 / 0.65 | byjm1sra |
| leg_rest | `go2_leg_family_runs/go2_leg_rest_adv` | `go2_leg_rest` | `go2_leg_rest_ppo.yaml` | — | E076 | 35 min | 0.048 / 1.00 / 0.90 | 3zekk4n3 |

Notes:

- **"recal" (E075)**: the original stand_hi was trained on W ∈ [0,150] N, most of which is infeasible with the
  raised-CoM load (0.78 training failure), and its value net was too noisy for a precise trigger. Retraining on
  the comfortably feasible band W ∈ [0,120] gave a calibrated certificate near the edge (discrimination 0.97 →
  1.16). The same feasible-band principle was then applied from the start to leg_stand (θ ∈ [0.5,1]),
  compound_stand (θ ∈ [0.4,1]) and getup (W ∈ [0,160]).
- **getup_v2 (E087)** was retrained because v1's certificate `V_up` read pessimistically on the rest poses the
  automaton actually produces (v1 trained from hand-designed prone spawns with hips ±0.2; measured settled
  poses have hips ±0.75–0.9). The prone-spawn ranges were widened to the measured ones and v2 warm-started from
  v1.
- **descend v1/v3** failed because their spawn event read a stale base position and dropped every robot from
  z=0.445 m under load. v4 replaced it with a disturbed-standing spawn and warm-started from rest_hi. Its final
  explained variance (−0.805) is uninformative — with ~0 failures the returns are nearly constant — but keep it
  in mind before using `V_descend` as a gate.

## 3. The nominal walker

The walking experiments (E089–E092) use a **weight-naive joystick walker that was not trained in this
project**: sandbox task `go2_walker_flat` (plain reward PPO from stock SB3, no adversary, no margins), 4096 envs,
150M steps, `net 512,256,128`, seed 0, trained at W=0. Recipe: `external/robot-safety-sandbox/configs/go2_walker_flat.yaml`.
Its actor and observation normaliser were extracted into `go2_atomic_skills`
(`assets/extracted/walker_actor.pt`, `walker_norm.npz`; extraction script `build/extract_weights.py`), which the
experiments load through `experiments/E089_goal_walk.py::Walker` (a batched port of `go2_atomic_skills.WalkSkill`,
rebuilding the 47-d observation). It walks ~0.84 m/s in the weight env and is only reliable unloaded.

## 4. What the policies are trained on

Base env (`go2_stabilize`, every mode): mjlab `unitree_go2_flat` with the velocity command pinned to zero,
50 Hz control, 20 s episodes (evaluations override this to effectively infinite), terminations `time_out`,
`fell_over` (tilt > 70°) and `illegal_contact`. The adversary is a learned 3-D base force up to 25 N. Actor
observation = 47-d proprioception + 1 ODD scalar = **48-d**; the value is read on that same 48-d observation.

**Carried load (weight ladder).** W newtons applied as an extra base wrench `[0, 0, −W]`, added to the
adversary's force (sandbox `base.py`). For the `*_hi`, get-up, descent and compound tasks the load sits at
body-frame height `h = 0.25 m`, adding the inverted-pendulum torque `τ = R(q)[0,0,h] × [0,0,−W]`
(|τ| = W·h·sin(tilt)), which pushes further into any tilt. Training samples W per episode; evaluations drive it
per step through `env.base_load` (plus `env.mj._weight_W` / `_weight_h`). ODD observation `W/150`.

**Leg degradation.** θ scales the allowable torque of the three front-right actuators
(`actuator_forcerange`, absolute write from the nominal 23.5 / 23.5 / 45 Nm). Observation: θ.

**Margins** (g = safety, fail if < 0; l = target, reached if ≥ 0):

| mode / task | g | l | ODD range in training | illegal-contact termination |
|---|---|---|---|---|
| STAND (`go2_weight_stand(_hi)`, `go2_leg_stand`, `go2_compound_stand`) | lowest trunk-corner height − 0.10 m | corners in [0.10, 0.40] m, |v| < 0.20, |ω| < 0.174 | W ∈ [0,150] (recal [0,120]); leg θ ∈ [0.5,1]; compound θ ∈ [0.4,1] at W=80 | 10 N |
| REST (`go2_weight_rest(_hi)`, `go2_compound_rest`) | (80 + 1.3·W − max non-foot contact force) / 80 — a load-scaled no-slam cap | low (z < 0.15), level (tilt < 0.25), settled (|v| < 0.30, |ω| < 0.50) | W ∈ [0,300]; compound θ ∈ [0,1] at W=80 | 500 N |
| REST, leg (`go2_leg_rest`) | (80 − max non-foot contact) / 80 | same | θ ∈ [0,1] incl. dead leg | 500 N |
| unified baselines | = REST g | max(l_stand, l_rest − δ), δ = 0 / 0.5 | W ∈ [0,300] | 500 N |
| GET-UP funnel (`go2_getup`) | = REST g | = STAND l (reach and hold a stand) | W ∈ [0,160]; prone spawns | 500 N |
| DESCENT funnel (`go2_descend`) | = REST g | = REST l | W ∈ [80,260]; disturbed standing spawns | 500 N |

The automaton evaluations all run in the `go2_weight_rest_hi_at_0` env with W driven per step, so a "death" there
is `fell_over` (70°, or 80° under E092's loose criterion) or a non-foot contact above a flat 500 N.

## 5. Which experiment loads which checkpoint

Every path ends in `/checkpoints/model_49999872_steps.zip`. `CK` is the dict in `experiments/E084_automaton.py`
(stand = E075 recal stand_hi, rest = rest_hi, getup = getup_v2, descend = descend_v4).

| script | checkpoints |
|---|---|
| E070_matrix, E071_handoff, E072_video | flat-load stand / rest |
| E073_hicom_probe | flat-load stand (zero-shot on high-CoM physics) |
| E074_matrix, E074_ramp, E074_video | stand_hi, rest_hi, unified_hi, unified_disc_hi |
| E074_tune / E074_value | stand_hi, rest_hi, unified_hi / stand_hi, unified_hi |
| E075_ramp / E075_value | E075 recal stand_hi + rest_hi / recal + original stand_hi |
| E076_matrix, E076_video / E076_value | leg_stand, leg_rest / leg_stand |
| E078_gate | E075 recal stand_hi (zero-shot on compound physics) |
| E078_matrix, E078_ramp, E078_video / E078_value | compound_stand, compound_rest / compound_stand |
| E079_region_sweep | E075 recal stand_hi, rest_hi, compound_stand, compound_rest |
| E080_bidirectional, E080_video, E081_survival | E075 recal stand_hi + rest_hi |
| E084_automaton, E085_video, E086_single_pulse, E094_seeds | CK |
| E084b_early | CK but with getup **v1** |
| E089_goal_walk, E090_walk_viz, E091_leg_walk, E092_payload_walk | CK + the walker |
| E077_figure, E078_figure, E093_figpolish | none (re-plot data) |

Earlier experiments (E040–E069) load the payload and weak-leg runs (`results/go2_payload_runs/`,
`results/go2_weak_leg_runs/`, ~4 GB); those checkpoints are not in the bundle — ask if you need them.
