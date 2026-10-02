# Reproducing the results

Every number in the paper draft maps to one target of `scripts/reproduce.sh`. Set up first
([INSTALL.md](INSTALL.md)): environment, `source activate.sh`, checkpoints installed with
`scripts/fetch_bundles.sh --from <dir>`, `pytest -q tests/`.

```bash
bash scripts/reproduce.sh e092 e091 standing certificates figures   # the core of the paper, ~1.2 h
```

Outputs go to `repro/<E0XX-slug>/` (override with `ODD_ARTIFACTS`), logs to `repro/logs/`. Compare them
with the expected tables below. Runtimes are for one RTX 4070.

## How to read "expected"

- **No evaluation run is seeded.** A rerun draws fresh spawns, disturbances and gait phases. Expect each
  rate to land within about ±0.04 of the reference at N=256 (the binomial standard error is ≈0.03; the
  measured run-to-run sd over 4 reps is ≤0.04).
- **Every policy was trained once, with seed 0.** "4 reps" in the paper means 4 independent *evaluation*
  repetitions of the same policies, not 4 trained seeds.
- Rates are fractions of the original N=256 fleet. **safe** = never hit a death condition over the
  horizon (no respawn). **success** = reached the goal while alive.
- Arm names: `V2` = ODD-conditioned (the full system), `V1` = ODD-conditioned (direct) — the ablation
  without the certified return bridge, `V2-REUSE` = V2 with rest_hi flying the descent. Plus the anchors
  WALK-ONLY / STAND-ONLY (task expert alone), REST-ONLY (safety expert alone), ONE-WAY (descend, never
  return). Exact switching rules: [SWITCHING_LOGIC.md](SWITCHING_LOGIC.md).

---

## 1. Payload-swap walking — E092 (paper §3, the main walking result)

```bash
bash scripts/reproduce.sh e092            # ~6 min  -> repro/E092-payload-walk/
```

Walk 12 m; at t=5 s a tall crate (220 N, CoM 0.35 m) is loaded for 8 s, then removed; 40 s horizon; death
= flip-over (|tilt| > 80°). The goal is unreachable before the crate arrives (~4.2 m), so success measures
the ability to **resume**. Schedules: `pulse` (instant), `period` (sin² ramp), `dip` (ramp with a 1.6 s
false lightening).

Expected (success / safe, mean ± sd over 4 reps; ± omitted where sd ≤ 0.01):

| arm | pulse | period | dip |
|---|---|---|---|
| WALK-ONLY | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| REST-ONLY | 0.00 / 1.00 | 0.00 / 1.00 | 0.00 / 1.00 |
| ONE-WAY | 0.00 / 0.02 | 0.00 / 0.42 | 0.00 / 0.42±.01 |
| V1 (direct) | 0.03±.01 / 0.03±.01 | 0.37 / 0.37 | 0.33±.01 / 0.36±.01 |
| **V2 (ODD-conditioned)** | 0.03±.01 / 0.03±.01 | **0.38±.02 / 0.39±.02** | **0.40±.02 / 0.40±.02** |

What must reproduce: the structural anchors exactly (REST-ONLY 1.00, WALK-ONLY 0.00, ONE-WAY success 0);
V2 ≈ ONE-WAY safety on the ramps with ~0.4 success instead of 0; on `dip`, V2's success above V1's (by 0.07 on
average; the safety gap is smaller, 0.04 on average, and can vanish in a single run); on `period` V1 and V2 tie
(either can come out ahead in a single run); every walking arm ≈ 0 on `pulse` (the maneuver-latency boundary — an instant 220 N load flips a mid-stride robot
in ~0.3 s, faster than detect + brake + descend). Also produced: `claims_{pulse,period,dip}.png`
(survival + success-CDF), `topdown_*.png`, `traj_*.npz`. Deaths by automaton state are printed; on the
ramps most V2 deaths are in BRAKE (~45 %, the walk→stand handoff).

## 2. Leg-fault walking — E091 (paper §6)

```bash
bash scripts/reproduce.sh e091            # ~2 min  -> repro/E091-leg-walk/
```

Walk 11 m; at t=5 s the front-right leg's torque limit derates 1.0 → 0.15 over 3 s, stays derated until
t=15 s, then recovers; 30 s horizon; death = the env's own terminations (tilt > 70°, or a non-foot
ground contact above 500 N).
The descent trigger is the torque-saturation residual (the value is blind to this fault); the return gate
is the leg-healthy belief + `V_up`.

Expected (success / safe, the single reference run):

| WALK-ONLY | REST-ONLY | ONE-WAY | V1 (direct) | V2-REUSE (ODD-conditioned) |
|---|---|---|---|---|
| 0.20 / 0.20 | 0.00 / 1.00 | 0.06 / 0.25 | 0.15 / 0.19 | **0.28 / 0.29** |

What must reproduce: V2-REUSE has the highest success of any arm (it is the only one that both survives
the fault and resumes); ONE-WAY is safer than walking but never finishes; most switching-arm deaths are in
BRAKE. The draft quotes 0.26/0.26 vs 0.18/0.18 from an earlier run of the same configuration — within
run-to-run noise. This experiment has no 4-rep pass.

## 3. Weight-excursion walking — E089 (boundary finding, paper §6)

```bash
bash scripts/reproduce.sh e089            # ~15 min (est.)  -> repro/E089-goal-walk/
```

Same task with the load-naive walker carrying a 220 N load at h=0.25 during [5,13) s; goal 12 m; three
disturbance conditions. **This is a negative result:** every walking arm ends at safe ≤ 0.05 and
success ≤ 0.02 in every condition (REST-ONLY 1.00). Walking under this weight excursion is infeasible for
the naive walker, and settling under peak load from a gait flips robots. This is why the walking
demonstration moved to E092 (W=0 light phase, belief trigger, brake handoff, dedicated descent funnel).
(An earlier 9 m-goal version of this protocol looked positive only because robots finished before the load
arrived; this script runs the 12 m protocol.)

## 4. Standing automaton under load — E084 + E086 (paper §4)

```bash
bash scripts/reproduce.sh standing        # ~30 min (est., incl. videos)  -> repro/E084-automaton/
```

The robot stands under adversarial pushes while the carried load W (h=0.25) varies; no task beyond
standing; death = env terminations. E084 = repeating waves over 20 s (`square` 40↔220 N every 3 s, `sine`
130±110 N, 8 s period); E086 = one excursion over 24 s (`pulse` 40→220→40 N, `period` one cosine period).
`benign` = 10 N ambient push, `gusty` = 35 N gusts.

Expected safe rate (mean ± sd over 4 reps; ± omitted where sd ≤ 0.01):

| scenario | V2 | V1 | ONE-WAY | STAND-ONLY | REST-ONLY |
|---|---|---|---|---|---|
| single benign / period | **0.74±.01** | 0.21±.02 | 0.62±.01 | 0.04 | 1.00 |
| single benign / pulse | 0.49±.01 | 0.31±.01 | **0.54±.04** | 0.06 | 1.00 |
| single gusty / period | 0.22±.02 | 0.09 | 0.19±.02 | 0.02 | 1.00 |
| single gusty / pulse | 0.21±.02 | 0.12 | 0.23±.03 | 0.05 | 1.00 |
| waves benign / square | 0.46±.02 | 0.20±.02 | **0.56±.03** | 0.01 | 1.00 |
| waves benign / sine | 0.29 | 0.11±.02 | 0.29 | 0.00 | 1.00 |
| waves gusty / square | 0.21 | 0.09 | 0.23±.02 | 0.00 | 1.00 |
| waves gusty / sine | 0.10 | 0.02 | 0.06 | 0.00 | 1.00 |

What must reproduce: REST-ONLY 1.00 everywhere (settled rest is the one true safe set of this spec);
STAND-ONLY ≈ 0; V2 the best switching arm on smooth changes (period), ONE-WAY ahead on steps and square
waves (every return costs another descent, and descents are where benign-condition deaths happen); under
gusts, everything except REST bleeds while still standing. Also produced: `survival_*.png`
(regenerate with the `figures` target), `single_{pulse,period}.mp4`.

## 5. Evaluation repetitions — E094

```bash
bash scripts/reproduce.sh e092 standing seeds     # seeds alone ~1.5 h; needs the e092 + standing outputs as rep0
```

Runs 3 more repetitions of the E092, E084-waves and E086-single tables and aggregates mean ± sd with the
first run as rep0 → `repro/PAPER-draft/seeds/seeds_summary.md`. Compare with the mean ± sd tables of §1
and §4.

## 6. Figures — E093 (paper F2, F3, F7, claims, top-downs, survival)

```bash
bash scripts/reproduce.sh certificates e092 standing   # produce the data first (see §1, §4, §7)
bash scripts/reproduce.sh figures                      # ~1 min -> repro/E077-figures/, repro/PAPER-draft/figs/
```

Re-plots the polished figures from on-disk data; it does not simulate. If an input is missing, the runner
names the target that produces it.

| figure | data it reads | produced by |
|---|---|---|
| F2 certifiable regions | `E077-figures/F2_grid.json` | E079 (`certificates` target) |
| F3 mode-ribbon timeline | `E074-hicom-demo/task3_ramp.json` | E074_ramp (`certificates`) |
| F7 certificate contraction | `E074-hicom-demo/task2_value.json`, `E075-recal-eval/partA_value.json`, `E076-leg-demo/partB_value.json`, `E078-compound-demo/task2_value.json` | E074/E075/E076/E078 value sweeps (`certificates`) |
| claims / top-down (walking) | `E092-payload-walk/results.json`, `traj_*.npz` | `e092` |
| survival (standing) | `E084-automaton/results.json`, `results_single.json` | `standing` |
| compound timeline + demo video | — (renders its own rollout) | E078_video (`compound`) |

Figures F1, F4, F5, F6, F8 of an earlier draft (v2) have **no producing script in the repo**; they were made
ad hoc on 2026-08-22 and are not used by the current draft.

## 7. Certificate evidence — the value sweeps behind F2 / F3 / F7 (paper §1)

```bash
bash scripts/reproduce.sh certificates    # ~30 min (est.)
bash scripts/reproduce.sh figures         # then re-plot from your sweeps
```

| script | measures | expected |
|---|---|---|
| E079_region_sweep | STAND failure over (W × pull) and (leg θ × pull) at 80 N; the 20 %-failure contour = certifiable boundary | STAND is an island around the benign corner; at 35 N pull it ends near W≈90, at 5 N past W≈130 |
| E074_value / E075_value | `V_stand` vs W on operating states (original vs recalibrated stand_hi) | contracts with W; discrimination 0.97 → 1.16 after recalibration (E075); 1.21σ in F7 units |
| E076_value | `V_leg_stand` vs θ, unloaded | flat — the negative control (0.48σ): the reactive policy absorbs this axis, so the filter never fires |
| E078_value | compound `V_stand` vs θ while carrying 80 N | the strongest contraction (1.54σ) |
| E074_ramp | live 0→250 N ramp with gusts: family vs unified-spec value | the unified baseline's value *rises* (~0.87) as standing dies (the certification deficit); the family's crosses its trigger |

## 8. Compound demo — leg dies while loaded, E078 (paper §5)

```bash
bash scripts/reproduce.sh compound        # ~20 min (est.) -> repro/E078-compound-demo/
```

Fixed-θ matrix (N=128/cell): the stance expert keeps ≥ 0.95 stance fraction down to θ=0.3, then tips
49 % at θ=0.2 and 58 % at θ=0.1; the rest expert is tip-free across θ. The θ-ramp handoff switches at
θ≈0.24 (on the certifiable boundary) and cuts tips ~3× versus stand-only (0.20 vs 0.63) at 0.57
affordance; slams stay high (0.53 vs 0.57) — the known weak point. The video and
`compound_timeline.png` show one episode end to end. Demonstration-level evidence only: there is no
walking-protocol or multi-rep evaluation of this scenario.

## 9. Videos

```bash
bash scripts/reproduce.sh videos          # E092 single-best-run videos (period, dip) + E091 3-panel video
```

E092 `--videos` reads its own `results.json` for the fleet curves drawn under the video, so run `e092`
first. The standing videos are produced by the `standing` target (E086).

## 10. Retraining the policies

All 16 policies are reach-avoid PPO twins trained with the sandbox trainer; recipes, configs, warm-start
chains and wall-clock times are in [TRAINING.md](TRAINING.md). A single policy takes 15–70 min on a 4070.
Retrained policies will not reproduce the checkpoints byte-for-byte (GPU nondeterminism), and the runtime
thresholds in [SWITCHING_LOGIC.md](SWITCHING_LOGIC.md) were calibrated on these particular value nets —
recalibrate them (E089 `--cal`, E091 `--cal`) after retraining.

---

## Verification record

2026-10-02, fresh clone of the `cleanup` branch (submodules at the pinned commits, weights from the bundle,
the pinned `mjlab` env), RTX 4070 shared with another job:

| check | result |
|---|---|
| `pytest -q tests/` | 12 passed, 1 skipped (the E011 parity test needs grid data that is not bundled) |
| `e091` (98 s) | WALK-ONLY 0.17/0.17 · REST-ONLY 0.00/1.00 · ONE-WAY 0.06/0.26 · V1 0.18/0.22 · **V2-REUSE 0.27/0.27** |
| `e092` (351 s) pulse | WALK-ONLY 0/0 · REST-ONLY 0/1.00 · ONE-WAY 0/0.02 · V1 0.02/0.02 · V2 0.02/0.02 |
| `e092` period | WALK-ONLY 0/0 · REST-ONLY 0/1.00 · ONE-WAY 0/0.41 · V1 0.38/0.38 · V2 0.35/0.36 |
| `e092` dip | WALK-ONLY 0/0 · REST-ONLY 0/1.00 · ONE-WAY 0/0.41 · V1 0.32/0.37 · **V2 0.37/0.37** |
| `figures` (11 s) | F2/F3/F7 regenerated from data (certificate sweeps taken from the maintainer's runs); claims/top-down rebuilt from the fresh E092 run |

All within ±0.04 of the reference. Peak GPU memory of an N=256 evaluation: ~0.6 GB.
