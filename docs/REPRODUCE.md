# Reproducing the results

Every reported number maps to one target of `scripts/reproduce.sh`. Set up first ([INSTALL.md](INSTALL.md)):
environment, `source activate.sh`, checkpoints installed with `scripts/fetch_weights.sh`, `pytest -q tests/`.

```bash
bash scripts/reproduce.sh all        # every target in dependency order, ~3.5 h on one RTX 4070
bash scripts/reproduce.sh payload leg standing certificates figures   # the core, ~1.1 h
```

Results land in `outputs/` (override with `ODD_OUTPUTS`): one directory per target with its JSON (and
trajectories), `outputs/figures/`, `outputs/videos/`, and logs in `outputs/logs/`. Single targets can also be
run directly: `python scripts/evaluate.py <target> [--seed K] [--n N]`.

## How to read "expected"

- **Rollouts are seeded** (seed 0 by default; the `seeds` target adds seeds 1–3). On the same machine a rerun
  gives the same numbers up to rare GPU floating-point nondeterminism: contact-rich steps occasionally differ
  in the last bit, which then changes the fate of individual robots. Across machines, or between seeds, expect
  each rate within about ±0.04 at N = 256 (binomial standard error ≈ 0.03; the measured sd across seeds is in
  the tables).
- **safe** = fraction of the original N = 256 robots that never hit a death condition over the horizon (no
  respawn, no timeout). **success** = reached the goal while alive.
- **Methods** (exact rules: [SWITCHING_LOGIC.md](SWITCHING_LOGIC.md)): **ODD-conditioned** (`odd`, the full
  system), **ODD-conditioned (direct)** (`direct`, the ablation without the certified return), **ONE-WAY**
  (descend, never return), **WALK-ONLY / STAND-ONLY** (`task-only`: the task policy alone), **REST-ONLY**
  (`rest-only`: the REST expert alone), and, standing only, **ODD-conditioned (rest descent)**
  (`odd-rest-descent`: the REST expert flies the descent instead of the funnel).
- **Every policy is one training run** (seed 0). The seeds below are evaluation seeds. Retrained policies give
  different numbers — see [TRAINING.md §5](TRAINING.md#5-training-variance-and-acceptance).

---

## 1. Payload-swap walking (`payload`, ~6 min; `seeds` adds seeds 1–3)

Walk 12 m; at t = 5 s a tall crate (220 N, centre of mass 0.35 m) is loaded for 8 s, then removed; 40 s
horizon; death = flip-over (tilt > 80°). The goal is unreachable before the crate arrives (~4 m), so success
measures the ability to **resume**. Profiles: `pulse` (instant load), `period` (sin² ramp), `dip` (the ramp
with a ~1.6 s false lightening mid-window). Outputs: `outputs/payload/` (results, trajectories), figures
`payload_claims_*.png` (survival and goal-completion curves) and `payload_topdown_*.png`.

Expected (success / safe, mean ± sd over seeds 0–3; ± omitted where sd < 0.005):

| method | pulse | period | dip |
|---|---|---|---|
| WALK-ONLY | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| REST-ONLY | 0.00 / 1.00 | 0.00 / 1.00 | 0.00 / 1.00 |
| ONE-WAY | 0.00 / 0.02±0.01 | 0.00 / 0.42±0.03 | 0.00 / 0.42±0.02 |
| ODD-conditioned (direct) | 0.02 / 0.02±0.01 | 0.39±0.03 / 0.40±0.03 | 0.38±0.03 / 0.41±0.03 |
| **ODD-conditioned** | 0.02±0.01 / 0.02±0.01 | **0.43±0.02 / 0.43±0.02** | **0.43±0.03 / 0.43±0.03** |

What must reproduce: the anchors exactly (REST-ONLY 1.00, WALK-ONLY 0.00, ONE-WAY success 0); ODD-conditioned
keeps ONE-WAY's safety on the ramps while turning zero success into ~0.4; every survivor of ODD-conditioned
reaches the goal (success ≈ safe), while the direct ablation leaves some survivors lying short of it on the
dip. Within a seed all methods share initial conditions, so compare them **paired by seed**: ODD-conditioned
exceeds the direct ablation in every seed, by 0.03–0.04 success on `period` and by 0.04–0.07 success
(0.02–0.03 safe) on `dip`. On `pulse` every walking method is ≈ 0: an instantly applied 220 N tall load flips a
mid-stride robot in ~0.3 s, faster than detect + brake + descend. About 80 % of the switching methods' deaths
happen in BRAKE (the walk → stand handoff under a rising load).

## 2. Leg-fault walking (`leg`, ~2 min)

Walk 11 m; at t = 5 s the front-right leg's torque limits derate 1.0 → 0.15 over 3 s, stay derated until
t = 15 s, then recover; 30 s horizon; death = the environment's terminations (tilt > 70°, or a non-foot contact
above 500 N). The descent trigger is the torque-saturation residual (the value cannot see this fault); the
return gate is the leg-healthy belief plus `V_up`. Outputs: `outputs/leg/`, figure `leg_topdown.png`.

Expected (success / safe, seed 0, N = 256):

| WALK-ONLY | REST-ONLY | ONE-WAY | ODD-conditioned (direct) | ODD-conditioned |
|---|---|---|---|---|
| 0.19 / 0.19 | 0.00 / 1.00 | 0.07 / 0.27 | 0.16 / 0.19 | **0.28 / 0.28** |

What must reproduce: ODD-conditioned has the highest success (it is the only method that both survives the
fault and resumes); the residual fires a median 7.9 s into the run, after the fault begins and never before
it; ONE-WAY is safer than walking on but rarely finishes; most switching-method deaths are in BRAKE.

## 3. Weight walking — the boundary (`weight-walk`, ~15 min)

The same walking task with the load-naive walker carrying 220 N at h = 0.25 m during [5, 13) s, goal 12 m,
three push conditions. **A negative result**: every walking method ends at safe ≤ 0.03 and success ≤ 0.01 in
every condition (REST-ONLY 1.00). Walking under this load is infeasible for a load-naive walker, and settling
under peak load from a gait flips robots — which is why the payload walk uses an unloaded light phase, a load
belief trigger, a brake and a dedicated descent funnel.

## 4. Standing automaton (`standing`, ~30 min; `seeds` adds seeds 1–3)

The robot stands under a lateral push while the carried load W (h = 0.25 m) varies; death = the environment's
terminations. **Waves** (20 s): `square` 40 ↔ 220 N every 3 s, `sine` 130 ± 110 N with an 8 s period.
**Single excursions** (24 s): `pulse` 40 → 220 → 40 N, `period` one cosine period 20 → 240 → 20 N. Push
`benign` = 10 N, `gusty` = 35 N gusts for 0.5 s every 2 s. Figures `standing_survival_{waves,single}.png`.

Expected safe rate (mean ± sd over seeds 0–3):

| scenario | ODD-conditioned | ODD-cond. (rest descent) | ODD-cond. (direct) | ONE-WAY | STAND-ONLY | REST-ONLY |
|---|---|---|---|---|---|---|
| single benign / period | **0.74±0.03** | 0.54±0.02 | 0.25±0.01 | 0.62±0.02 | 0.03±0.01 | 1.00 |
| single benign / pulse | 0.51±0.01 | 0.44±0.01 | 0.33±0.03 | **0.56±0.03** | 0.07±0.02 | 1.00 |
| single gusty / period | 0.22±0.03 | 0.16±0.02 | 0.08±0.01 | 0.20±0.01 | 0.03±0.01 | 1.00 |
| single gusty / pulse | 0.23±0.01 | 0.23±0.01 | 0.16±0.01 | 0.24±0.01 | 0.05±0.01 | 1.00 |
| waves benign / square | 0.47±0.03 | 0.41±0.04 | 0.21±0.04 | **0.57±0.03** | 0.01±0.01 | 1.00 |
| waves benign / sine | 0.31±0.01 | 0.25±0.03 | 0.12±0.01 | 0.34±0.03 | 0.00 | 1.00 |
| waves gusty / square | 0.20±0.01 | 0.20±0.02 | 0.09±0.01 | 0.25±0.01 | 0.00 | 1.00 |
| waves gusty / sine | 0.06±0.01 | 0.04±0.01 | 0.02±0.01 | 0.06±0.01 | 0.01 | 1.00 |

What must reproduce: REST-ONLY 1.00 everywhere (settled rest is the one true safe set of this specification);
STAND-ONLY ≈ 0; ODD-conditioned beats its direct ablation in every scenario and seed (by 0.46–0.51 on single
benign/period), and the descent funnel beats the rest-expert descent under benign pushes (by 0.14–0.25 on single
benign/period); ODD-conditioned is the best switching method on the smooth single excursion, while ONE-WAY edges
it on steps and square waves — every return costs another descent, and descents are where benign-push deaths
happen. Under gusts every method except REST bleeds while still standing.

## 5. Seeds (`seeds`, ~1.5 h)

Reruns the payload and standing tables at seeds 1–3 and aggregates mean ± sd with the main runs (seed 0) →
`outputs/seeds/summary.md`. The tables in §1 and §4 are this summary.

## 6. Certificate experiments (`certificates`, ~25 min)

| output | measures | expected |
|---|---|---|
| `region_grid.json` → `certifiable_regions.png` | STAND / REST failure fraction (tip or slam) over (W × push) and (leg θ × push at 80 N) | STAND is an island: at 5–10 N it holds up to W ≈ 140–150 N, at 20 N roughly 30–110 N, and nowhere above ~25 N; REST fails ≤ 0.04 over the whole weight plane. Compound STAND holds to θ ≈ 0.3 at 10 N |
| `ramp_weight.json` | the demo ramp (W 0 → 250 N with gusts), stand expert handing off to REST on its value | handoff at W ≈ 121 ± 62 N (the spread is gust timing); tip 0.14 vs 0.67 for STAND-ONLY |
| `ramp_compound.json` → `compound_timeline.png` | the leg dies (θ 1.0 → 0.1) while carrying 80 N | handoff at θ ≈ 0.28 ± 0.34, on the certifiable boundary; tip 0.28 vs 0.63 (STAND-ONLY), slam 0.40 vs 0.52 — slams are the weak point; standing kept 0.55 of the time (REST-ONLY 0) |
| `value_*.json` → `certificate_contraction.png` | V_stand along each ODD axis at reached states | contraction (V(0) − V(x_max)) / σ̄: weight 1.30σ, compound 1.57σ, unloaded leg 0.52σ — the negative control: its band discrimination is 0.02, the certificate is flat because the policy absorbs the fault |
| `ramp_weight-wide.json` → `certification_deficit.png` | the same ramp with the wide-range stand expert, against single-specification baselines | V_unified **rises** to ~0.85 as standing becomes impossible, while V_stand falls through its trigger — only the family signals the switch; the unified policy gives up standing early (0.13 of the time vs 0.47 for the handoff); the wide stand certificate is noisier than the feasible-band one (discrimination 0.96 vs 1.17; handoff tip 0.19 vs 0.14) |
| `forced_switch.json` | the hand-off forced at a fixed load on that ramp | tip ≤ 0.06 for switches between 40 and 120 N; ≥ 0.20 from 160 N; 0.33 when switching at once — the window the demo triggers must land in |
| `compound_matrix.json` | compound STAND / REST at fixed θ while carrying 80 N | STAND: stance ≥ 0.97 down to θ = 0.3 (tip ≤ 0.11), tip 0.51 at θ = 0.2 and 0.63 at θ = 0.1; REST tip-free at every θ |

## 7. Figures and videos

`figures` re-plots everything from `outputs/` (no simulation, ~1 min); a figure whose inputs are missing is
skipped with the target that makes them. `videos` renders `payload_{period,dip}.mp4` (one solo robot per
method, best of up to six seeds, above the fleet curves — run `payload` first), `leg_fault.mp4`,
`standing_{pulse,period}_benign.mp4` and `compound_leg_death.mp4` (run `certificates` first).

## 8. With retrained policies

Train ([TRAINING.md](TRAINING.md)), accept (`scripts/check_policies.py`), calibrate (`scripts/calibrate.py`), then
rerun. The structural results — the anchors, the pulse and weight-walk boundaries, the leg-fault ordering, the
certification deficit — hold for retrained policies; the walking and standing levels depend on how well the
stand expert brakes and holds a loaded stance.

---

## Verification record

**The package against the experiment scripts it replaces.** Seeded copies of the original experiment scripts
were run against the package at the same seed for every scenario × method. Rollouts on this GPU simulator are
not bit-deterministic run to run (contact-rich steps occasionally differ in the last bit), so cases were
classified by their first divergence: walking 26/40 bit-identical over full horizons and the rest diverging from
a single-ulp difference (≤ 2.4e-7 m) in the first steps; every ODD schedule and push profile identical at every
step; in the loaded standing and certificate environments the original scripts also disagree with themselves
at the same seed, by as much as they disagree with the package (e.g. forced switch at 160 N: tip 0.19 / 0.21 /
0.21 original vs 0.22 / 0.20 package). No logic difference was found.

**Earlier repetitions were not independent.** Loading a stable-baselines3 checkpoint reseeds torch with the
seed it was trained with, and the original scripts loaded policies between building the environment and
resetting it, so every repetition reset from the same random state. Earlier tables with "± over 4 repetitions"
therefore understated the spread (e.g. payload period: 0.38 ± 0.02 then, 0.43 ± 0.02 over independent seeds
now; an unseeded control gives 0.42 ± 0.02). The package reseeds after loading, so its seeds are independent.

**Fresh clone** of this branch with the published weights: smoke suite 10/10, leg-fault target reproduced.
