# ODD-conditioned safety filters

Safety filters whose guarantee survives **runtime changes of the operating design domain (ODD)** — a payload
loaded mid-mission, a motor that derates — demonstrated on a Unitree Go2 in MuJoCo (mjlab).

The filter is an automaton over **specification modes** (stand/walk ↔ rest). Each mode has its own
adversarially trained reach-avoid expert whose learned value is that mode's certificate; the modes are joined by
**certified transition funnels** (descend, get up) and switched by **belief-primary triggers checked by the
certificates**. It withdraws to the safe mode only while the ODD requires it, and resumes the task when the ODD
clears.

| walking under a payload swap — success / safe (N = 256, mean ± sd over 4 seeds) | ramp | deceptive ramp | instant |
|---|---|---|---|
| walker alone (WALK-ONLY) | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| lie down and stay (REST-ONLY) | 0.00 / 1.00 | 0.00 / 1.00 | 0.00 / 1.00 |
| descend once, never return (ONE-WAY) | 0.00 / 0.42±.03 | 0.00 / 0.42±.02 | 0.00 / 0.02 |
| ODD-conditioned (direct) — no certified return | 0.39±.03 / 0.40±.03 | 0.38±.03 / 0.41±.03 | 0.02 / 0.02 |
| **ODD-conditioned** | **0.43±.02 / 0.43±.02** | **0.43±.03 / 0.43±.03** | 0.02 / 0.02 |

The instant load is a stated boundary: a 220 N load landing mid-stride flips the robot faster than any
detect-and-descend maneuver. Standing, leg-fault and certificate results, and their caveats:
[docs/FINDINGS.md](docs/FINDINGS.md).

## What is in this repository

| | |
|---|---|
| [`odd_conditioned/`](odd_conditioned) | the filter (`automaton.py`), the scenarios (`scenarios.py`), the certificate experiments (`certificates.py`), and the code that regenerates every result, figure and video |
| [`scripts/`](scripts) | `reproduce.sh` (every result), `train.sh` (every policy from scratch), `calibrate.py` (switching thresholds for new networks), `check_policies.py` (acceptance tests for retrained policies) |
| [`external/`](external) | three pinned submodules: [safety-stable-baselines](https://github.com/SafeRoboticsLab/safety-stable-baselines) (the reach-avoid learners, v0.4.0), [robot-safety-sandbox](https://github.com/SafeRoboticsLab/robot-safety-sandbox) (the mjlab tasks and margins — branch `project/odd-conditioned`), [go2_atomic_skills](https://github.com/SafeRoboticsLab/go2_atomic_skills) (the pretrained walker) |

Two test cases, each under several ODD profiles: a **varying payload** (a tall crate loaded and unloaded while
walking: a step, a smooth ramp, or a ramp with a deceptive dip; and load waves and single excursions while
standing), and a **degraded leg** (the front-right motors derate while walking, then recover).

## Quick start

```bash
git clone --recurse-submodules git@github.com:SafeRoboticsLab/odd-conditioned.git && cd odd-conditioned
conda create -y -n mjlab python=3.11 && conda activate mjlab
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
source activate.sh                                # every session; prints "resolution : OK"
bash scripts/fetch_weights.sh <archive or URL>    # the published checkpoints -> checkpoints/ (verified)
pytest -q tests/                                  # smoke suite, ~2 min
bash scripts/reproduce.sh payload figures         # the walking table above, ~6 min -> outputs/
```

Or train every policy yourself: `bash scripts/train.sh core certificates` (~4.5 h on one RTX 4070) — see
[docs/TRAINING.md](docs/TRAINING.md) first: the stance experts vary from run to run and are accepted by test.

## Documentation

| | |
|---|---|
| [docs/INSTALL.md](docs/INSTALL.md) | environment, submodules, checkpoints, verification, troubleshooting, repository layout |
| [docs/REPRODUCE.md](docs/REPRODUCE.md) | every result: command, runtime, expected numbers |
| [docs/FINDINGS.md](docs/FINDINGS.md) | what the results establish, and their caveats |
| [docs/SWITCHING_LOGIC.md](docs/SWITCHING_LOGIC.md) | the automaton: every guard and threshold; the direct ablation |
| [docs/TRAINING.md](docs/TRAINING.md) | how each policy is trained, training variance and acceptance, threshold calibration |
