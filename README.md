# ODD-conditioned safety filters

Safety filters whose guarantee survives **runtime changes of the operating design domain (ODD)** — a payload
loaded mid-mission, a motor that derates — demonstrated on a Unitree Go2 in MuJoCo (mjlab).

The filter is an automaton over **specification modes** (stand/walk ↔ rest), each with its own adversarially
trained reach-avoid expert whose learned value is that mode's certificate, joined by **certified transition
funnels** (descend, get up) and switched by **belief-primary triggers checked by the certificates**. It
withdraws to a safe mode only while the ODD requires it and resumes the task when the ODD clears.

| walking under a payload swap (E092; success / safe, N=256, 4 evaluation reps) | ramp | deceptive ramp | instant |
|---|---|---|---|
| walker alone | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| lie down and stay (REST-ONLY) | 0.00 / 1.00 | 0.00 / 1.00 | 0.00 / 1.00 |
| descend once, never return (ONE-WAY) | 0.00 / 0.42 | 0.00 / 0.42 | 0.00 / 0.02 |
| ODD-conditioned (direct) — no certified return | 0.37 / 0.37 | 0.33 / 0.36 | 0.03 / 0.03 |
| **ODD-conditioned** | **0.38 / 0.39** | **0.40 / 0.40** | 0.03 / 0.03 |

The instant load is a stated boundary: a 220 N load landing mid-stride flips the robot faster than any
detect-and-descend maneuver. Full results and caveats: [docs/FINDINGS.md](docs/FINDINGS.md).

## Quick start

```bash
git clone --recurse-submodules git@github.com:SafeRoboticsLab/odd-conditioned.git && cd odd-conditioned
conda create -y -n mjlab python=3.11 && conda activate mjlab
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
source activate.sh                          # every session; prints "resolution : OK"
bash scripts/fetch_bundles.sh --from <dir>  # the checkpoint archive you were sent -> results/ (verified)
pytest -q tests/                            # smoke suite, ~35 s
bash scripts/reproduce.sh e092              # the walking table above, ~6 min -> repro/E092-payload-walk/
```

## Documentation

| | |
|---|---|
| [docs/INSTALL.md](docs/INSTALL.md) | environment, submodules, checkpoints, verification, troubleshooting, repo layout |
| [docs/REPRODUCE.md](docs/REPRODUCE.md) | every paper result: command, runtime, expected numbers |
| [docs/FINDINGS.md](docs/FINDINGS.md) | what the project established, the negative results that shaped it, caveats |
| [docs/SWITCHING_LOGIC.md](docs/SWITCHING_LOGIC.md) | the automaton: every guard and threshold, V1 vs V2 |
| [docs/TRAINING.md](docs/TRAINING.md) | how each of the 16 policies was trained; env and margin definitions |
| [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) | index of all experiments E001–E094 and what superseded what |
| [docs/STATUS.md](docs/STATUS.md) | where the work stands and what to pick up next |
| `docs/history/` | superseded write-ups (the B̂ certificate note, the 0.4.0 migration, env layout) |

## Dependencies

Three pinned submodules under `external/`:
[safety-stable-baselines](https://github.com/SafeRoboticsLab/safety-stable-baselines) (the reach-avoid learners,
v0.4.0), [robot-safety-sandbox](https://github.com/SafeRoboticsLab/robot-safety-sandbox) (mjlab envs and tasks;
this project's tasks are on branch `project/odd-conditioned-go2-payload`), and
[go2_atomic_skills](https://github.com/SafeRoboticsLab/go2_atomic_skills) (the pretrained walker).
