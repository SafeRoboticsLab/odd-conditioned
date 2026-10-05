# Installation and first run

From a clean machine to a reproduced result in about 30 minutes (most of it downloads).

## 0. Requirements

- Linux with an NVIDIA GPU and a driver new enough for CUDA 12.8 wheels (tested: RTX 4070 12 GB, driver 580).
  An N = 256 evaluation needs ~0.6 GB of GPU memory; training (1024 envs) fits a 12 GB card.
- conda (miniconda or miniforge), git, ~10 GB free disk (conda env ~6 GB, repo + checkpoints ~0.5 GB).
- The submodule URLs use SSH; to clone over HTTPS instead, run once:
  `git config --global url."https://github.com/".insteadOf git@github.com:`

## 1. Clone with submodules

```bash
git clone --recurse-submodules git@github.com:SafeRoboticsLab/odd-conditioned.git
cd odd-conditioned
git submodule status          # three pinned commits, none prefixed with '-'
```

| submodule | provides |
|---|---|
| `external/safety-stable-baselines` | `safety_sb3` — the reach-avoid / two-player learners (`ReachAvoidPPO2P`), tag `v0.4.0` |
| `external/robot-safety-sandbox` | `robot_safety_sandbox` — the mjlab Go2 environments, tasks, margins and training configs. This project's tasks (weight ladder, leg family, compound, transition funnels) live on branch `project/odd-conditioned` |
| `external/go2_atomic_skills` | the pretrained Go2 joystick walker (`walker_actor.pt`), the nominal task policy of the walking scenarios |

## 2. Python environment

```bash
conda create -y -n mjlab python=3.11
conda activate mjlab
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

The pins in `requirements.txt` are the versions every reported result was produced with. The simulator pins
are not optional: mjlab does not pin mujoco / mujoco-warp itself, and an unpinned install fails at the first
env build (`ls_parallel was removed`). This project uses **mjlab 1.1.1**.

Already have a conda env with these packages under another name? `export ODD_CONDA_ENV=<name>`.

## 3. Activate

```bash
source activate.sh
```

Every session starts with this. On first use it creates `.venv/` (inheriting the conda env) and
editable-installs the two safety packages; every time, it prepends the three submodules to `PYTHONPATH`, sets
`MUJOCO_GL=egl`, `cd`s to the repo root, and prints where each package resolved from. You must see
`resolution : OK` — anything else means another copy of `safety_sb3` / `robot_safety_sandbox` /
`go2_atomic_skills` in the conda env shadows the pinned submodule.

## 4. Trained checkpoints

Checkpoints are not in git. Install the published archive (~155 MB) and verify it:

```bash
bash scripts/fetch_weights.sh <odd-conditioned-weights-v1.tar.gz or its URL>
```

This puts every policy under `checkpoints/<name>/` (`model.zip`, `tensornormalize.pt`, `config.yaml`) and checks
each file against `weights/MANIFEST.sha256`. The policies and what they are for are listed in
[TRAINING.md](TRAINING.md#2-the-policies); four of them drive the automaton, the rest serve the certificate
experiments. Or train them yourself (`bash scripts/train.sh core certificates`, ~4.5 h) — read
[TRAINING.md](TRAINING.md) first: the stance experts vary from run to run and need an acceptance check.

## 5. Verify

```bash
pytest -q tests/                 # ~2 min on an RTX 4070
pytest -q tests/ -m "not gpu"    # the CPU-only subset
```

The smoke suite checks that the submodules resolve, the tasks are registered, every checkpoint matches its
published checksum and loads, the walker loads, and that short rollouts of the standing, payload-walk and
leg-fault automata run and behave sanely.

## 6. First reproduction

```bash
bash scripts/reproduce.sh leg          # leg-fault walking, ~2 min
bash scripts/reproduce.sh payload      # payload-swap walking, ~6 min
bash scripts/reproduce.sh figures      # the figures of whatever has been run
```

Outputs go to `outputs/` (override with `ODD_OUTPUTS=<dir>`). Expected numbers: [REPRODUCE.md](REPRODUCE.md).

## Troubleshooting

| symptom | cause / fix |
|---|---|
| `resolution : WRONG` from `activate.sh` | a non-editable copy of one of the three packages sits in the conda env. `source` the script (not `bash activate.sh`), and make sure nothing in your shell rc re-sets `PYTHONPATH` afterwards |
| `missing checkpoint .../model.zip` | weights not installed (`scripts/fetch_weights.sh`), or `ODD_CHECKPOINTS` points elsewhere |
| long pause at the first env build | mujoco-warp JIT-compiles its kernels for your GPU once; later runs reuse the cache |
| OpenGL / EGL errors when rendering videos | headless machines need `MUJOCO_GL=egl` (`activate.sh` sets it) and the NVIDIA EGL libraries |
| `ls_parallel was removed` | unpinned mujoco-warp: reinstall from `requirements.txt` |
| CUDA out of memory | close other GPU jobs, or pass `--n 64` to `scripts/evaluate.py` for a quick check |

## Repository layout

```
activate.sh                  environment entry point (source it)
requirements.txt             conda-env pins (torch installed separately, see above)
odd_conditioned/             the package
  scenarios.py               tasks, ODD schedules, per-regime switching thresholds
  automaton.py               the safety filter and its baselines: rollout(scenario, method)
  certificates.py            value sweeps, handoff ramps, region grid, compound matrix
  policies.py                reach-avoid twins, the walker, the leg residual
  sim.py                     the robot-safety-sandbox environment layer
  evaluate.py figures.py videos.py   the reported results, regenerated
  paths.py                   ODD_CHECKPOINTS / ODD_OUTPUTS
scripts/
  reproduce.sh               every result: bash scripts/reproduce.sh <target>
  evaluate.py make_figures.py make_videos.py
  train.sh                   train the policies from scratch
  check_policies.py          acceptance tests for retrained policies
  calibrate.py               re-place the switching thresholds for new networks
  fetch_weights.sh pack_weights.py train_curves.py
tests/                       smoke suite
weights/MANIFEST.sha256      checksums of the published checkpoints
external/                    the three pinned submodules
checkpoints/  (git-ignored)  installed policies
outputs/      (git-ignored)  results, figures, videos
runs/         (git-ignored)  training runs (scripts/train.sh)
```
