# Installation and first run

From a clean machine to a reproduced paper number in about 30 minutes (most of it downloads).

## 0. Requirements

- Linux with an NVIDIA GPU and a driver new enough for CUDA 12.8 wheels (tested: RTX 4070 12 GB, driver
  580). A full N=256 evaluation needs only ~0.6 GB of GPU memory; training (1024 envs) fits a 12 GB card.
- conda (miniconda or miniforge), git, ~10 GB free disk (conda env ~6 GB, repo + checkpoints ~1 GB).
- Access to the four public SafeRoboticsLab repositories (this one plus three submodules). The submodule
  URLs use SSH; if you clone over HTTPS instead, run once:
  `git config --global url."https://github.com/".insteadOf git@github.com:`

## 1. Clone with submodules

```bash
git clone --recurse-submodules git@github.com:SafeRoboticsLab/odd-conditioned.git
cd odd-conditioned
git submodule status          # three pinned commits, none prefixed with '-'
```

| submodule | what it provides | pinned at |
|---|---|---|
| `external/safety-stable-baselines` | `safety_sb3` — the reach-avoid / two-player learners (`ReachAvoidPPO2P`) | tag `v0.4.0` |
| `external/robot-safety-sandbox` | `robot_safety_sandbox` — mjlab Go2 envs, tasks, margins, trainers. **This project's tasks live on branch `project/odd-conditioned-go2-payload`** (weight ladder, leg family, compound, transitions) | `a3c14f2` |
| `external/go2_atomic_skills` | the pretrained Go2 joystick walker (`walker_actor.pt`), the nominal task policy in the walking experiments | `ce0169d` (v0.4.4) |

## 2. Python environment

```bash
conda create -y -n mjlab python=3.11
conda activate mjlab
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

The pins in `requirements.txt` are the versions every reported result was produced with. The simulator
pins are not optional: mjlab does not pin mujoco / mujoco-warp itself, and an unpinned install fails at the
first env build (`ls_parallel was removed`). Note this project uses **mjlab 1.1.1** even though the
sandbox's own INSTALL.md now recommends 1.2.0.

Already have a conda env with these packages under another name? Use it: `export ODD_CONDA_ENV=<name>`.

## 3. Activate

```bash
source activate.sh
```

Every session starts with this. On first use it creates `.venv/` (inheriting the conda env) and
editable-installs the two safety packages; every time, it prepends the three submodules to `PYTHONPATH`,
sets `MUJOCO_GL=egl`, `cd`s to the repo root, and prints where each package resolved from. You must see
`resolution : OK` — anything else means another copy of `safety_sb3` / `robot_safety_sandbox` /
`go2_atomic_skills` installed in the conda env is shadowing the pinned submodule.

Run every experiment script **from the repo root** (`python experiments/E092_payload_walk.py`):
checkpoint paths and submodule imports are repo-root relative, and the scripts refuse to start elsewhere.

## 4. Trained checkpoints

Checkpoints are not in git. You get one archive, `odd-conditioned-weights-v1.tar.gz` (228 MB), from Buzi.
Put it in any directory and install it:

```bash
bash scripts/fetch_bundles.sh --from <directory containing the archive>
```

This extracts the 16 reach-avoid PPO twins the experiments load (17.5 MB each) plus the two warm-start
sources needed to retrain, at their recorded paths under `results/<family>/<run>/...`, and checks every file
against `weights/MANIFEST.sha256`. Only four of them drive the paper's automaton (see
[TRAINING.md §2](TRAINING.md#2-the-16-policies)); the rest are needed by the certificate and demo
experiments. Without the archive you can retrain everything ([TRAINING.md](TRAINING.md)), but the numbers
will differ and the switching thresholds need recalibrating.

## 5. Verify

```bash
pytest -q tests/                 # 12 passed, 1 skipped in ~35 s on an RTX 4070
pytest -q tests/ -m "not gpu"    # the CPU-only subset (imports, task registry, checksums)
```

The smoke suite checks that the submodules resolve, the project's tasks are registered, every checkpoint
matches its published checksum and loads, the walker loads, and that short rollouts of the standing
(E084), payload-walk (E092) and leg-walk (E091) automata run and behave sanely.

## 6. First reproduction

```bash
bash scripts/reproduce.sh e091     # leg-fault walking, ~2 min
bash scripts/reproduce.sh e092     # payload-swap walking (the paper's main walking table), ~6 min
```

Outputs go to `repro/` (override with `ODD_ARTIFACTS=<dir>`). The expected numbers for every target are in
[REPRODUCE.md](REPRODUCE.md).

## Troubleshooting

| symptom | cause / fix |
|---|---|
| `resolution : WRONG` from `activate.sh` | a non-editable copy of one of the three packages sits in the conda env and wins over `.venv`. The PYTHONPATH prepend should still win — make sure you `source`d the script (not `bash activate.sh`), and that nothing in your shell rc re-sets `PYTHONPATH` after it |
| `missing checkpoints — run bash scripts/fetch_bundles.sh` | weights not fetched, or you are not at the repo root |
| long pause at the first env build | mujoco-warp JIT-compiles its kernels for your GPU architecture once; later runs reuse the cache |
| OpenGL / EGL errors when rendering videos | headless machines need `MUJOCO_GL=egl` (`activate.sh` sets it) and the NVIDIA EGL libraries |
| `ls_parallel was removed` | unpinned mujoco-warp ≥ 3.9.1: reinstall from `requirements.txt` |
| `fatal: ... not our ref` on `submodule update` | your clone predates the submodule push, or the submodule branch is not yet on GitHub — `git submodule sync && git submodule update --init` after pulling |
| CUDA out of memory | close other GPU jobs; the evaluation scripts take `n=` in `rollout(...)` if you need a smaller fleet for a quick check |

## Repository layout

```
activate.sh                 environment entry point (source it)
requirements.txt            conda-env pins (torch installed separately, see above)
experiments/E0XX_*.py       one script per experiment ID — the provenance record; see docs/EXPERIMENTS.md
experiments/_paths.py       ODD_ARTIFACTS (output root) + repo-root guard
odd_conditioned/            the bicycle / friction ODD envs of the early toy experiments (E010–E013)
scripts/                    reproduce.sh, fetch_bundles.sh, make_bundles.py (maintainers)
tests/                      smoke suite (+ the E011 friction-env parity test)
weights/MANIFEST.sha256     checksums of every checkpoint file in the weights bundle
external/                   the three pinned submodules
docs/                       this documentation; docs/history/ = superseded write-ups
results/   (git-ignored)    checkpoints (from the bundle) and training logs
repro/     (git-ignored)    your reproductions (scripts/reproduce.sh)
```
