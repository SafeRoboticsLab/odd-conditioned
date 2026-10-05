#!/usr/bin/env bash
# odd-conditioned environment. `source activate.sh` from anywhere, then run the scripts as normal python:
# `python scripts/evaluate.py payload` — no wrapper needed.
#
# Layout: a conda env (default name `mjlab`, override with ODD_CONDA_ENV) supplies the heavy/compiled
# deps (torch+CUDA, mujoco, mujoco-warp, warp, mjlab 1.1.1 — see requirements.txt); a per-project
# `.venv --system-site-packages` inherits them; and a PYTHONPATH prepend makes `safety_sb3`,
# `robot_safety_sandbox` and `go2_atomic_skills` resolve to OUR pinned submodules under external/ —
# not to whatever else is installed in the conda env.
#
# Why PYTHONPATH (not just the .venv editable installs): a `--system-site-packages` venv editable
# CANNOT override a package that also exists in the base env (base's PathFinder wins over the venv's
# PEP 660 finder). PYTHONPATH precedes site-packages in sys.path, so prepending our source trees is
# what gives them precedence.
#
# Overrides: ODD_CONDA_ENV (conda env name), CONDA_BASE (conda install root),
#            ODD_CHECKPOINTS / ODD_OUTPUTS (where policies are read / results written; odd_conditioned/paths.py).
# Deliberately no `set -e`: this file is SOURCED, and `set -e` would leak into your shell.

_odd_activate() {
  local WS SB3 RSS G2S base envname d
  WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  SB3="$WS/external/safety-stable-baselines"        # safety_sb3 v0.4.0
  RSS="$WS/external/robot-safety-sandbox"            # RSS 0.4.0 + this project's tasks (project/odd-conditioned)
  G2S="$WS/external/go2_atomic_skills"               # pretrained Go2 walker (nominal task policy)

  for d in "$SB3/safety_sb3" "$RSS/robot_safety_sandbox" "$G2S/go2_atomic_skills"; do
    if [ ! -d "$d" ]; then
      echo "ERROR: missing $d — run: git -C \"$WS\" submodule update --init --recursive" >&2
      return 1
    fi
  done

  envname="${ODD_CONDA_ENV:-mjlab}"
  base="${CONDA_BASE:-$(conda info --base 2>/dev/null)}"
  base="${base:-$HOME/miniconda3}"
  if [ ! -f "$base/etc/profile.d/conda.sh" ]; then
    echo "ERROR: conda not found at $base (set CONDA_BASE)" >&2
    return 1
  fi
  # shellcheck disable=SC1091
  source "$base/etc/profile.d/conda.sh"
  conda activate "$envname" || { echo "ERROR: conda env '$envname' missing — see docs/INSTALL.md" >&2; return 1; }

  if [[ ! -d "$WS/.venv" ]]; then
    echo "[env] creating .venv (inherits conda env '$envname': torch/CUDA/warp/mjlab)"
    python -m venv "$WS/.venv" --system-site-packages || return 1
    # editable installs give each package its dist-info/metadata; --no-deps so RSS's
    # `safety_sb3 @ git+...` pin is ignored; import precedence comes from PYTHONPATH below.
    "$WS/.venv/bin/pip" install -q --no-deps --no-build-isolation --force-reinstall -e "$SB3" || return 1
    "$WS/.venv/bin/pip" install -q --no-deps --no-build-isolation --force-reinstall -e "$RSS" || return 1
  fi

  # shellcheck disable=SC1091
  source "$WS/.venv/bin/activate"
  export PYTHONPATH="$SB3:$RSS:$G2S${PYTHONPATH:+:$PYTHONPATH}"
  export MUJOCO_GL="${MUJOCO_GL:-egl}"

  python - <<'PY'
import safety_sb3, robot_safety_sandbox as r, go2_atomic_skills as g
ok = all(p in m.__file__ for p, m in (("external/safety-stable-baselines", safety_sb3),
                                       ("external/robot-safety-sandbox", r),
                                       ("external/go2_atomic_skills", g)))
print("safety_sb3          :", safety_sb3.__file__)
print("robot_safety_sandbox:", r.__file__)
print("go2_atomic_skills   :", g.__file__)
print("resolution          :", "OK" if ok else "WRONG — another copy shadows the submodules")
PY
  cd "$WS" || return 1
  echo "[env] ready — e.g. python scripts/evaluate.py payload"
}
_odd_activate
