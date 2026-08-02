#!/usr/bin/env bash
# odd-conditioned environment. After `source activate.sh`, run experiments as NORMAL python
# scripts: `python experiments/E0XX.py` — no wrapper needed.
#
# Layout: shared `mjlab` conda base supplies the heavy/compiled deps (torch, CUDA, warp,
# mjlab 1.1.1); a per-project `.venv --system-site-packages` inherits them; and a PYTHONPATH
# prepend makes `safety_sb3` + `robot_safety_sandbox` resolve to OUR pinned copies under
# external/ — not to whatever another agent has installed in the shared base. Both are 0.4.0.
#
# Why PYTHONPATH (not just the .venv editable installs): a `--system-site-packages` venv
# editable CANNOT override a package that also exists in base (base's PathFinder wins over the
# venv's PEP 660 finder), and base ships a pinned safety_sb3. PYTHONPATH precedes site-packages
# in sys.path, so prepending our source trees is what gives them precedence. Vault:
# "Multi-project workflow — shared conda base + per-project venv overlays".
set -e
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SB3="$WS/external/safety-stable-baselines"        # our submodule/checkout, safety_sb3 v0.4.0
RSS="$WS/external/robot-safety-sandbox"            # our submodule fork, RSS 0.4.0

for d in "$SB3/safety_sb3" "$RSS/robot_safety_sandbox"; do
  [ -d "$d" ] || { echo "ERROR: missing $d" >&2; return 1 2>/dev/null || exit 1; }
done

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate mjlab

if [[ ! -d "$WS/.venv" ]]; then
  echo "[env] creating .venv (inherits base mjlab: torch/CUDA/warp/mjlab)"
  python -m venv "$WS/.venv" --system-site-packages
  # editable installs give each package its dist-info/metadata + an isolated pip target;
  # --no-deps so RSS's `safety_sb3 @ ...` pin is ignored; import precedence is from PYTHONPATH.
  "$WS/.venv/bin/pip" install --no-deps --no-build-isolation --force-reinstall -e "$SB3"
  "$WS/.venv/bin/pip" install --no-deps --no-build-isolation --force-reinstall -e "$RSS"
fi

source "$WS/.venv/bin/activate"
export PYTHONPATH="$SB3:$RSS${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

python - <<'PY'
import safety_sb3, robot_safety_sandbox as r, os
ok = "external/safety-stable-baselines" in safety_sb3.__file__ and "external/robot-safety-sandbox" in r.__file__
print("safety_sb3          :", safety_sb3.__file__)
print("robot_safety_sandbox:", r.__file__)
print("ReachAvoidSAC2P     :", hasattr(safety_sb3, "ReachAvoidSAC2P"), "| resolution", "OK" if ok else "WRONG")
PY
cd "$WS"
echo "[env] ready — run: python experiments/<name>.py"
