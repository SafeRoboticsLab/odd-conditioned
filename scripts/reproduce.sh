#!/usr/bin/env bash
# Reproduce the paper-draft results.   bash scripts/reproduce.sh <target> [<target> ...]
#
#   smoke       pytest smoke suite (env, checkpoints, short rollouts)                      ~1 min
#   e092        payload-swap walking: pulse / period / dip x 5 arms  (paper Sec. 3)        ~6 min
#   e091        leg-fault walking: FR-leg derate -> heal x 5 arms    (paper Sec. 6)        ~2 min
#   e089        weight-excursion walking with the naive walker (boundary finding)          ~15 min
#   standing    E084 load waves + E086 single excursions (+ videos)  (paper Sec. 4)        ~30 min
#   seeds       E094: 3 more reps of e092/waves/single, then mean+-sd (needs e092+standing) ~1.5 h
#   figures     regenerate F2/F3/F7 + walking/standing figures from data (E093) — needs the
#               outputs of `certificates compound e092 standing` first                     ~1 min
#   compound    the leg-death-while-loaded demo (E078: matrix, value, ramp, video) (Sec. 5) ~20 min
#   certificates  the value sweeps + demo ramps behind F2/F7/F3 (E079, E074, E075, E076, E078) ~15 min
#   videos      E092 + E091 demo videos (rendered, slow)                                    ~20 min
#
# Outputs go to $ODD_ARTIFACTS (default: <repo>/repro — NOT ~/artifacts, so a reproduction never
# overwrites the maintainer's reference outputs). Logs: $ODD_ARTIFACTS/logs/<target>.log.
# If a reference/ directory exists (maintainer-only bundle), `figures` first copies it into $ODD_ARTIFACTS
# without clobbering, so it can re-plot without simulating.
# Runtimes are for one RTX 4070 (12 GB).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
if [ -z "${VIRTUAL_ENV:-}" ] || [[ "${PYTHONPATH:-}" != *external/robot-safety-sandbox* ]]; then
  # shellcheck disable=SC1091
  source "$REPO/activate.sh" >/dev/null
fi
export ODD_ARTIFACTS="${ODD_ARTIFACTS:-$REPO/repro}"
mkdir -p "$ODD_ARTIFACTS/logs"

run() {   # run <log name> <python args...>
  local name="$1"; shift
  echo "[repro] $name: python $*  (log: $ODD_ARTIFACTS/logs/$name.log)"
  python "$@" 2>&1 | grep --line-buffered -v -i 'warn' | tee -a "$ODD_ARTIFACTS/logs/$name.log"
}

seed_reference() {   # copy reference data in (if present) without overwriting anything this reproduction made
  if [ -d "$REPO/reference" ]; then
    cp -rn "$REPO/reference/." "$ODD_ARTIFACTS/"
  fi
}

check_figure_inputs() {   # E093 re-plots from data: say which target produces anything missing
  local missing=0 f
  declare -A from=(
    [E077-figures/F2_grid.json]=certificates [E074-hicom-demo/task2_value.json]=certificates
    [E074-hicom-demo/task3_ramp.json]=certificates [E075-recal-eval/partA_value.json]=certificates
    [E076-leg-demo/partB_value.json]=certificates [E078-compound-demo/task2_value.json]=certificates
    [E075-recal-eval/partA_ramp.json]=certificates [E078-compound-demo/task3_ramp.json]=compound
    [E092-payload-walk/results.json]=e092 [E092-payload-walk/traj_dip.npz]=e092
    [E084-automaton/results.json]=standing [E084-automaton/results_single.json]=standing
  )
  for f in "${!from[@]}"; do
    if [ ! -f "$ODD_ARTIFACTS/$f" ]; then
      echo "[repro] figures needs $f — run: bash scripts/reproduce.sh ${from[$f]}" >&2
      missing=1
    fi
  done
  return $missing
}

run_target() {   # one target; its commands are chained so a failure stops that target only
  case "$1" in
    smoke)    python -m pytest -q tests/ -W ignore ;;
    e092)     run e092 experiments/E092_payload_walk.py ;;
    e091)     run e091 experiments/E091_leg_walk.py ;;
    e089)     run e089 experiments/E089_goal_walk.py ;;
    standing) run e084 experiments/E084_automaton.py &&
              run e086 experiments/E086_single_pulse.py ;;
    seeds)    for rep in 1 2 3; do
                for table in e092 waves single; do
                  run "e094_${table}_rep$rep" experiments/E094_seeds.py --table "$table" --rep "$rep" || return 1
                done
              done
              run e094_aggregate experiments/E094_seeds.py --aggregate ;;
    figures)  seed_reference
              check_figure_inputs && run e093 experiments/E093_figpolish.py ;;
    compound) run e078_matrix experiments/E078_matrix.py &&
              run e078_value experiments/E078_value.py &&
              run e078_ramp experiments/E078_ramp.py &&
              run e078_video experiments/E078_video.py ;;
    certificates)
              run e079 experiments/E079_region_sweep.py &&
              run e074_value experiments/E074_value.py &&
              run e074_ramp experiments/E074_ramp.py &&
              run e075_ramp experiments/E075_ramp.py &&
              run e075_value experiments/E075_value.py &&
              run e076_value experiments/E076_value.py &&
              run e078_value experiments/E078_value.py ;;
    videos)   run e092_videos experiments/E092_payload_walk.py --videos &&
              run e091_videos experiments/E091_leg_walk.py --videos ;;
    *) echo "unknown target: $1" >&2; return 2 ;;
  esac
}

[ $# -gt 0 ] || { sed -n 2,22p "$0"; exit 1; }
failed=()
for target in "$@"; do
  if ! run_target "$target"; then
    echo "[repro] target '$target' FAILED — continuing with the rest" >&2
    failed+=("$target")
  fi
done
if [ ${#failed[@]} -gt 0 ]; then
  echo "[repro] done with failures: ${failed[*]} (logs: $ODD_ARTIFACTS/logs/)" >&2
  exit 1
fi
echo "[repro] done -> $ODD_ARTIFACTS"
