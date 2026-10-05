#!/usr/bin/env bash
# Reproduce the reported results with the installed checkpoints.   bash scripts/reproduce.sh <target> [...]
#
#   smoke         pytest suite: environment, checkpoints, short rollouts                     ~2 min
#   payload       payload-swap walking: pulse / period / dip x 5 methods                      ~6 min
#   leg           leg-fault walking x 5 methods                                               ~2 min
#   standing      standing automaton: load waves + single excursions x 6 methods              ~30 min
#   weight-walk   walking under a 220 N excursion: the boundary case                          ~15 min
#   certificates  value sweeps, handoff ramps, forced switch, region grid, compound matrix    ~25 min
#   seeds         payload + standing at seeds 1-3, mean ± sd (run payload + standing first)  ~1.5 h
#   figures       every figure from the saved results                                         ~1 min
#   videos        the demo videos (rendered; slow)                                            ~30 min
#   all           everything above except smoke, in dependency order                          ~3.5 h
#
# Outputs: $ODD_OUTPUTS (default <repo>/outputs); logs in $ODD_OUTPUTS/logs/<target>.log.
# A failing target does not stop the others; the exit status reports any failure.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
if [ -z "${VIRTUAL_ENV:-}" ] || [[ "${PYTHONPATH:-}" != *external/robot-safety-sandbox* ]]; then
  # shellcheck disable=SC1091
  source "$REPO/activate.sh" >/dev/null
fi
OUT="${ODD_OUTPUTS:-$REPO/outputs}"
mkdir -p "$OUT/logs"

run() {   # run <log name> <python args...>
  local name="$1"; shift
  echo "[repro] $name: python $*  (log: $OUT/logs/$name.log)"
  python "$@" 2>&1 | grep --line-buffered -v -i 'warn' | tee "$OUT/logs/$name.log"
  return "${PIPESTATUS[0]}"
}

run_target() {
  case "$1" in
    smoke)        python -m pytest -q tests/ -W ignore ;;
    payload|leg|standing|weight-walk|certificates|seeds)
                  run "$1" scripts/evaluate.py "$1" ;;
    figures)      run figures scripts/make_figures.py ;;
    videos)       run videos scripts/make_videos.py payload leg standing compound ;;
    *) echo "unknown target: $1" >&2; return 2 ;;
  esac
}

[ $# -gt 0 ] || { sed -n 2,16p "$0"; exit 1; }
targets=()
for a in "$@"; do
  if [ "$a" = all ]; then targets+=(payload leg standing certificates weight-walk seeds figures videos)
  else targets+=("$a"); fi
done
failed=()
for t in "${targets[@]}"; do
  run_target "$t" || { echo "[repro] target '$t' FAILED — continuing" >&2; failed+=("$t"); }
done
if [ ${#failed[@]} -gt 0 ]; then
  echo "[repro] done with failures: ${failed[*]} (logs: $OUT/logs/)" >&2
  exit 1
fi
echo "[repro] done -> $OUT"
