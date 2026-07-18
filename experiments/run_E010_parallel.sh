#!/usr/bin/env bash
# E010 driver: fan the 5 arms across cores (4 torch threads each; 5x4=20 <= 24), wait, then score.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=~/miniconda3/envs/odd-cond/bin/python
STEPS=${STEPS:-150000}; SEED=${SEED:-0}; OUT=${OUT:-results/E010}
mkdir -p "$OUT"; export TORCH_THREADS=4
echo "E010 parallel: 5 arms, $STEPS steps, seed $SEED, 4 threads each -> $OUT"
declare -a PIDS=()
for arm in blind blind_dyn spec_hi spec_lo oracle; do
  $PY -u experiments/E010_bicycle_vanilla_dynamicodd.py --steps "$STEPS" --seed "$SEED" \
      --out "$OUT" --arm "$arm" > "$OUT/${arm}.log" 2>&1 &
  PIDS+=($!); echo "  launched $arm (pid $!)"
done
fail=0; for pid in "${PIDS[@]}"; do wait "$pid" || fail=$((fail+1)); done
echo "trainings done ($fail failed)"
echo "=== scoring ==="
$PY experiments/E010_bicycle_vanilla_dynamicodd.py --steps "$STEPS" --seed "$SEED" --out "$OUT" \
    --score-only --no-wandb 2>&1 | grep -vE "Adroit|render_mode|UserWarning|gymnasium"
