#!/usr/bin/env bash
# E013 Rung1 driver: fan the 4 arms across cores (4 torch threads each; 4x4=16 <= cores), then done.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=~/miniconda3/envs/odd-cond/bin/python
STEPS=${STEPS:-400000}; SEED=${SEED:-0}; OUT=${OUT:-results/E013}
mkdir -p "$OUT"; export TORCH_THREADS=4
echo "E013 Rung1: 4 arms, $STEPS steps, seed $SEED, 4 threads each -> $OUT"
declare -a PIDS=()
for arm in conditioned blind spec_hi spec_lo; do
  $PY -u experiments/E013_rung1_conditioned_critic.py --steps "$STEPS" --seed "$SEED" \
      --out "$OUT" --arm "$arm" > "$OUT/${arm}.log" 2>&1 &
  PIDS+=($!); echo "  launched $arm (pid $!)"
done
fail=0; for pid in "${PIDS[@]}"; do wait "$pid" || fail=$((fail+1)); done
echo "E013 trainings done ($fail failed)"
