#!/usr/bin/env bash
# E008c driver: fan the 5 independent trainings out across cores, wait, then score.
# Each job caps torch at 4 threads (measured optimum for the tiny nets); 5×4=20 ≤ 24 cores,
# so wall-clock ≈ one training (~50 min at 300k), not 5× that.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=~/miniconda3/envs/odd-cond/bin/python
STEPS=${STEPS:-300000}
SEED=${SEED:-0}
OUT=${OUT:-results/E008c}
mkdir -p "$OUT"

export TORCH_THREADS=4
common="--steps $STEPS --seed $SEED --g0 0.99 --g-end 0.999 --spawn-omega-frac 0.98 --out $OUT"

echo "E008c parallel: 5 arms, $STEPS steps, seed $SEED, 4 threads each -> $OUT"
declare -a PIDS=()
for spec in "conditioned:" "blind:" "specialist:2.0" "specialist:5.0" "specialist:8.0"; do
  arm="${spec%%:*}"; mass="${spec##*:}"
  tag="$arm${mass:+_m$mass}_s$SEED"
  massflag=""; [ -n "$mass" ] && massflag="--mass $mass"
  $PY -u experiments/E008c_gate.py $common --arm "$arm" $massflag > "$OUT/${tag}.log" 2>&1 &
  PIDS+=($!)
  echo "  launched $tag (pid $!)"
done

echo "waiting on ${#PIDS[@]} trainings ..."
fail=0
for pid in "${PIDS[@]}"; do wait "$pid" || fail=$((fail+1)); done
echo "all trainings done ($fail failed)"

echo "=== scoring ==="
$PY experiments/E008c_gate.py --steps $STEPS --seeds $SEED --out "$OUT" --score-only \
    --specialist-rungs 2.0 5.0 8.0 --rungs 2.0 3.5 5.0 6.5 8.0 --no-wandb \
    2>&1 | grep -vE "Adroit|render_mode|UserWarning"
