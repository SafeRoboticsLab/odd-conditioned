#!/usr/bin/env bash
# Train the policies the experiments load, from scratch, in dependency order.
#
#   bash scripts/train_all.sh core            # the 4 automaton policies (paper sections 3-4, 6)   ~2 h
#   bash scripts/train_all.sh certificates    # the 7 behind the certificate figures + demos      ~4 h
#   bash scripts/train_all.sh all             # both
#   bash scripts/train_all.sh rest stand ...  # individual policies (names below)
#
# core         = rest, stand, getup1, getup, descend
# certificates = stand_hi_orig, unified, unified_disc, leg_stand, leg_rest, compound_stand, compound_rest
#
# Runs land at the exact results/ paths the experiment scripts load, so the experiments use them with no
# code change. Afterwards, RECALIBRATE the switching thresholds (docs/TRAINING.md "Recalibrating").
# Runtimes: one RTX 4070, one run at a time (14-25 min per policy).
#
# Environment knobs:
#   ODD_RESULTS=<dir>  output root instead of results/ (the experiments only read results/)
#   ODD_WANDB=1        log to wandb (project odd-conditioned, your default entity); default: off
#   FORCE=1            allow writing into a run directory that already exists (default: refuse, so shipped
#                      or finished checkpoints are never overwritten)
#   SMOKE=1            a few hundred thousand steps per policy into a scratch root — checks the commands
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
if [ -z "${VIRTUAL_ENV:-}" ] || [[ "${PYTHONPATH:-}" != *external/robot-safety-sandbox* ]]; then
  # shellcheck disable=SC1091
  source "$REPO/activate.sh" >/dev/null
fi

CFG=external/robot-safety-sandbox/configs
EXTRA=()
if [ "${SMOKE:-0}" = 1 ]; then
  ROOT="${ODD_RESULTS:-$(mktemp -d)/results}"
  EXTRA+=(--steps 300000 --num-envs 256 --no-wandb)
  echo "[train] SMOKE run -> $ROOT"
else
  ROOT="${ODD_RESULTS:-results}"
  [ "${ODD_WANDB:-0}" = 1 ] || EXTRA+=(--no-wandb)
fi
mkdir -p "$ROOT/logs"

# name -> "config | --out | run dir it creates | extra flags"
spec() {
  case "$1" in
    rest)           echo "go2_weight_rest_hi_ppo.yaml|$ROOT/go2_weight_runs|$ROOT/go2_weight_runs/go2_weight_rest_hi_adv|" ;;
    stand)          echo "go2_weight_stand_hi_ppo.yaml|$ROOT/go2_weight_runs/E075_recal|$ROOT/go2_weight_runs/E075_recal/go2_weight_stand_hi_adv|--env-override hi=120" ;;
    getup1)         echo "go2_getup_ppo.yaml|$ROOT/go2_transition_runs|$ROOT/go2_transition_runs/go2_getup_adv|" ;;
    getup)          echo "go2_getup_ppo.yaml|$ROOT/go2_transition_runs/getup_v2|$ROOT/go2_transition_runs/getup_v2/go2_getup_adv|--load $ROOT/go2_transition_runs/go2_getup_adv/final_model.zip" ;;
    descend)        echo "go2_descend_ppo.yaml|$ROOT/go2_transition_runs/descend_v4|$ROOT/go2_transition_runs/descend_v4/go2_descend_adv|--load $ROOT/go2_weight_runs/go2_weight_rest_hi_adv/final_model.zip" ;;
    stand_hi_orig)  echo "go2_weight_stand_hi_ppo.yaml|$ROOT/go2_weight_runs|$ROOT/go2_weight_runs/go2_weight_stand_hi_adv|" ;;
    unified)        echo "go2_weight_unified_hi_ppo.yaml|$ROOT/go2_weight_runs|$ROOT/go2_weight_runs/go2_weight_unified_hi_adv|" ;;
    unified_disc)   echo "go2_weight_unified_disc_hi_ppo.yaml|$ROOT/go2_weight_runs|$ROOT/go2_weight_runs/go2_weight_unified_disc_hi_adv|" ;;
    leg_stand)      echo "go2_leg_stand_ppo.yaml|$ROOT/go2_leg_family_runs|$ROOT/go2_leg_family_runs/go2_leg_stand_adv|" ;;
    leg_rest)       echo "go2_leg_rest_ppo.yaml|$ROOT/go2_leg_family_runs|$ROOT/go2_leg_family_runs/go2_leg_rest_adv|" ;;
    compound_stand) echo "go2_compound_stand_ppo.yaml|$ROOT/go2_compound_runs|$ROOT/go2_compound_runs/go2_compound_stand_adv|" ;;
    compound_rest)  echo "go2_compound_rest_ppo.yaml|$ROOT/go2_compound_runs|$ROOT/go2_compound_runs/go2_compound_rest_adv|" ;;
    *) return 1 ;;
  esac
}

train() {
  local name="$1" s cfg out rundir flags load
  s="$(spec "$name")" || { echo "unknown policy: $name" >&2; exit 2; }
  IFS='|' read -r cfg out rundir flags <<< "$s"
  if [ -e "$rundir" ] && [ "${FORCE:-0}" != 1 ]; then
    echo "[train] $name: $rundir exists — skipping (FORCE=1 to retrain into it)"
    return 0
  fi
  load="$(sed -n 's/.*--load \([^ ]*\).*/\1/p' <<< "$flags")"
  if [ -n "$load" ] && [ ! -f "$load" ]; then
    echo "[train] $name needs its warm-start $load — train that policy first" >&2
    exit 1
  fi
  echo "[train] $name -> $rundir  (log: $ROOT/logs/$name.log)"
  # shellcheck disable=SC2086
  python external/robot-safety-sandbox/examples/train.py --config "$CFG/$cfg" --out "$out" $flags \
    "${EXTRA[@]}" > "$ROOT/logs/$name.log" 2>&1 || { echo "[train] $name FAILED — see $ROOT/logs/$name.log" >&2; exit 1; }
  echo "[train] $name done: $(grep -o 'safety/failure_rate *| *[0-9.]*' "$ROOT/logs/$name.log" | tail -1)"
}

[ $# -gt 0 ] || { sed -n 2,21p "$0"; exit 1; }
names=()
for a in "$@"; do
  case "$a" in
    core)         names+=(rest stand getup1 getup descend) ;;
    certificates) names+=(stand_hi_orig unified unified_disc leg_stand leg_rest compound_stand compound_rest) ;;
    all)          names+=(rest stand getup1 getup descend stand_hi_orig unified unified_disc leg_stand leg_rest compound_stand compound_rest) ;;
    *)            names+=("$a") ;;
  esac
done
for n in "${names[@]}"; do train "$n"; done
echo "[train] finished -> $ROOT"
