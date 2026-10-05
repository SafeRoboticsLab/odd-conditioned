#!/usr/bin/env bash
# Train the policies from scratch with safety-stable-baselines on the robot-safety-sandbox tasks.
#
#   bash scripts/train.sh core            # rest, stand, getup_stage1, getup, descend: the automaton   ~2 h
#   bash scripts/train.sh certificates    # stand_wide, unified, unified_discounted, leg_stand,
#                                         # compound_stand, compound_rest: the certificate figures      ~2.5 h
#   bash scripts/train.sh all             # both
#   bash scripts/train.sh rest stand ...  # single policies (names: odd_conditioned/policies.py)
#
# Each policy trains into $ODD_RUNS/<name>/ (default runs/) and its final-step checkpoint is then INSTALLED
# into $ODD_CHECKPOINTS/<name>/{model.zip, tensornormalize.pt, config.yaml} (default checkpoints/) — where
# every evaluation looks. getup and descend warm-start from the installed final models of getup_stage1 and
# rest ($ODD_CHECKPOINTS/<src>/final/), so train those first ("core" does it in order).
#
# Keep a retrain apart from the published weights:
#   ODD_CHECKPOINTS=checkpoints_retrained bash scripts/train.sh core
#   ODD_CHECKPOINTS=checkpoints_retrained python scripts/check_policies.py stand checkpoints_retrained/stand/model.zip
#   ODD_CHECKPOINTS=checkpoints_retrained python scripts/calibrate.py
#
# Variance: the stance-type policies (stand, compound_stand) vary a lot from seed to seed. Train a few seeds
# (SEED=1 bash scripts/train.sh stand ...) and keep one that passes scripts/check_policies.py.
#
# Environment knobs:
#   SEED=<k>     training seed (default 0, the published runs)
#   ODD_WANDB=1  log to wandb (project odd-conditioned); default off
#   FORCE=1      overwrite an existing run directory / installed checkpoint (default: refuse)
#   SMOKE=1      300k steps x 256 envs per policy into a scratch directory: checks the commands only
# Runtimes: one RTX 4070, one run at a time (14-25 min per policy).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
if [ -z "${VIRTUAL_ENV:-}" ] || [[ "${PYTHONPATH:-}" != *external/robot-safety-sandbox* ]]; then
  # shellcheck disable=SC1091
  source "$REPO/activate.sh" >/dev/null
fi

CFG=external/robot-safety-sandbox/configs
EXTRA=(--seed "${SEED:-0}")
if [ "${SMOKE:-0}" = 1 ]; then
  scratch="$(mktemp -d)"
  RUNS="$scratch/runs"; CKPT="$scratch/checkpoints"
  EXTRA+=(--steps 300000 --num-envs 256 --no-wandb)
  echo "[train] SMOKE -> $scratch"
else
  RUNS="${ODD_RUNS:-$REPO/runs}"; CKPT="${ODD_CHECKPOINTS:-$REPO/checkpoints}"
  [ "${ODD_WANDB:-0}" = 1 ] || EXTRA+=(--no-wandb)
fi
mkdir -p "$RUNS/logs" "$CKPT"

# name -> "config | task | extra flags | warm-start source (installed final model)"
spec() {
  case "$1" in
    rest)               echo "go2_weight_rest_hi_ppo.yaml|go2_weight_rest_hi||" ;;
    stand)              echo "go2_weight_stand_hi_ppo.yaml|go2_weight_stand_hi|--env-override hi=120|" ;;
    getup_stage1)       echo "go2_getup_ppo.yaml|go2_getup||" ;;
    getup)              echo "go2_getup_ppo.yaml|go2_getup||getup_stage1" ;;
    descend)            echo "go2_descend_ppo.yaml|go2_descend||rest" ;;
    stand_wide)         echo "go2_weight_stand_hi_ppo.yaml|go2_weight_stand_hi||" ;;
    unified)            echo "go2_weight_unified_hi_ppo.yaml|go2_weight_unified_hi||" ;;
    unified_discounted) echo "go2_weight_unified_disc_hi_ppo.yaml|go2_weight_unified_disc_hi||" ;;
    leg_stand)          echo "go2_leg_stand_ppo.yaml|go2_leg_stand||" ;;
    compound_stand)     echo "go2_compound_stand_ppo.yaml|go2_compound_stand||" ;;
    compound_rest)      echo "go2_compound_rest_ppo.yaml|go2_compound_rest||" ;;
    *) return 1 ;;
  esac
}
WARMSTART_SOURCES=" rest getup_stage1 "

install() {   # install <name> <run dir>: final-step checkpoint -> $CKPT/<name>/
  local name="$1" run="$2" dst="$CKPT/$1" zip norm
  read -r zip norm < <(python - "$run" <<'PY'
import glob, os, re, sys
from robot_safety_sandbox.eval.policies import find_obs_stats
run = sys.argv[1]
zips = glob.glob(f"{run}/checkpoints/model_*_steps.zip")
if zips:   # the last periodic checkpoint (what the published policies are)
    z = max(zips, key=lambda f: int(re.search(r"model_(\d+)_steps", f).group(1)))
    print(z, find_obs_stats(z, "tensornorm", "tensornormalize.pt"))
else:      # a run shorter than the checkpoint interval (e.g. SMOKE=1): its final model
    print(f"{run}/final_model.zip", f"{run}/tensornormalize.pt")
PY
)
  [ -f "$zip" ] && [ -f "$norm" ] || { echo "[train] $name: no checkpoint found in $run" >&2; exit 1; }
  rm -rf "$dst"; mkdir -p "$dst"
  cp "$zip" "$dst/model.zip"; cp "$norm" "$dst/tensornormalize.pt"; cp "$run/config.yaml" "$dst/config.yaml"
  if [[ "$WARMSTART_SOURCES" == *" $name "* ]]; then
    mkdir -p "$dst/final"
    cp "$run/final_model.zip" "$dst/final/final_model.zip"; cp "$run/tensornormalize.pt" "$dst/final/tensornormalize.pt"
  fi
  echo "[train] $name installed -> $dst ($(basename "$zip"))"
}

train() {
  local name="$1" s cfg task flags src run load=()
  s="$(spec "$name")" || { echo "unknown policy: $name" >&2; exit 2; }
  IFS='|' read -r cfg task flags src <<< "$s"
  run="$RUNS/$name/${task}_adv"
  if [ "${FORCE:-0}" != 1 ]; then
    if [ -e "$CKPT/$name/model.zip" ]; then
      echo "[train] $name: $CKPT/$name is installed — skipping (FORCE=1 to retrain and replace it)"; return 0
    fi
    if [ -e "$run" ]; then
      echo "[train] $name: $run exists — skipping (FORCE=1 to retrain into it)"; return 0
    fi
  fi
  if [ -n "$src" ]; then
    if [ ! -f "$CKPT/$src/final/final_model.zip" ]; then
      echo "[train] $name warm-starts from $CKPT/$src/final/final_model.zip — train '$src' first" >&2; exit 1
    fi
    load=(--load "$CKPT/$src/final/final_model.zip")
  fi
  echo "[train] $name -> $run  (log: $RUNS/logs/$name.log)"
  # shellcheck disable=SC2086
  python external/robot-safety-sandbox/examples/train.py --config "$CFG/$cfg" --out "$RUNS/$name" $flags \
    "${load[@]}" "${EXTRA[@]}" > "$RUNS/logs/$name.log" 2>&1 \
    || { echo "[train] $name FAILED — see $RUNS/logs/$name.log" >&2; exit 1; }
  install "$name" "$run"
}

[ $# -gt 0 ] || { sed -n 2,31p "$0"; exit 1; }
names=()
for a in "$@"; do
  case "$a" in
    core)         names+=(rest stand getup_stage1 getup descend) ;;
    certificates) names+=(stand_wide unified unified_discounted leg_stand compound_stand compound_rest) ;;
    all)          names+=(rest stand getup_stage1 getup descend stand_wide unified unified_discounted leg_stand
                          compound_stand compound_rest) ;;
    *)            names+=("$a") ;;
  esac
done
for n in "${names[@]}"; do train "$n"; done
echo "[train] finished — installed in $CKPT. Next: scripts/check_policies.py, then scripts/calibrate.py"
