#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32GB
#SBATCH -c 8
#SBATCH -p klab-gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=yolo_experiments
#SBATCH --error=slurm/outputs/logs/yolo_experiments_%A_%a.err
#SBATCH --output=slurm/outputs/logs/yolo_experiments_%A_%a.out

# Usage:
#   sbatch slurm/run_yolo_experiments.sh 251008_160341 --ids R0a,R0b
# For concurrent runs with arrays:
#   sbatch --array=1-2%2 slurm/run_yolo_experiments.sh 251008_160341 --ids R0a,R0b
#   sbatch --array=1-11%3 slurm/run_yolo_experiments.sh 251008_160341 --ids F1
#     (F1 has no fixed fold, so it expands to fold-level tasks)

# ---- Env setup (mirror run_yolo_detect.sh) ----
spack load miniconda3

eval "$(conda shell.bash hook)"
conda activate yoloenv

export BASE_DIR="/share/klab/danthes/lliessduques/test/baleen_whale_benchmark"
cd "$BASE_DIR" || exit 1

# ---- Parse positional run_id ----
if [ -z "$1" ]; then
  echo "Error: No run id provided."
  echo "Usage: sbatch [--array=...] $0 <run_id> --ids ID1,ID2,... [--extra \"...\"]"
  exit 1
fi

RUN_ID="$1"
RUN_DIR="$BASE_DIR/outputs/cnn_results/$RUN_ID"
if [ ! -d "$RUN_DIR" ]; then
  echo "Error: Run directory not found: $RUN_DIR"
  exit 1
fi
shift 1

# ---- Parse script arguments ----
IDS=""
EXTRA_ARGS_STR=""

while (( "$#" )); do
  case "$1" in
    --ids)
      IDS="$2"
      shift 2
      ;;
    --extra)
      EXTRA_ARGS_STR="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument to run_yolo_experiments.sh: $1"
      exit 1
      ;;
  esac
done

if [ -z "$IDS" ]; then
  echo "Error: --ids ID1,ID2,... is required."
  exit 1
fi

echo "Using CNN run: $RUN_DIR"
echo "Requested experiment IDs: $IDS"

# Discover folds under run directory for expansion/validation
mapfile -t FOLDS < <(ls -1d "$RUN_DIR"/fold_* 2>/dev/null | xargs -I{} basename {})
if [ ${#FOLDS[@]} -eq 0 ]; then
  echo "Error: No fold_* directories found under: $RUN_DIR"
  exit 1
fi

# Expand requested experiment IDs into concrete (exp_id, fold) tasks:
# - If registry has exp.fold -> one task on that fold
# - Otherwise -> one task per detected fold_*
mapfile -t TASKS < <(
  python - "$RUN_DIR" "$IDS" <<'PY'
import sys
from pathlib import Path
import yaml

run_dir = Path(sys.argv[1])
ids = [x.strip() for x in sys.argv[2].split(",") if x.strip()]
registry_path = Path("experiments/registry.yaml")

if not registry_path.exists():
    raise SystemExit(f"Registry not found: {registry_path}")

with registry_path.open("r") as f:
    reg = yaml.safe_load(f) or {}

experiments = {e["id"]: e for e in reg.get("experiments", []) if isinstance(e, dict) and "id" in e}
missing = [eid for eid in ids if eid not in experiments]
if missing:
    raise SystemExit(f"Experiment IDs not found in registry: {missing}")

fold_dirs = sorted([p.name for p in run_dir.glob("fold_*") if p.is_dir()])
if not fold_dirs:
    raise SystemExit(f"No fold_* directories found under: {run_dir}")
fold_set = set(fold_dirs)

for eid in ids:
    exp = experiments[eid]
    fold = exp.get("fold")
    if fold:
        if fold not in fold_set:
            raise SystemExit(f"Registry fold '{fold}' for ID '{eid}' not found under {run_dir}")
        print(f"{eid}\t{fold}")
    else:
        for fold_name in fold_dirs:
            print(f"{eid}\t{fold_name}")
PY
)

NUM_TASKS=${#TASKS[@]}
if [ "$NUM_TASKS" -eq 0 ]; then
  echo "No runnable tasks after expansion."
  exit 1
fi

echo "Expanded to $NUM_TASKS task(s):"
for t in "${TASKS[@]}"; do
  IFS=$'\t' read -r exp_id fold_name <<< "$t"
  echo "  - $exp_id :: $fold_name"
done
echo "Array hint: use --array=1-${NUM_TASKS}%<concurrency>"
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
  echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID (1-based), total tasks: $NUM_TASKS"
fi

# Helper to run one concrete (experiment, fold) task via tools/run_experiment.py
run_one_task() {
  local exp_id="$1"
  local fold_name="$2"

  CMD=(
    python -u tools/run_experiment.py
    --run-dir "outputs/cnn_results/$RUN_ID"
    --ids "$exp_id"
    --fold-override "$fold_name"
  )

  if [ -n "$EXTRA_ARGS_STR" ]; then
    # EXTRA_ARGS_STR is a single string, we need to split it safely
    # shellcheck disable=SC2206
    EXTRA_TOKENS=($EXTRA_ARGS_STR)
    CMD+=("${EXTRA_TOKENS[@]}")
  fi

  echo "[run_yolo_experiments] Running task: $exp_id :: $fold_name"
  echo "[run_yolo_experiments] Command: ${CMD[*]}"
  "${CMD[@]}"
}

# ---- Array-aware dispatch ----
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
  # 1-based index from SLURM
  idx=$((SLURM_ARRAY_TASK_ID - 1))
  if [ "$idx" -lt 0 ] || [ "$idx" -ge "$NUM_TASKS" ]; then
    echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID out of range for $NUM_TASKS tasks; exiting."
    exit 1
  fi
  THIS_TASK="${TASKS[$idx]}"
  IFS=$'\t' read -r THIS_ID THIS_FOLD <<< "$THIS_TASK"
  run_one_task "$THIS_ID" "$THIS_FOLD"
else
  # No array: run all expanded tasks sequentially
  for task in "${TASKS[@]}"; do
    IFS=$'\t' read -r exp_id fold_name <<< "$task"
    run_one_task "$exp_id" "$fold_name"
  done
fi