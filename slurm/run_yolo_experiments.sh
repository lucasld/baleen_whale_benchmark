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
#   sbatch --array=1-3%3 slurm/run_yolo_experiments.sh 251008_160341 --ids A1,A2,A3

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

# Split comma-separated IDs into array
IFS=',' read -r -a EXP_IDS <<< "$IDS"
NUM_IDS=${#EXP_IDS[@]}

echo "Using CNN run: $RUN_DIR"
echo "Experiment IDs: ${EXP_IDS[*]}"
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
  echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID (1-based), total IDs: $NUM_IDS"
fi

# Helper to run a single experiment ID via tools/run_experiment.py
run_one_experiment() {
  local exp_id="$1"

  CMD=(
    python -u tools/run_experiment.py
    --run-dir "outputs/cnn_results/$RUN_ID"
    --ids "$exp_id"
  )

  if [ -n "$EXTRA_ARGS_STR" ]; then
    # EXTRA_ARGS_STR is a single string, we need to split it safely
    # shellcheck disable=SC2206
    EXTRA_TOKENS=($EXTRA_ARGS_STR)
    CMD+=("${EXTRA_TOKENS[@]}")
  fi

  echo "[run_yolo_experiments] Running experiment ID: $exp_id"
  echo "[run_yolo_experiments] Command: ${CMD[*]}"
  "${CMD[@]}"
}

# ---- Array-aware dispatch ----
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
  # 1-based index from SLURM
  idx=$((SLURM_ARRAY_TASK_ID - 1))
  if [ "$idx" -lt 0 ] || [ "$idx" -ge "$NUM_IDS" ]; then
    echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID out of range for $NUM_IDS experiment IDs; exiting."
    exit 1
  fi
  THIS_ID="${EXP_IDS[$idx]}"
  run_one_experiment "$THIS_ID"
else
  # No array: run all experiments sequentially
  for exp_id in "${EXP_IDS[@]}"; do
    run_one_experiment "$exp_id"
  done
fi