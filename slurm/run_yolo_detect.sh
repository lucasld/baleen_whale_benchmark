#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32GB
#SBATCH -c 8
#SBATCH -p klab-gpu
#SBATCH --gres=gpu
#SBATCH --job-name=detect_yolo
#SBATCH --error=slurm/outputs/logs/detect_yolo_%j.err
#SBATCH --output=slurm/outputs/logs/detect_yolo_%j.out

# gpu:H100.80gb:8

# #!/bin/bash
# #SBATCH --time=02:00:00
# #SBATCH --nodes=1
# #SBATCH --ntasks-per-node=1
# #SBATCH --mem=256G
# #SBATCH -c 4
# #SBATCH -p klab-l40s
# #SBATCH --gres=gpu:1




# Load required modules
spack load miniconda3

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate yoloenv

# Set base paths
export BASE_DIR="/share/klab/danthes/lliessduques/test/baleen_whale_benchmark"

# --- Script Logic ---

# Go to the base directory
cd "$BASE_DIR"

# Require a run directory id (e.g., 250901_013612)
if [ -z "$1" ]; then
  echo "Error: No run id provided."
  echo "Usage: sbatch $0 <run_id> [--first_only] [extra yolo_detect.py args]"
  echo "Examples:"
  echo "  sbatch $0 250901_013612 --conf_sweep --eval_conf 0.05"
  echo "  sbatch $0 250901_013612 --first_only --skip_train --conf_sweep"
  exit 1
fi

RUN_DIR="$BASE_DIR/outputs/cnn_results/$1"
if [ ! -d "$RUN_DIR" ]; then
  echo "Error: Run directory not found: $RUN_DIR"
  exit 1
fi

shift 1

# Parse optional flag --first_only and collect extra args for yolo_detect.py
FIRST_ONLY=0
EXTRA_ARGS=()
while (( "$#" )); do
  case "$1" in
    --first_only)
      FIRST_ONLY=1
      shift 1
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift 1
      ;;
  esac
done

echo "Using CNN run: $RUN_DIR"

# Discover folds under the run directory
mapfile -t FOLDS < <(ls -1d "$RUN_DIR"/fold_* 2>/dev/null | xargs -I{} basename {})
if [ ${#FOLDS[@]} -eq 0 ]; then
  echo "Error: No fold_* directories found under: $RUN_DIR"
  exit 1
fi

if [ "$FIRST_ONLY" -eq 1 ]; then
  FOLDS=("${FOLDS[0]}")
  echo "Test mode: processing only first fold: ${FOLDS[0]}"
else
  echo "Processing all folds: ${FOLDS[*]}"
fi

# Iterate folds; train/evaluate YOLO per fold
for FOLD_NAME in "${FOLDS[@]}"; do
  echo "\n=== YOLO on fold: $FOLD_NAME ==="
  python -u src/yolo/yolo_detect.py \
    --run_dir "$RUN_DIR" \
    --train_fold "$FOLD_NAME" \
    --weights yolo11n.pt \
    --epochs 50 \
    --batch 16 \
    --imgsz 128 \
    --device 0 \
    --name det \
    --skip_train \
    "${EXTRA_ARGS[@]}"
done

