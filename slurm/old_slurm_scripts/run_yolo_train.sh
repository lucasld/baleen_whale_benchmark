#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=256G
#SBATCH -c 4
#SBATCH -p klab-l40s
#SBATCH --gres=gpu:1
#SBATCH --job-name=train_yolo
#SBATCH --error=slurm/outputs/logs/train_yolo_%j.err
#SBATCH --output=slurm/outputs/logs/train_yolo_%j.out

# Load required modules
spack load miniconda3

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate yoloenv

# Set base paths
export BASE_DIR="/share/klab/danthes/lliessduques/test/baleen_whale_benchmark"
# CONFIG_PATH will be set after MODEL_FOLDER is determined
# EVAL_OUTPUT_DIR will be set per-run after MODEL_FOLDER is known

# --- Script Logic ---

# Go to the base directory
cd "$BASE_DIR"

# Require a run directory id (e.g., 250901_013612)
if [ -z "$1" ]; then
  echo "Error: No run id provided."
  echo "Usage: sbatch $0 <run_id>"
  echo "Example: sbatch $0 250901_013612"
  exit 1
fi

RUN_DIR="$BASE_DIR/outputs/cnn_results/$1"
if [ ! -d "$RUN_DIR" ]; then
  echo "Error: Run directory not found: $RUN_DIR"
  exit 1
fi

echo "Using CNN run: $RUN_DIR"

# Train once on the available fold's train/val and evaluate on all present per-noise test sets
python -u src/yolo/yolo_test.py \
  --run_dir "$RUN_DIR" \
  --epochs 50 \
  --batch 32 \
  --imgsz 224 \
  --device 0
