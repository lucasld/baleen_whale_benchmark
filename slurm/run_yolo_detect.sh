#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=256G
#SBATCH -c 4
#SBATCH -p klab-l40s
#SBATCH --gres=gpu:1
#SBATCH --job-name=detect_yolo
#SBATCH --error=slurm/outputs/logs/detect_yolo_%j.err
#SBATCH --output=slurm/outputs/logs/detect_yolo_%j.out

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
  echo "Usage: sbatch $0 <run_id> [extra_yolo_detect_args]"
  echo "Example: sbatch $0 250901_013612 --epochs 50 --batch 32 --imgsz 512"
  exit 1
fi

RUN_DIR="$BASE_DIR/outputs/cnn_results/$1"
if [ ! -d "$RUN_DIR" ]; then
  echo "Error: Run directory not found: $RUN_DIR"
  exit 1
fi

shift 1  # pass any remaining args to yolo_detect.py

echo "Using CNN run: $RUN_DIR"

# Train detector on the selected fold and evaluate across per-noise test sets.
# The script auto-runs label cleaning (idempotent) before building the detection dataset.
python -u src/yolo/yolo_detect.py \
  --run_dir "$RUN_DIR" \
  --weights yolo11n.pt \
  --epochs 50 \
  --batch 32 \
  --imgsz 512 \
  --device 0 \
  --name det \
  --overwrite \
  "$@"

