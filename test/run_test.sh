#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32GB
#SBATCH -c 8
#SBATCH -p klab-gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=yolo_custom_test
#SBATCH --error=test/logs/test_%j.err
#SBATCH --output=test/logs/test_%j.out

# ---- Env setup ----
spack load miniconda3
eval "$(conda shell.bash hook)"
conda activate yoloenv

export BASE_DIR="/share/klab/danthes/lliessduques/test/baleen_whale_benchmark"
cd "$BASE_DIR" || exit 1

# ---- Run the test script ----
# Usage: sbatch test/run_test.sh <run_id> <fold_name>
# Example: sbatch test/run_test.sh 251008_160341 fold_BallenyIslands2015_noise_0.25

RUN_ID="$1"
FOLD_NAME="$2"

if [ -z "$RUN_ID" ] || [ -z "$FOLD_NAME" ]; then
  echo "Usage: sbatch $0 <run_id> <fold_name>"
  exit 1
fi

python -u test/test_custom_trainer.py \
    --run_dir "outputs/cnn_results/$RUN_ID" \
    --train_fold "$FOLD_NAME"
