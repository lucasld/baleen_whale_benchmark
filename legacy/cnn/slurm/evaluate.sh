#!/bin/bash
#SBATCH --job-name=whale_evaluate
#SBATCH --output=eval_output_%j.txt
#SBATCH --error=eval_error_%j.txt
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=klab-gpu

export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
start_time=$(date +%s)
echo "Evaluation job started at: $(date)"

# Activate conda
spack load miniconda3@4.10.3
eval "$(conda shell.bash hook)"
conda activate baleen_whale_env

# Set paths - adjust these as needed
MODEL_PATH="/share/klab/$(whoami)/whale_output/models"
CONFIG_PATH="$(pwd)/legacy/cnn/config.json"

# Run evaluation
cd "$(pwd)"
python legacy/cnn/test.py <<< "$CONFIG_PATH"

# Report completion
end_time=$(date +%s)
echo "Evaluation job ended at: $(date)"
elapsed_time=$((end_time - start_time))
echo "Elapsed time: $(date -u -d @$elapsed_time +%H:%M:%S)" 