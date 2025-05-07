#!/bin/bash
#SBATCH --job-name=whale_train
#SBATCH --output=train_output_%j.txt
#SBATCH --error=train_error_%j.txt
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=klab-gpu

export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
start_time=$(date +%s)
echo "Training job started at: $(date)"

# Activate conda
spack load miniconda3@4.10.3
eval "$(conda shell.bash hook)"
conda activate baleen_whale_env

# Set paths
CONFIG_PATH="$(pwd)/legacy/cnn/config.json"

# Run training
cd "$(pwd)"
python legacy/cnn/train.py <<< "$CONFIG_PATH"

# Report completion
end_time=$(date +%s)
echo "Training job ended at: $(date)"
elapsed_time=$((end_time - start_time))
echo "Elapsed time: $(date -u -d @$elapsed_time +%H:%M:%S)" 