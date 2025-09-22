#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32GB
#SBATCH -c 2
#SBATCH -p gpu
#SBATCH --job-name=dataset_yolo
#SBATCH --error=slurm/outputs/logs/dataset_yolo%j.err
#SBATCH --output=slurm/outputs/logs/dataset_yolo%j.out

# This script recreates YOLO datasets for a trained model run.
#
# Usage: sbatch run_yolo_dataset.sh <model_directory_name>
#
# Example: sbatch run_yolo_dataset.sh 250920_021045
#
# The <model_directory_name> is the timestamped folder created during training,
# located in 'outputs/cnn_results/'.

# --- Configuration ---

# Load required modules
spack load miniconda3

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate whale_env2

# Set base paths
export BASE_DIR="/share/klab/danthes/lliessduques/test/baleen_whale_benchmark"

# --- Script Logic ---

# Go to the base directory
cd "$BASE_DIR"

# Check if a model directory name was provided as an argument
if [ -z "$1" ]; then
    echo "Error: No model directory specified."
    echo "Usage: sbatch $0 <model_directory_name>"
    echo "Example: sbatch $0 250920_021045"
    exit 1
fi

# Construct the full path to the model folder
MODEL_FOLDER="${BASE_DIR}/outputs/cnn_results/$1"

# Run yolo dataset script
python -u src/yolo/recreate_yolo_labels.py "$MODEL_FOLDER"