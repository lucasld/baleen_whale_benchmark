#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32GB
#SBATCH -c 2
#SBATCH -p gpu
#SBATCH --job-name=eval_cnn
#SBATCH --error=slurm/outputs/logs/eval_cnn_%j.err
#SBATCH --output=slurm/outputs/logs/eval_cnn_%j.out

# This script runs the test and evaluation steps on a trained model.
#
# Usage: sbatch run_evaluation.sh <model_directory_name>
#
# Example: sbatch run_evaluation.sh 250626_134153
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
# CONFIG_PATH will be set after MODEL_FOLDER is determined
# EVAL_OUTPUT_DIR will be set per-run after MODEL_FOLDER is known

# --- Script Logic ---

# Go to the base directory
cd "$BASE_DIR"

# Check if a model directory name was provided as an argument
if [ -z "$1" ]; then
    echo "Error: No model directory specified."
    echo "Usage: sbatch $0 <model_directory_name>"
    echo "Example: sbatch $0 250626_134153"
    exit 1
fi

# Construct the full path to the model folder
MODEL_FOLDER="${BASE_DIR}/outputs/cnn_results/$1"

if [ ! -d "$MODEL_FOLDER" ]; then
    echo "Error: Model directory not found at: ${MODEL_FOLDER}"
    exit 1
fi

echo "Using model folder: $MODEL_FOLDER"

# Set the config path to use the saved config from the training run
export CONFIG_PATH="${MODEL_FOLDER}/config.json"

# Check if the saved config exists
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Saved config file not found at: ${CONFIG_PATH}"
    echo "Falling back to global config..."
    export CONFIG_PATH="${BASE_DIR}/src/config.json"
else
    echo "Using saved config from training run: $CONFIG_PATH"
fi

# Set evaluation output directory inside the specific run folder
export EVAL_OUTPUT_DIR="${MODEL_FOLDER}/evaluation"

# Create evaluation output directory if it doesn't exist
mkdir -p "$EVAL_OUTPUT_DIR"

# Record start time and memory usage
echo "Job started at: $(date)"
free -h

python -u src/new_test.py \
  --model_folder "$MODEL_FOLDER" \
  --config "$CONFIG_PATH" \
  --output_dir "$EVAL_OUTPUT_DIR"

# --- Job Completion ---
echo "Job ended at: $(date)"
free -h

JOB_ID=$SLURM_JOB_ID
echo "To see detailed job statistics, run:"
echo "sacct -j $JOB_ID --format=JobID,JobName,MaxRSS,AveRSS" 