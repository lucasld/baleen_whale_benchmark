#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=256G
#SBATCH -c 4
#SBATCH -p klab-l40s
#SBATCH --gres=gpu:1
#SBATCH --job-name=train_cnn
#SBATCH --error=slurm/outputs/logs/train_cnn_%j.err
#SBATCH --output=slurm/outputs/logs/train_cnn_%j.out

# Load required modules
spack load miniconda3

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate whale_env2

# Set paths
export BASE_DIR="/share/klab/danthes/lliessduques/test/baleen_whale_benchmark"
export CONFIG_PATH="${BASE_DIR}/src/config.json"

# Go to the base directory
cd "$BASE_DIR"

# Create output directory if it doesn't exist
mkdir -p "${BASE_DIR}/outputs/cnn_results"

# Record start time and memory usage
echo "Job started at: $(date)"
echo "Initial memory usage:"
free -h

# Run the training script by calling the run_from_config function directly.
# This avoids the interactive input() in train.py's __main__ block.
# Make sure the "DATA_DIR" in your config.json points to the correct spectrograms folder.
# TODO: Added environment and provenance printouts to aid result traceability in .out
python -u -c "import sys, json, platform, tensorflow as tf, subprocess; sys.path.insert(0, 'src'); import train; import pathlib; print('Starting training with config: ${CONFIG_PATH}'); print(f'Python: {platform.python_version()} | TF: {tf.__version__}');
print('GPU(s):', subprocess.getoutput('nvidia-smi --query-gpu=name,memory.total --format=csv,noheader'));
train.run_from_config(pathlib.Path('${CONFIG_PATH}'))"


echo "Training job finished."

# Record end time and memory usage
echo "Job ended at: $(date)"
echo "Final memory usage:"
free -h

# Get the job ID from the environment
JOB_ID=$SLURM_JOB_ID

# Print detailed job statistics - will be available after job completes
echo "To see detailed job statistics after this job completes, run:"
echo "sacct -j $JOB_ID --format=JobID,JobName,MaxRSS,MaxVMSize,AveRSS,AveVMSize"
echo "This will show the maximum and average memory usage of your job." 