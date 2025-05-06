#!/bin/bash
#SBATCH --job-name=setup_env
#SBATCH --output=setup_env_%j.out
#SBATCH --error=setup_env_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH -p klab-cpu

echo "Starting environment setup at: $(date)"

# Load necessary modules
spack load miniconda3@4.10.3
eval "$(conda shell.bash hook)"

# Create conda environment from the YAML file
REPO_ROOT=$(dirname $(dirname $(dirname $(dirname $(realpath $0)))))
ENV_FILE="$REPO_ROOT/legacy/cnn/environment/environment.yml"

echo "Creating conda environment from: $ENV_FILE"
conda env create -f "$ENV_FILE" -n whale_cnn_legacy

echo "Environment setup completed at: $(date)"
echo "To use this environment, run: conda activate whale_cnn_legacy" 