#!/bin/bash
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=4G
#SBATCH -c 2
#SBATCH -p klab-cpu
#SBATCH --job-name=extract_tags
#SBATCH --error=slurm/outputs/logs/extract_tags_%j.err
#SBATCH --output=slurm/outputs/logs/extract_tags_%j.out

# Load required modules
spack load miniconda3

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate whale_env

# Set paths
export BASE_DIR="/share/klab/danthes/lliessduques/baleen_whale_benchmark/legacy/cnn"
export RAW_DATA_PATH="/share/klab/danthes/data/ArcticWhales_AADC"
export OUTPUT_DIR="${BASE_DIR}/slurm/outputs"

# Add custom_preprocessing to Python path
export PYTHONPATH="${BASE_DIR}/custom_preprocessing:${PYTHONPATH}"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run tag extraction script
cd "$BASE_DIR"
python custom_preprocessing/extract_tags.py \
  --raw_data "$RAW_DATA_PATH" \
  --output "$OUTPUT_DIR" 