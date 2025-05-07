#!/bin/bash
#SBATCH --job-name=whale_preprocess
#SBATCH --output=preprocess_output_%j.txt
#SBATCH --error=preprocess_error_%j.txt
#SBATCH --time=12:00:00
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH --partition=klab-cpu

export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
start_time=$(date +%s)
echo "Preprocessing job started at: $(date)"

# Activate conda
spack load miniconda3@4.10.3
eval "$(conda shell.bash hook)"
conda activate baleen_whale_env

# Set paths
RAW_DATA_PATH="/share/klab/danthes/data/ArcticWhales_AADC"
OUTPUT_DIR="/share/klab/$(whoami)/whale_output"
CONFIG_PATH="$(pwd)/legacy/cnn/config.json"
TAG_MAPPING_PATH="$(pwd)/legacy/cnn/custom_preprocessing/tag_mapping.json"

# Create output directory if not exists
mkdir -p "$OUTPUT_DIR"

# Run the preprocessing workflow
cd "$(pwd)"
python legacy/cnn/custom_preprocessing/preprocess_workflow.py \
  --raw_data "$RAW_DATA_PATH" \
  --output "$OUTPUT_DIR" \
  --config "$CONFIG_PATH" \
  --tag_mapping "$TAG_MAPPING_PATH"

# Report completion
end_time=$(date +%s)
echo "Preprocessing job ended at: $(date)"
elapsed_time=$((end_time - start_time))
echo "Elapsed time: $(date -u -d @$elapsed_time +%H:%M:%S)" 