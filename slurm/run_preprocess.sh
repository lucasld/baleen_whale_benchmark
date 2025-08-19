#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH -c 4
#SBATCH -p klab-cpu
#SBATCH --job-name=preprocess_stage2
#SBATCH --error=slurm/outputs/logs/preprocess_%j.err
#SBATCH --output=slurm/outputs/logs/preprocess_%j.out

# Load required modules
spack load miniconda3

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate whale_env

# Set paths
export BASE_DIR="/share/klab/danthes/lliessduques/test/baleen_whale_benchmark"
export RAW_DATA_PATH="${BASE_DIR}/datasets/prepared_dataset"
export OUTPUT_DIR="${BASE_DIR}/datasets/preprocessed_dataset_new4"
export CONFIG_PATH="${BASE_DIR}/src/config.json"
export TAG_MAPPING_PATH="${BASE_DIR}/src/custom_preprocessing/file_tag_mapping.json"

# Add custom_preprocessing to Python path
export PYTHONPATH="${BASE_DIR}/src/custom_preprocessing:${PYTHONPATH}"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run preprocessing script
cd "$BASE_DIR"
python -u src/custom_preprocessing/preprocess_workflow.py \
  --raw_data "$RAW_DATA_PATH" \
  --output "$OUTPUT_DIR" \
  --config "$CONFIG_PATH" \
  --tag_mapping "$TAG_MAPPING_PATH" \
  --skip_koogu \
  --save_yolo_labels \
  --preview_stride 200 \
  --preview_dir "$OUTPUT_DIR/annotated_specs" \
  --yolo_labels_dir "$OUTPUT_DIR/spectrograms_labels"

