#!/bin/bash
#SBATCH --job-name=whale_preprocess
#SBATCH --output=logs/whale_preprocess_%j.out
#SBATCH --error=logs/whale_preprocess_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=klab-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G

# Record start time
start_time=$(date +%s)
echo "Job started at: $(date)"

# Load required modules
spack load miniconda3@4.10.3
eval "$(conda shell.bash hook)"

# Check if environment exists, if not create it
if ! conda env list | grep -q "whale_legacy"; then
    echo "Creating whale_legacy conda environment..."
    conda env create -f ${SLURM_SUBMIT_DIR}/slurm/environment.yml
else
    echo "whale_legacy environment already exists."
fi

# Activate environment
conda activate whale_legacy

# Set paths
RAW_DATA_PATH="/share/klab/danthes/data/ArcticWhales_AADC"
OUTPUT_PATH="/share/klab/danthes/data/ArcticWhales_AADC_processed"
SCRIPT_DIR="${SLURM_SUBMIT_DIR}/custom_preprocessing"
TAG_MAPPING="${SCRIPT_DIR}/tag_mapping.json"
CONFIG_PATH="${SLURM_SUBMIT_DIR}/config.json"

# Create output directory if it doesn't exist
mkdir -p ${OUTPUT_PATH}
mkdir -p ${SLURM_SUBMIT_DIR}/logs

# Print configuration
echo "Using the following configuration:"
echo "Raw Data: ${RAW_DATA_PATH}"
echo "Output Path: ${OUTPUT_PATH}"
echo "Script Directory: ${SCRIPT_DIR}"
echo "Tag Mapping: ${TAG_MAPPING}"
echo "Config Path: ${CONFIG_PATH}"

# Change to the legacy CNN directory
cd ${SLURM_SUBMIT_DIR}

# Run the preprocessing workflow script
echo "Running preprocessing workflow..."
python ${SCRIPT_DIR}/preprocess_workflow.py \
    --raw_data ${RAW_DATA_PATH} \
    --output ${OUTPUT_PATH} \
    --config ${CONFIG_PATH} \
    --tag_mapping ${TAG_MAPPING}

# Check if the job completed successfully
if [ $? -eq 0 ]; then
    echo "Preprocessing completed successfully."
else
    echo "Preprocessing failed with exit code $?."
    # Deactivate environment
    conda deactivate
    exit 1
fi

# Calculate and print elapsed time
end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "Job ended at: $(date)"
echo "Elapsed time: $(date -u -d @${elapsed_time} +%H:%M:%S)"

# Deactivate environment
conda deactivate 