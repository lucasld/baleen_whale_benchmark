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

# Enable verbose mode for debugging
set -x

# Create logs directory first (to capture all output)
mkdir -p ${SLURM_SUBMIT_DIR}/logs

# Record start time
start_time=$(date +%s)
echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "Current working directory: $(pwd)"
echo "Slurm job ID: ${SLURM_JOB_ID}"

# Check which modules are available
echo "Available modules:"
module avail 2>&1

# Try different approaches to load conda
echo "Attempting to load conda environment..."

# Method 1: Try loading with spack directly
echo "Method 1: Using spack load command"
if spack find miniconda3 &>/dev/null; then
    echo "miniconda3 found in spack, loading..."
    spack load miniconda3
elif spack find miniconda3@4.10.3 &>/dev/null; then
    echo "miniconda3@4.10.3 found in spack, loading..."
    spack load miniconda3@4.10.3
else
    echo "miniconda3 not found in spack using 'spack find'"
fi

# Method 2: Try using module load command if available
echo "Method 2: Using module load command"
if module avail miniconda3 &>/dev/null; then
    echo "miniconda3 module found, loading..."
    module load miniconda3
elif module avail anaconda &>/dev/null; then
    echo "anaconda module found, loading..."
    module load anaconda
else
    echo "No conda module found using 'module avail'"
fi

# Check if conda command is available
if command -v conda &>/dev/null; then
    echo "Conda is available in PATH: $(which conda)"
    eval "$(conda shell.bash hook)"
else
    echo "ERROR: conda command not found in PATH after attempting to load modules"
    echo "Available commands in PATH:"
    echo $PATH | tr ':' '\n'
    exit 1
fi

# Check if environment exists, if not create it
if ! conda env list | grep -q "whale_legacy"; then
    echo "Creating whale_legacy conda environment..."
    echo "Using environment file: ${SLURM_SUBMIT_DIR}/slurm/environment.yml"
    
    # Print the content of the environment file for debugging
    echo "Environment file content:"
    cat ${SLURM_SUBMIT_DIR}/slurm/environment.yml
    
    conda env create -f ${SLURM_SUBMIT_DIR}/slurm/environment.yml
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to create conda environment"
        exit 1
    fi
else
    echo "whale_legacy environment already exists."
fi

# Activate environment
echo "Activating whale_legacy environment..."
conda activate whale_legacy

# Check if activation was successful
if [ $? -ne 0 ] || [ "$CONDA_DEFAULT_ENV" != "whale_legacy" ]; then
    echo "ERROR: Failed to activate whale_legacy environment"
    echo "Current conda environment: $CONDA_DEFAULT_ENV"
    exit 1
fi

echo "Successfully activated conda environment: $CONDA_DEFAULT_ENV"
echo "Python version: $(python --version)"

# Set paths
RAW_DATA_PATH="/share/klab/danthes/data/ArcticWhales_AADC"
OUTPUT_PATH="/share/klab/danthes/data/ArcticWhales_AADC_processed"
SCRIPT_DIR="${SLURM_SUBMIT_DIR}/custom_preprocessing"
TAG_MAPPING="${SCRIPT_DIR}/tag_mapping.json"
CONFIG_PATH="${SLURM_SUBMIT_DIR}/config.json"

# Create output directory if it doesn't exist
mkdir -p ${OUTPUT_PATH}

# Print configuration
echo "Using the following configuration:"
echo "Raw Data: ${RAW_DATA_PATH}"
echo "Output Path: ${OUTPUT_PATH}"
echo "Script Directory: ${SCRIPT_DIR}"
echo "Tag Mapping: ${TAG_MAPPING}"
echo "Config Path: ${CONFIG_PATH}"

# Verify that the required files exist
echo "Checking if required files exist:"
for file in "${SCRIPT_DIR}/preprocess_workflow.py" "${TAG_MAPPING}" "${CONFIG_PATH}"; do
    if [ -f "$file" ]; then
        echo "  - $file exists"
    else
        echo "ERROR: $file does not exist"
        exit 1
    fi
done

# Change to the legacy CNN directory
cd ${SLURM_SUBMIT_DIR}
echo "Changed working directory to: $(pwd)"

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

# Disable verbose mode
set +x 