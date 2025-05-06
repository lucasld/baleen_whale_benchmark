#!/bin/bash
#SBATCH --job-name=whale_cnn_train
#SBATCH --output=logs/whale_cnn_train_%j.out
#SBATCH --error=logs/whale_cnn_train_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=klab-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
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

# Try loading CUDA modules
echo "Loading CUDA modules..."
if spack find cuda@11.8.0 &>/dev/null; then
    echo "CUDA 11.8.0 found in spack, loading..."
    spack load cuda@11.8.0
else
    echo "CUDA 11.8.0 not found in spack, trying module load..."
    module load cuda/11.8.0 2>/dev/null || echo "Failed to load CUDA via module"
fi

if spack find cudnn@8.6.0.163-11.8 &>/dev/null; then
    echo "cuDNN 8.6.0.163-11.8 found in spack, loading..."
    spack load cudnn@8.6.0.163-11.8
else
    echo "cuDNN 8.6.0.163-11.8 not found in spack, trying module load..."
    module load cudnn/8.6.0 2>/dev/null || echo "Failed to load cuDNN via module"
fi

# Method 2: Try using module load command if available
echo "Method 2: Using module load command for conda"
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
echo "TensorFlow version: $(python -c 'import tensorflow as tf; print(tf.__version__)')"

# Configure GPU environment variables
echo "Setting up GPU environment variables..."
mkdir -p $CONDA_PREFIX/lib/nvvm/libdevice/
cp -p $CONDA_PREFIX/lib/libdevice.10.bc $CONDA_PREFIX/lib/nvvm/libdevice/ 2>/dev/null || true
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX

# Set temporary directory to a location with sufficient space
export TMPDIR="/share/klab/danthes/tmp"
mkdir -p $TMPDIR

# Set CONFIG_PATH and make sure DATA_DIR in config points to processed data
CONFIG_PATH="${SLURM_SUBMIT_DIR}/config.json"

# Verify that the config file exists
if [ ! -f "$CONFIG_PATH" ]; then
    echo "ERROR: Config file not found: $CONFIG_PATH"
    exit 1
else
    echo "Config file exists: $CONFIG_PATH"
    echo "Config file content:"
    cat $CONFIG_PATH
fi

# Change to the legacy CNN directory
cd ${SLURM_SUBMIT_DIR}
echo "Changed working directory to: $(pwd)"

# Print configuration
echo "Using config file: ${CONFIG_PATH}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "GPU devices available:"
nvidia-smi

# Verify tensorflow can see the GPU
echo "Checking if TensorFlow can see the GPU:"
python -c "import tensorflow as tf; print('Num GPUs Available:', len(tf.config.list_physical_devices('GPU')), tf.config.list_physical_devices('GPU'))"

# Run the training script
echo "Starting CNN training..."
python train.py ${CONFIG_PATH}

# Check if the job completed successfully
if [ $? -eq 0 ]; then
    echo "Training completed successfully."
else
    echo "Training failed with exit code $?."
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