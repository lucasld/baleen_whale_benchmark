#!/bin/bash
#SBATCH --job-name=setup_env
#SBATCH --output=setup_env_%j.out
#SBATCH --error=setup_env_%j.err
#SBATCH --time=10:00
#SBATCH --partition=klab-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

# Enable verbose mode for debugging
set -x

echo "Starting environment setup at: $(date)"
echo "Running on host: $(hostname)"
echo "Current working directory: $(pwd)"
echo "Slurm job ID: ${SLURM_JOB_ID}"

# Display environment variables
echo "Environment variables:"
env | sort

# Check which modules are available
echo "Available modules:"
module avail 2>&1

# Attempt all possible ways to find conda
echo "Searching for conda installations:"

# Method 1: Check if conda is in PATH
if command -v conda &>/dev/null; then
    echo "Conda found in PATH: $(which conda)"
else
    echo "Conda not found in PATH"
fi

# Method 2: Try loading with spack
echo "Attempting to load conda via spack:"
if command -v spack &>/dev/null; then
    echo "Spack is available: $(which spack)"
    echo "Available spack packages:"
    spack find || echo "Failed to list spack packages"
    
    echo "Looking for miniconda3 in spack:"
    spack find miniconda3 || echo "miniconda3 not found in spack"
    
    echo "Attempting to load miniconda3 from spack:"
    spack load miniconda3 || echo "Failed to load miniconda3 from spack"
else
    echo "Spack command not found"
fi

# Method 3: Try module load
echo "Attempting to load via module command:"
if command -v module &>/dev/null; then
    echo "Module command is available: $(which module)"
    
    echo "Checking for conda modules:"
    module avail conda miniconda anaconda 2>&1 || echo "No conda modules found"
    
    echo "Trying to load any available conda module:"
    module load miniconda3 2>/dev/null || module load anaconda 2>/dev/null || echo "Failed to load any conda module"
else
    echo "Module command not found"
fi

# Check again if conda is available after loading attempts
if command -v conda &>/dev/null; then
    echo "Conda is now available in PATH: $(which conda)"
    eval "$(conda shell.bash hook)"
    
    echo "Conda environments:"
    conda env list || echo "Failed to list conda environments"
    
    echo "Creating conda environment from: ${SLURM_SUBMIT_DIR}/slurm/environment.yml"
    conda env create -f ${SLURM_SUBMIT_DIR}/slurm/environment.yml || echo "Failed to create environment"
    
    echo "Conda environments after creation attempt:"
    conda env list || echo "Failed to list conda environments"
else
    echo "Conda still not available after loading attempts"
fi

echo "Environment setup completed at: $(date)"
echo "To use this environment, run: conda activate whale_legacy"
echo "PATH at end of script: $PATH"

# Disable verbose mode
set +x 