#!/bin/bash
#SBATCH --job-name=create_env
#SBATCH --output=create_env_output_%j.txt
#SBATCH --error=create_env_error_%j.txt
#SBATCH --time=01:00:00
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH --partition=klab-cpu

echo "Starting conda environment setup at: $(date)"

# Load miniconda
spack load miniconda3@4.10.3
eval "$(conda shell.bash hook)"

# Create environment if it doesn't exist
if conda env list | grep -q "baleen_whale_env"; then
    echo "Environment baleen_whale_env already exists"
else
    echo "Creating new environment: baleen_whale_env"
    conda create -y -n baleen_whale_env python=3.9
fi

# Activate and install packages
conda activate baleen_whale_env

echo "Installing core dependencies..."
conda install -y pandas scipy numpy pillow matplotlib

echo "Installing TensorFlow 2.13.0..."
pip install tensorflow==2.13.0

echo "Installing scikit-learn..."
pip install scikit-learn

echo "Installing Koogu..."
pip install koogu

echo "Environment setup completed at: $(date)"
echo "To use this environment, run:"
echo "  spack load miniconda3@4.10.3"
echo "  eval \"\$(conda shell.bash hook)\""
echo "  conda activate baleen_whale_env" 