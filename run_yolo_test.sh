#!/bin/bash
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32GB
#SBATCH -c 4
#SBATCH -p klab-gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=yolo_test
#SBATCH --error=yolo_test_%j.err
#SBATCH --output=yolo_test_%j.out

# Load required modules
spack load miniconda3

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate yoloenv

# Set base paths
export BASE_DIR="$(pwd)"

# Go to the base directory
cd "$BASE_DIR"

# Run the YOLO assumptions test script
echo "Starting YOLO assumptions verification..."
python test_yolo_assumptions.py

echo "YOLO test completed."
