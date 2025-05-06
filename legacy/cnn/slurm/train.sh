#!/bin/bash
#SBATCH --job-name=whale_train
#SBATCH --output=whale_train_%j.out
#SBATCH --error=whale_train_%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -p klab-gpu

echo "Starting CNN training job at: $(date)"
start_time=$(date +%s)

# Load necessary modules
spack load cuda@11.8.0
spack load cudnn@8.6.0.163-11.8
spack load miniconda3@4.10.3
eval "$(conda shell.bash hook)"

# Activate the conda environment
conda activate whale_cnn_legacy

# Set up TensorFlow environment variables for GPU
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH
mkdir -p $CONDA_PREFIX/lib/nvvm/libdevice/
cp -p $CONDA_PREFIX/lib/libdevice.10.bc $CONDA_PREFIX/lib/nvvm/libdevice/ 2>/dev/null || echo "Warning: libdevice.10.bc not found"
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX
export TMPDIR="/share/klab/danthes/tmp"
mkdir -p $TMPDIR

# Get the repository root directory
REPO_ROOT=$(dirname $(dirname $(dirname $(dirname $(realpath $0)))))
LEGACY_DIR="$REPO_ROOT/legacy/cnn"
PROCESSED_DATA_DIR="/share/klab/danthes/data/ArcticWhales_AADC_processed"
OUTPUT_DIR="$PROCESSED_DATA_DIR/cnn_results"

# Create the output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Change to the legacy CNN directory
cd $LEGACY_DIR

echo "Repository root: $REPO_ROOT"
echo "Working directory: $(pwd)"

# Create a temporary configuration file with correct paths
cat > temp_config.json << EOL
{
    "DATA_DIR": "$PROCESSED_DATA_DIR/spectrograms",
    "OUTPUT_DIR": "$OUTPUT_DIR",
    "LOCATIONS": ["BallenyIslands2015", "DrygalskiIceNorthBeaufort2016", "FoynIslandCapeShirreff2015", "RossSeaFoynIsland2013"],
    "SAMPLES_PER_CLASS": 250,
    "EPOCHS": 100,
    "BATCH_SIZE": 32,
    "EARLY_STOPPING": 20,
    "LEARNING_RATE": 0.001,
    "MODEL_OPTIMIZER": "Adam",
    "NOISE_RATIO": 0.25,
    "VALIDATION_SPLIT": 0.3,
    "CLASSES": ["A", "N", "Z"]
}
EOL

# Check if GPU is available for TensorFlow
echo "Checking GPU availability..."
python -c "import tensorflow as tf; print('Number of GPUs available:', len(tf.config.list_physical_devices('GPU')))"

# Run the training script
echo "Starting CNN training..."
python train.py --config temp_config.json

# Clean up
rm temp_config.json

# Deactivate the conda environment
conda deactivate

end_time=$(date +%s)
echo "Training completed at: $(date)"
elapsed_time=$((end_time - start_time))
echo "Total elapsed time: $(date -u -d @$elapsed_time +%H:%M:%S)" 