#!/bin/bash
#SBATCH --job-name=whale_evaluate
#SBATCH --output=whale_evaluate_%j.out
#SBATCH --error=whale_evaluate_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -p klab-gpu

echo "Starting CNN evaluation job at: $(date)"
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

# Change to the legacy CNN directory
cd $LEGACY_DIR

echo "Repository root: $REPO_ROOT"
echo "Working directory: $(pwd)"

# Create paths for prediction file, ground truth file, and class list
RESULTS_DIR="$OUTPUT_DIR/latest"  # Use most recent results folder
PREDICTION_FILE=$(find $RESULTS_DIR -name "total_predictions.csv" -type f | sort -r | head -n 1)
GROUND_TRUTH_FILE="$PROCESSED_DATA_DIR/ground_truth.csv"  # This path may need adjustment
CLASS_LIST_FILE="$LEGACY_DIR/temp_config.json"  # Using the temp config for class list

# Check if necessary files exist
if [ ! -f "$PREDICTION_FILE" ]; then
    echo "Error: Prediction file not found at $PREDICTION_FILE"
    exit 1
fi

# Create a temporary class list file if ground truth CSV doesn't exist
if [ ! -f "$GROUND_TRUTH_FILE" ]; then
    echo "Warning: Ground truth file not found, will use prediction file as reference"
    GROUND_TRUTH_FILE=$PREDICTION_FILE
fi

# Create class list file
cat > $CLASS_LIST_FILE << EOL
{
    "CLASSES": ["A", "N", "Z"]
}
EOL

# Run the evaluation script
echo "Starting specialized evaluation..."
python evaluation.py --predictions "$PREDICTION_FILE" --ground-truth "$GROUND_TRUTH_FILE" --class-list "$CLASS_LIST_FILE" --output "$RESULTS_DIR/specialized_metrics"

# Clean up
rm -f $CLASS_LIST_FILE

# Deactivate the conda environment
conda deactivate

end_time=$(date +%s)
echo "Evaluation completed at: $(date)"
elapsed_time=$((end_time - start_time))
echo "Total elapsed time: $(date -u -d @$elapsed_time +%H:%M:%S)" 