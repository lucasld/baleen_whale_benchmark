#!/bin/bash
#SBATCH --job-name=whale_preprocess
#SBATCH --output=whale_preprocess_%j.out
#SBATCH --error=whale_preprocess_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH -p klab-cpu

echo "Starting preprocessing job at: $(date)"
start_time=$(date +%s)

# Load necessary modules
spack load miniconda3@4.10.3
eval "$(conda shell.bash hook)"

# Activate the conda environment
conda activate whale_cnn_legacy

# Set environment variables
export RAW_DATA_DIR="/share/klab/danthes/data/ArcticWhales_AADC"
export PROCESSED_DATA_DIR="/share/klab/danthes/data/ArcticWhales_AADC_processed"

# Create the processed data directory if it doesn't exist
mkdir -p $PROCESSED_DATA_DIR

# Get the repository root directory
REPO_ROOT=$(dirname $(dirname $(dirname $(dirname $(realpath $0)))))
PREPROCESS_DIR="$REPO_ROOT/legacy/cnn/custom_preprocessing"

# Change to the preprocessing directory
cd $PREPROCESS_DIR

echo "Repository root: $REPO_ROOT"
echo "Working directory: $(pwd)"

# Create a temporary configuration file with correct paths
cat > temp_config.json << EOL
{
    "raw_data_path": "$RAW_DATA_DIR",
    "output_path": "$PROCESSED_DATA_DIR"
}
EOL

# Run the preprocessing workflow
echo "Running preprocessing workflow..."
python preprocess_workflow.py --config temp_config.json

# Clean up
rm temp_config.json

# Deactivate the conda environment
conda deactivate

end_time=$(date +%s)
echo "Preprocessing completed at: $(date)"
elapsed_time=$((end_time - start_time))
echo "Total elapsed time: $(date -u -d @$elapsed_time +%H:%M:%S)" 