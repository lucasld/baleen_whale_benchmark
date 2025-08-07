#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32GB
#SBATCH -c 2
#SBATCH -p klab-l40s
#SBATCH --gres=gpu:1
#SBATCH --job-name=eval_cnn
#SBATCH --error=slurm/outputs/logs/eval_cnn_%j.err
#SBATCH --output=slurm/outputs/logs/eval_cnn_%j.out

# This script runs the test and evaluation steps on a trained model.
#
# Usage: sbatch run_evaluation.sh <model_directory_name>
#
# Example: sbatch run_evaluation.sh 250626_134153
#
# The <model_directory_name> is the timestamped folder created during training,
# located in 'outputs/cnn_results/'.

# --- Configuration ---

# Load required modules
spack load miniconda3

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate whale_env2

# Set base paths
export BASE_DIR="/share/klab/danthes/lliessduques/baleen_whale_benchmark/legacy/cnn"
export CONFIG_PATH="${BASE_DIR}/config.json"
export EVAL_OUTPUT_DIR="${BASE_DIR}/outputs/cnn_results/evaluation"

# --- Script Logic ---

# Go to the base directory
cd "$BASE_DIR"

# Check if a model directory name was provided as an argument
if [ -z "$1" ]; then
    echo "Error: No model directory specified."
    echo "Usage: sbatch $0 <model_directory_name>"
    echo "Example: sbatch $0 250626_134153"
    exit 1
fi

# Construct the full path to the model folder
MODEL_FOLDER="${BASE_DIR}/outputs/cnn_results/$1"

if [ ! -d "$MODEL_FOLDER" ]; then
    echo "Error: Model directory not found at: ${MODEL_FOLDER}"
    exit 1
fi

echo "Using model folder: $MODEL_FOLDER"

# Create evaluation output directory if it doesn't exist
mkdir -p "$EVAL_OUTPUT_DIR"

# Record start time and memory usage
echo "Job started at: $(date)"
free -h

# --- Step 1: Run test.py ---
# This script generates confusion matrices and other test results from the trained model.
echo "Running test.py..."
python test.py --config "${CONFIG_PATH}" --model_folder "${MODEL_FOLDER}"

# --- Step 2: Loop through all folds and run evaluation ---
echo "Searching for evaluation input files in ${MODEL_FOLDER}..."

# Find all prediction files
PREDICTION_FILES=$(find "${MODEL_FOLDER}" -name "*predictions*.csv")

if [ -z "$PREDICTION_FILES" ]; then
    echo "Error: No prediction files (*predictions*.csv) found in ${MODEL_FOLDER}. Test step may have failed."
    exit 1
fi

# The label list is the same for all folds
LABEL_LIST_FILE="${MODEL_FOLDER}/labels.json"
if [ ! -f "$LABEL_LIST_FILE" ]; then
    echo "Error: labels.json not found in ${MODEL_FOLDER}."
    exit 1
fi
echo "Using label list file: $LABEL_LIST_FILE"

# Loop through each prediction file
for PREDICTIONS_FILE in $PREDICTION_FILES; do
    echo "--- Evaluating Fold: $(basename "$PREDICTIONS_FILE") ---"

    # TODO: Derive the ground truth filename from the prediction filename
    # The filenames are consistent, e.g., predictions_... corresponds to data_used_...
    FILENAME=$(basename "$PREDICTIONS_FILE")
    GROUND_TRUTH_FILENAME="${FILENAME/predictions/data_used}"
    GROUND_TRUTH_FILE="${MODEL_FOLDER}/${GROUND_TRUTH_FILENAME}"
    
    # The ground truth file for a specific prediction might be in a sub-directory
    # This happens when noise_ratio_test is not 'all'
    if [ ! -f "$GROUND_TRUTH_FILE" ]; then
        # Extract fold name, e.g., "fold_BallenyIslands2015_noise_0.25"
        DIRNAME=$(echo "$FILENAME" | sed -e 's/predictions_//' -e 's/_noiseall_noiseall.csv//' -e 's/_noise_/_noiseall_/' -e 's/_noiseall_/_noise_/')
        # Fallback to searching the sub-directory
        GROUND_TRUTH_FILE=$(find "${MODEL_FOLDER}" -name "*${DIRNAME}*.csv" | grep "data_used")
    fi

    if [ -z "$GROUND_TRUTH_FILE" ] || [ ! -f "$GROUND_TRUTH_FILE" ]; then
        echo "Warning: Could not find matching ground truth file for $PREDICTIONS_FILE. Skipping."
        continue
    fi

    echo "Found predictions file: $PREDICTIONS_FILE"
    echo "Found ground truth file: $GROUND_TRUTH_FILE"

    # --- Step 3: Run evaluation.py ---
    echo "Running evaluation.py for this fold..."
    python evaluation.py \
        --predictions "${PREDICTIONS_FILE}" \
        --ground_truth "${GROUND_TRUTH_FILE}" \
        --label_list "${LABEL_LIST_FILE}" \
        --output_path "${EVAL_OUTPUT_DIR}"
done

echo "--- Full Evaluation Complete ---"
echo "Results for all folds are in ${EVAL_OUTPUT_DIR}"

# --- Job Completion ---
echo "Job ended at: $(date)"
free -h

JOB_ID=$SLURM_JOB_ID
echo "To see detailed job statistics, run:"
echo "sacct -j $JOB_ID --format=JobID,JobName,MaxRSS,AveRSS" 