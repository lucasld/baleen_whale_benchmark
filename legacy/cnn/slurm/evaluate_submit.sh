#!/bin/bash
#SBATCH --job-name=whale_cnn_eval
#SBATCH --output=logs/whale_cnn_eval_%j.out
#SBATCH --error=logs/whale_cnn_eval_%j.err
#SBATCH --time=4:00:00
#SBATCH --partition=klab-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

# Record start time
start_time=$(date +%s)
echo "Job started at: $(date)"

# Load required modules
spack load miniconda3@4.10.3
spack load cuda@11.8.0
spack load cudnn@8.6.0.163-11.8
eval "$(conda shell.bash hook)"

# Activate environment (assuming it was created by the training job)
conda activate whale_legacy

# Configure GPU environment variables
mkdir -p $CONDA_PREFIX/lib/nvvm/libdevice/
cp -p $CONDA_PREFIX/lib/libdevice.10.bc $CONDA_PREFIX/lib/nvvm/libdevice/ 2>/dev/null || true
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX

# Set temporary directory to a location with sufficient space
export TMPDIR="/share/klab/danthes/tmp"
mkdir -p $TMPDIR

# Change to the legacy CNN directory
cd ${SLURM_SUBMIT_DIR}

# Ensure logs directory exists
mkdir -p ${SLURM_SUBMIT_DIR}/logs

# Print GPU information
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "GPU devices available:"
nvidia-smi

# Check if required parameters are provided
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ] || [ -z "$4" ]; then
    echo "Usage: sbatch evaluate_submit.sh <predictions_csv> <ground_truth_csv> <class_list_json> <output_folder>"
    exit 1
fi

PREDICTIONS_CSV="$1"
GROUND_TRUTH_CSV="$2"
CLASS_LIST_JSON="$3"
OUTPUT_FOLDER="$4"

# Print configuration
echo "Using the following parameters:"
echo "Predictions CSV: ${PREDICTIONS_CSV}"
echo "Ground Truth CSV: ${GROUND_TRUTH_CSV}"
echo "Class List JSON: ${CLASS_LIST_JSON}"
echo "Output Folder: ${OUTPUT_FOLDER}"

# Run the evaluation script
echo "Starting CNN evaluation..."
python evaluation.py ${PREDICTIONS_CSV} ${GROUND_TRUTH_CSV} ${CLASS_LIST_JSON} ${OUTPUT_FOLDER}

# Check if the job completed successfully
if [ $? -eq 0 ]; then
    echo "Evaluation completed successfully."
else
    echo "Evaluation failed with exit code $?."
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