#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=256G
#SBATCH -c 4
#SBATCH -p klab-l40s
#SBATCH --gres=gpu:1
#SBATCH --job-name=train_cnn_test
#SBATCH --error=slurm/outputs/logs/train_cnn_%j.err
#SBATCH --output=slurm/outputs/logs/train_cnn_%j.out

# Parse command line arguments
MODEL_ARCH="${1:-IdilCNN}"  # Default to IdilCNN if no argument provided

# Validate model architecture
case "$MODEL_ARCH" in
    "efficientnet"|"EfficientNet"|"EfficientNet-B0")
        MODEL_ARCH="EfficientNet-B0"
        ;;
    "mobilenet"|"MobileNet"|"MobileNet-V2"|"mobilenetv2")
        MODEL_ARCH="MobileNet-V2"
        ;;
    "mobilenetv3"|"MobileNet-V3"|"MobileNet-V3-Small"|"mobilenetv3small")
        MODEL_ARCH="MobileNet-V3-Small"
        ;;
    "simplecnn"|"SimpleCNN"|"simple")
        MODEL_ARCH="SimpleCNN"
        ;;
    "idilcnn"|"IdilCNN"|"")
        MODEL_ARCH="IdilCNN"
        ;;
    *)
        echo "Error: Unknown model architecture '$MODEL_ARCH'"
        echo "Available options: efficientnet, mobilenet, mobilenetv3, simplecnn, idilcnn"
        echo "Usage: sbatch $0 [efficientnet|mobilenet|mobilenetv3|simplecnn|idilcnn]"
        exit 1
        ;;
esac

echo "Using model architecture: $MODEL_ARCH"

# Load required modules
spack load miniconda3

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate whale_env2

# Set paths
export BASE_DIR="/share/klab/danthes/lliessduques/test/baleen_whale_benchmark"
export CONFIG_PATH="${BASE_DIR}/src/config.json"
export MODEL_ARCHITECTURE="$MODEL_ARCH"

# Go to the base directory
cd "$BASE_DIR"

# Create output directory if it doesn't exist
mkdir -p "${BASE_DIR}/outputs/cnn_results"

# Record start time and memory usage
echo "Job started at: $(date)"
echo "Initial memory usage:"
free -h

# Run the training script by calling the run_from_config function directly.
# This avoids the interactive input() in train.py's __main__ block.
# Make sure the "DATA_DIR" in your config.json points to the correct spectrograms folder.
# TODO: Added environment and provenance printouts to aid result traceability in .out
python -u -c "
import sys, json, pathlib, platform, tensorflow as tf, subprocess
sys.path.insert(0, 'src')
import train

# Load and modify config with model architecture
config_path = pathlib.Path('${CONFIG_PATH}')
with open(config_path, 'r') as f:
    config = json.load(f)

# Override model architecture from environment variable
config['model_name'] = '${MODEL_ARCHITECTURE}'

print('=' * 60)
print(f'Starting training with model: ${MODEL_ARCHITECTURE}')
print(f'Config file: ${CONFIG_PATH}')
print(f'Python: {platform.python_version()} | TF: {tf.__version__}')
print('GPU(s):', subprocess.getoutput('nvidia-smi --query-gpu=name,memory.total --format=csv,noheader'))
print('=' * 60)

# Run training with modified config
train.run_from_config(config_path, config_override=config)
"

echo "Training job finished."

# Record end time and memory usage
echo "Job ended at: $(date)"
echo "Final memory usage:"
free -h

# Get the job ID from the environment
JOB_ID=$SLURM_JOB_ID

# Print detailed job statistics - will be available after job completes
echo "To see detailed job statistics after this job completes, run:"
echo "sacct -j $JOB_ID --format=JobID,JobName,MaxRSS,MaxVMSize,AveRSS,AveVMSize"
echo "This will show the maximum and average memory usage of your job." 