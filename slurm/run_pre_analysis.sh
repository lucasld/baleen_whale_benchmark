#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32GB
#SBATCH -c 4
#SBATCH -p klab-l40s
#SBATCH --job-name=pre_analysis
#SBATCH --error=slurm/outputs/logs/pre_analysis_%j.err
#SBATCH --output=slurm/outputs/logs/pre_analysis_%j.out

set -euo pipefail

# Load modules and activate env (match training style)
spack load miniconda3
eval "$(conda shell.bash hook)"

# If your training env exists, reuse it; otherwise fallback
TARGET_ENV="whale_env2"
if conda env list | awk '{print $1}' | grep -qx "$TARGET_ENV"; then
  conda activate "$TARGET_ENV"
else
  echo "Conda env '$TARGET_ENV' not found; creating a minimal env for pre-analysis..."
  conda create -y -n pre_analysis_env python=3.10
  conda activate pre_analysis_env
fi

BASE_DIR="/share/klab/danthes/lliessduques/test/baleen_whale_benchmark"
DATA_DIR="$BASE_DIR/datasets/preprocessed_dataset/spectrograms"
CONFIG_PATH="$BASE_DIR/src/config.json"
OUT_DIR="$BASE_DIR/outputs/analysis"

mkdir -p "$OUT_DIR" "${BASE_DIR}/slurm/outputs/logs"

echo "Job started at: $(date)"
echo "Host: $(hostname)"
echo "CWD: $(pwd)"
echo "Python: $(python -V 2>&1)"
echo "Conda env: ${CONDA_DEFAULT_ENV:-none} (${CONDA_PREFIX:-na})"
echo "GPU(s):"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "Disk usage at start:"; df -h "$BASE_DIR" || true
echo "Memory at start:"; free -h || true

# Install minimal deps if missing
python - <<'PY'
import importlib, sys
req = [
    ("cv2", "opencv-python-headless"),
    ("sklearn", "scikit-learn"),
    ("seaborn", "seaborn"),
    ("matplotlib", "matplotlib"),
    ("pandas", "pandas"),
    ("numpy", "numpy"),
]
missing = []
for mod, pkg in req:
    try:
        importlib.import_module(mod)
    except Exception:
        missing.append(pkg)
if missing:
    print("Installing missing packages:", missing)
    import subprocess
    cmd = [sys.executable, "-m", "pip", "install", "--no-input"] + missing
    subprocess.check_call(cmd)
else:
    print("All required packages are present.")
PY

echo "Running pre-analysis..."
python "$BASE_DIR/src/utils/pre_analysis.py" \
  --config "$CONFIG_PATH" \
  --data_dir "$DATA_DIR" \
  --max_per_class 2000 \
  --out_dir "$OUT_DIR" \
  --disable_tsne

STATUS=$?

echo "Job ended at: $(date) with status $STATUS"
echo "Disk usage at end:"; df -h "$BASE_DIR" || true
echo "Memory at end:"; free -h || true
echo "To view outputs, check: $OUT_DIR and the timestamped subdirectory."

exit $STATUS


