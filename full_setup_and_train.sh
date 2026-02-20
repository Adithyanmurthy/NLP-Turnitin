#!/bin/bash
# ============================================================
# BULLETPROOF FULL SETUP — Clone, Install, Download, Train
# Run with: bash full_setup_and_train.sh
# Survives tab close when run with nohup
# ============================================================
set -e  # Exit on error

export HF_HOME=/home/jovyan/hf_cache
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false

REPO_DIR="/home/jovyan/NLP-Turnitin"
LOG_PREFIX="[SETUP]"

log() {
    echo ""
    echo "============================================================"
    echo "$LOG_PREFIX $(date '+%H:%M:%S') — $1"
    echo "============================================================"
}

# ────────────────────────────────────────────────────────────
# STEP 1: CLONE REPO
# ────────────────────────────────────────────────────────────
log "STEP 1/7: Cloning repository"
cd /home/jovyan

if [ -d "$REPO_DIR" ]; then
    echo "Repo already exists, pulling latest..."
    cd "$REPO_DIR"
    git pull origin main || true
else
    git clone https://github.com/Adithyanmurthy/NLP-Turnitin.git
    cd "$REPO_DIR"
fi

echo "Repo ready at $REPO_DIR"

# ────────────────────────────────────────────────────────────
# STEP 2: INSTALL TORCH (correct CUDA version)
# ────────────────────────────────────────────────────────────
log "STEP 2/7: Installing PyTorch + CUDA 11.7"

# Remove old incompatible packages first
pip uninstall -y apex 2>/dev/null || true

# Install torch matching CUDA 11.x driver on this server
pip install --timeout 300 --force-reinstall \
    torch==2.0.1+cu117 \
    torchvision==0.15.2+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117

# Verify CUDA works
python -c "
import torch
print(f'torch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"

# ────────────────────────────────────────────────────────────
# STEP 3: INSTALL ALL PACKAGES (with known-good versions)
# ────────────────────────────────────────────────────────────
log "STEP 3/7: Installing all Python packages"

# Core ML packages
pip install --timeout 300 \
    transformers>=4.36.0 \
    datasets>=2.16.0 \
    "sentence-transformers>=2.2.0" \
    accelerate>=0.25.0 \
    peft>=0.6.0 \
    evaluate>=0.4.0 \
    huggingface-hub>=0.20.0 \
    safetensors>=0.4.0 \
    sentencepiece>=0.1.99

# Data processing
pip install --timeout 300 \
    "numpy>=1.24.0,<2.0" \
    pandas>=2.0.0 \
    "scikit-learn>=1.3.0" \
    nltk>=3.8.0 \
    tqdm>=4.65.0 \
    joblib>=1.3.0 \
    jsonlines>=4.0.0

# Specific version fixes for this server
pip install --timeout 300 protobuf==3.20.3
pip install --timeout 300 wandb==0.16.6
pip install --timeout 300 --only-binary :all: spacy==3.7.5

# GPU quantization
pip install --timeout 300 bitsandbytes>=0.41.0

# API and utilities
pip install --timeout 300 \
    fastapi>=0.100.0 \
    "uvicorn[standard]>=0.22.0" \
    pyyaml>=6.0 \
    tensorboard>=2.13.0 \
    PyMuPDF>=1.23.0 \
    python-docx>=0.8.11 \
    datasketch>=1.6.0

# NLTK data
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True); nltk.download('stopwords', quiet=True)" || true

# Force-fix numpy/sklearn compatibility (known issue on this server)
pip install --force-reinstall --no-deps "numpy>=1.24.0,<2.0" scikit-learn 2>/dev/null || true

echo ""
echo "Verifying all packages..."
python -c "
packages = {
    'torch': 'torch', 'transformers': 'transformers', 'datasets': 'datasets',
    'sentence_transformers': 'sentence-transformers', 'sklearn': 'scikit-learn',
    'numpy': 'numpy', 'pandas': 'pandas', 'nltk': 'nltk', 'tqdm': 'tqdm',
    'accelerate': 'accelerate', 'evaluate': 'evaluate', 'joblib': 'joblib',
    'peft': 'peft', 'bitsandbytes': 'bitsandbytes', 'sentencepiece': 'sentencepiece',
    'fastapi': 'fastapi', 'wandb': 'wandb', 'huggingface_hub': 'huggingface-hub',
    'safetensors': 'safetensors', 'jsonlines': 'jsonlines', 'datasketch': 'datasketch',
    'spacy': 'spacy', 'yaml': 'pyyaml', 'tensorboard': 'tensorboard',
    'fitz': 'PyMuPDF', 'docx': 'python-docx', 'uvicorn': 'uvicorn',
}
ok = 0
fail = 0
for imp, pip_name in packages.items():
    try:
        __import__(imp)
        ok += 1
    except ImportError:
        print(f'  MISSING: {pip_name}')
        fail += 1
print(f'  {ok}/{ok+fail} packages OK')
if fail > 0:
    print(f'  WARNING: {fail} packages missing')
"

# ────────────────────────────────────────────────────────────
# STEP 4: DOWNLOAD ALL DATASETS
# ────────────────────────────────────────────────────────────
log "STEP 4/7: Downloading datasets"
cd "$REPO_DIR"
python setup_all.py --step 2

# ────────────────────────────────────────────────────────────
# STEP 5: PREPROCESS ALL DATASETS
# ────────────────────────────────────────────────────────────
log "STEP 5/7: Preprocessing datasets"
cd "$REPO_DIR"
python setup_all.py --step 3

# ────────────────────────────────────────────────────────────
# STEP 6: DOWNLOAD PERSON 3 DATA + ALL MODELS
# ────────────────────────────────────────────────────────────
log "STEP 6/7: Downloading Person 3 data"
cd "$REPO_DIR"
python setup_all.py --step 4

log "STEP 6b/7: Downloading all pre-trained models (~80GB)"
cd "$REPO_DIR"
python setup_all.py --step 5

# ────────────────────────────────────────────────────────────
# STEP 7: FINAL STATUS CHECK
# ────────────────────────────────────────────────────────────
log "STEP 7/7: Final status check"
cd "$REPO_DIR"
python setup_all.py --check

echo ""
echo "============================================================"
echo "  SETUP COMPLETE — $(date)"
echo "  Ready for training: python run_all.py"
echo "============================================================"
