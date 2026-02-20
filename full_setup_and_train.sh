#!/bin/bash
# ============================================================
# BULLETPROOF FULL SETUP — Python 3.8 + CUDA 11.x compatible
# All package versions pinned to known-working versions.
# Run: nohup bash full_setup_and_train.sh > /home/jovyan/setup.log 2>&1 &
# ============================================================

REPO_DIR="/home/jovyan/NLP-Turnitin"
export HF_HOME="$REPO_DIR/hf_cache"
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export PIP_DEFAULT_TIMEOUT=300

log() {
    echo ""
    echo "============================================================"
    echo "[SETUP] $(date '+%H:%M:%S') — $1"
    echo "============================================================"
}

# ────────────────────────────────────────────────────────────
# STEP 1: CLONE REPO
# ────────────────────────────────────────────────────────────
log "STEP 1/7: Cloning repository"
cd /home/jovyan

if [ -d "$REPO_DIR" ]; then
    echo "Repo exists, pulling latest..."
    cd "$REPO_DIR"
    git pull origin main || true
else
    git clone https://github.com/Adithyanmurthy/NLP-Turnitin.git
    cd "$REPO_DIR"
fi
echo "Repo ready at $REPO_DIR"

# ────────────────────────────────────────────────────────────
# STEP 2: INSTALL TORCH (CUDA 11.7 compatible)
# ────────────────────────────────────────────────────────────
log "STEP 2/7: Installing PyTorch + CUDA 11.7"

# Remove old broken packages
pip uninstall -y apex 2>/dev/null || true

# CRITICAL: Install torch with --no-deps to prevent it from pulling
# typing-extensions 4.15.0 from the PyTorch wheel index (requires Python 3.9+)
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 \
    --no-deps \
    --extra-index-url https://download.pytorch.org/whl/cu117

# Now install torch's actual dependencies with pinned versions
pip install "typing-extensions==4.12.2" "filelock>=3.0" "sympy" "networkx" "jinja2" "pillow"

# Verify
python -c "
import torch
print(f'torch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
" || { echo "FATAL: torch CUDA check failed"; exit 1; }

# ────────────────────────────────────────────────────────────
# STEP 3: INSTALL ALL PACKAGES (pinned for Python 3.8)
# ────────────────────────────────────────────────────────────
log "STEP 3/7: Installing all packages (Python 3.8 pinned)"

# --- Pinned core versions (all verified on Python 3.8) ---
# typing-extensions already pinned above
# numpy: 1.24.x is last for Python 3.8
# pandas: 2.0.x supports 3.8
# scikit-learn: 1.3.x is last for 3.8
# transformers: 4.38.x is safe for 3.8 (4.41+ may need 3.9)
# accelerate: 0.27.x safe for 3.8
# peft: 0.8.x safe for 3.8
# datasets: 2.18.x safe for 3.8
# sentence-transformers: 2.7.x safe for 3.8
# huggingface-hub: 0.21.x safe for 3.8

pip install \
    "numpy==1.24.4" \
    "pandas==2.0.3" \
    "scikit-learn==1.3.2" \
    "transformers==4.38.2" \
    "datasets==2.18.0" \
    "accelerate==0.27.2" \
    "peft==0.8.2" \
    "sentence-transformers==2.6.1" \
    "huggingface-hub==0.21.4" \
    "safetensors==0.4.5" \
    "tokenizers==0.15.2"

pip install \
    "evaluate==0.4.1" \
    "sentencepiece==0.2.0" \
    "protobuf==3.20.3" \
    "nltk==3.8.1" \
    "tqdm==4.66.5" \
    "joblib==1.3.2" \
    "jsonlines==4.0.0"

pip install "wandb==0.16.6"
pip install --only-binary :all: "spacy==3.7.5"

pip install \
    "bitsandbytes==0.42.0" \
    "fastapi==0.109.2" \
    "uvicorn==0.27.1" \
    "pyyaml==6.0.1" \
    "tensorboard==2.14.1" \
    "PyMuPDF==1.24.0" \
    "python-docx==1.1.0" \
    "datasketch==1.6.5"

# Re-pin typing-extensions after all installs (some packages may have upgraded it)
pip install "typing-extensions==4.12.2"

# NLTK data
python -c "
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
" 2>/dev/null || true

# ────────────────────────────────────────────────────────────
# VERIFY ALL 27 PACKAGES
# ────────────────────────────────────────────────────────────
log "Verifying all 27 packages..."
# First verify typing-extensions is correct (most critical check)
python -c "
import typing_extensions
v = typing_extensions.__version__
print(f'  typing-extensions: {v}')
assert v.startswith('4.12'), f'FATAL: typing-extensions {v} will break on Python 3.8! Need 4.12.x'
print('  ✓ typing-extensions version OK for Python 3.8')
"
if [ $? -ne 0 ]; then
    echo "FIXING: Re-pinning typing-extensions to 4.12.2..."
    pip install --force-reinstall "typing-extensions==4.12.2"
fi

python -c "
import sys
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
        print(f'  OK  {pip_name}')
        ok += 1
    except ImportError:
        print(f'  FAIL {pip_name}')
        fail += 1
print(f'\n  Result: {ok}/{ok+fail} packages installed')
if fail > 0:
    print(f'  WARNING: {fail} packages failed — training may still work')
"

# ────────────────────────────────────────────────────────────
# STEP 4: DOWNLOAD ALL DATASETS
# ────────────────────────────────────────────────────────────
log "STEP 4/7: Downloading datasets"
cd "$REPO_DIR"
python setup_all.py --step 2 || echo "WARNING: Some datasets may have failed (manual ones are expected)"

# ────────────────────────────────────────────────────────────
# STEP 5: PREPROCESS ALL DATASETS
# ────────────────────────────────────────────────────────────
log "STEP 5/7: Preprocessing datasets"
cd "$REPO_DIR"
python setup_all.py --step 3 || echo "WARNING: Preprocessing had issues"

# ────────────────────────────────────────────────────────────
# STEP 6: PERSON 3 DATA + ALL MODELS
# ────────────────────────────────────────────────────────────
log "STEP 6a/7: Person 3 data"
cd "$REPO_DIR"
python setup_all.py --step 4 || echo "WARNING: Person 3 data had issues"

log "STEP 6b/7: Downloading pre-trained models (~80GB)"
cd "$REPO_DIR"
python setup_all.py --step 5 || echo "WARNING: Some models may have failed"

# ────────────────────────────────────────────────────────────
# STEP 7: FINAL STATUS
# ────────────────────────────────────────────────────────────
log "STEP 7/7: Final status check"
cd "$REPO_DIR"
python setup_all.py --check

echo ""
echo "============================================================"
echo "  SETUP COMPLETE — $(date)"
echo "  To start training: python run_all.py"
echo "============================================================"
