#!/bin/bash

# Person 3 - Setup Script
# Installs all dependencies and prepares the environment

echo "=========================================="
echo "PERSON 3 - HUMANIZATION MODULE SETUP"
echo "=========================================="

# Check Python version
echo ""
echo "[1/5] Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "  Python version: $python_version"

# Check if pip is installed
echo ""
echo "[2/5] Checking pip..."
if ! command -v pip3 &> /dev/null; then
    echo "  ✗ pip3 not found. Please install pip3 first."
    exit 1
fi
echo "  ✓ pip3 found"

# Install requirements
echo ""
echo "[3/5] Installing Python dependencies..."
echo "  This may take several minutes..."
pip3 install -r requirements.txt

# Check CUDA availability
echo ""
echo "[4/5] Checking CUDA availability..."
python3 -c "import torch; print('  CUDA available:', torch.cuda.is_available()); print('  CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A'); print('  GPU count:', torch.cuda.device_count() if torch.cuda.is_available() else 0)"

# Create necessary directories
echo ""
echo "[5/5] Creating directories..."
mkdir -p data
mkdir -p checkpoints
mkdir -p logs
mkdir -p models
echo "  ✓ Directories created"

echo ""
echo "=========================================="
echo "SETUP COMPLETE"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Run: python3 run_all.py"
echo "     (This will download datasets and train all models)"
echo ""
echo "  OR run step by step:"
echo "  1. python3 dataset_downloader.py"
echo "  2. python3 train_flan_t5.py"
echo "  3. python3 train_pegasus.py"
echo "  4. python3 train_mistral.py"
echo "  5. python3 setup_dipper.py"
echo ""
echo "For more information, see README.md"
