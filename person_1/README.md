# Person 1 — Data Pipeline & AI Detection Module

## Structure
```
person_1/
├── config.py                  # All configuration constants
├── requirements.txt           # Dependencies
├── data/                      # Downloaded datasets go here
│   └── raw/                   # Raw downloads
│   └── processed/             # Cleaned, formatted data
│   └── splits/                # Train/val/test splits
├── scripts/
│   ├── download_datasets.py   # Download all 18 datasets
│   └── preprocess.py          # Clean, format, split all datasets
├── data_loader.py             # Shared data loading utility (used by P2, P3, P4)
├── train_deberta.py           # Fine-tune DeBERTa-v3-large
├── train_roberta.py           # Fine-tune RoBERTa-large
├── train_longformer.py        # Fine-tune Longformer-base
├── train_xlm_roberta.py       # Fine-tune XLM-RoBERTa-large
├── train_meta_classifier.py   # Train ensemble meta-classifier
├── ai_detector.py             # Final detection module: detect(text) → float
├── evaluate.py                # Full evaluation on test sets
└── checkpoints/               # Saved model checkpoints
```

## Quick Start
```bash
pip install -r requirements.txt
python scripts/download_datasets.py
python scripts/preprocess.py
python train_deberta.py
python train_roberta.py
python train_longformer.py
python train_xlm_roberta.py
python train_meta_classifier.py
python evaluate.py
```

## Interface Contract
```python
from person_1.ai_detector import AIDetector

detector = AIDetector(checkpoint_dir="person_1/checkpoints")
score = detector.detect("Some text to analyze")  # Returns float 0.0 (human) to 1.0 (AI)
```
