"""
Person 1 — Pre-trained Model Downloader
Downloads all 4 base models from HuggingFace before training starts.
This avoids mid-training download failures on slow/unstable connections.

Models downloaded:
  1. microsoft/deberta-v3-large   (~1.7 GB)
  2. roberta-large                (~1.4 GB)
  3. allenai/longformer-base-4096 (~0.6 GB)
  4. xlm-roberta-large            (~2.2 GB)

Total: ~6 GB (cached in HuggingFace cache, typically ~/.cache/huggingface/)
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from config import MODELS


def main():
    print("=" * 60)
    print("  PRE-TRAINED MODEL DOWNLOADER")
    print("=" * 60)

    model_names = []
    for key, cfg in MODELS.items():
        name = cfg["name"]
        if name not in model_names:
            model_names.append((key, name))

    print(f"\n  Models to download: {len(model_names)}")
    for key, name in model_names:
        print(f"    - {name} ({key})")

    for key, name in model_names:
        print(f"\n{'─' * 40}")
        print(f"  Downloading: {name}")
        try:
            print(f"    Tokenizer...")
            AutoTokenizer.from_pretrained(name)
            print(f"    Model weights...")
            AutoModelForSequenceClassification.from_pretrained(name, num_labels=2)
            print(f"    [OK] {name} cached successfully")
        except Exception as e:
            print(f"    [ERROR] Failed to download {name}: {e}")
            print(f"    → Check your internet connection and try again")

    print(f"\n{'=' * 60}")
    print("  All pre-trained models downloaded and cached.")
    print("  They will load instantly during training.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
