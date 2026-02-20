#!/usr/bin/env python3
"""
Person 2 — Master Training Script
Runs the complete plagiarism detection pipeline:
  1. Build MinHash/LSH reference index
  2. Train Sentence-BERT
  3. Train Cross-Encoder
  4. Verify with example detection

Run from person_2/ directory:
    python run_all.py
"""

import sys
import os
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).parent
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
INDEX_DIR = BASE_DIR / "reference_index"
DATA_DIR = BASE_DIR / "data"

CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)


def run_command(cmd, description):
    """Run a command and handle errors."""
    print("\n" + "=" * 70)
    print(f"  {description}")
    print("=" * 70)
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"\n  ✓ {description} — completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n  ✗ {description} — failed (exit code {e.returncode})")
        return False
    except Exception as e:
        print(f"\n  ✗ {description} — error: {e}")
        return False


def index_exists():
    return (INDEX_DIR / "metadata.json").exists()


def checkpoint_exists(name):
    return (CHECKPOINTS_DIR / name).exists()


def main():
    print("=" * 70)
    print("  PERSON 2 — PLAGIARISM DETECTION ENGINE")
    print("  Complete Training Pipeline")
    print("=" * 70)
    print()
    print("  Steps:")
    print("    1. Build MinHash/LSH reference index from corpus")
    print("    2. Train Sentence-BERT (all-mpnet-base-v2)")
    print("    3. Train DeBERTa-v3 Cross-Encoder")
    print("    4. Run example plagiarism detection")
    print()

    py = sys.executable

    # Step 1: Build reference index
    if not index_exists():
        corpus_path = DATA_DIR
        if not corpus_path.exists() or not any(corpus_path.iterdir()):
            print("  [STEP 1] No corpus data found in person_2/data/.")
            print("           Place reference documents (*.txt) in person_2/data/ first,")
            print("           or use Person 1's preprocessed PAN plagiarism data.")
            print("           Skipping index build for now.\n")
        else:
            run_command(
                [py, str(BASE_DIR / "scripts" / "build_index.py"),
                 "--corpus_path", str(corpus_path),
                 "--output_path", str(INDEX_DIR)],
                "Step 1: Build MinHash/LSH Reference Index",
            )
    else:
        print("  [STEP 1] Reference index already exists — skipping\n")

    # Step 2: Train Sentence-BERT
    if not checkpoint_exists("sbert"):
        run_command(
            [py, str(BASE_DIR / "models" / "train_sentence_bert.py"),
             "--output_path", str(CHECKPOINTS_DIR / "sbert"),
             "--epochs", "3"],
            "Step 2: Train Sentence-BERT",
        )
    else:
        print("  [STEP 2] Sentence-BERT already trained — skipping\n")

    # Step 3: Train Cross-Encoder
    if not checkpoint_exists("cross_encoder"):
        run_command(
            [py, str(BASE_DIR / "models" / "train_cross_encoder.py"),
             "--output_path", str(CHECKPOINTS_DIR / "cross_encoder"),
             "--epochs", "3"],
            "Step 3: Train DeBERTa-v3 Cross-Encoder",
        )
    else:
        print("  [STEP 3] Cross-Encoder already trained — skipping\n")

    # Step 4: Quick test
    print("\n" + "=" * 70)
    print("  Step 4: Quick Verification")
    print("=" * 70)
    if index_exists():
        run_command(
            [py, str(BASE_DIR / "example.py")],
            "Step 4: Example Plagiarism Detection",
        )
    else:
        print("  Skipping verification — no index built yet.\n")

    print("\n" + "=" * 70)
    print("  PERSON 2 PIPELINE COMPLETE")
    print("=" * 70)
    print()
    print("  To use in Person 4's integration:")
    print("    from person_2.src.plagiarism_detector import PlagiarismDetector")
    print("    detector = PlagiarismDetector(index_path='person_2/reference_index')")
    print("    report = detector.check('your text here')")
    print()


if __name__ == "__main__":
    main()
