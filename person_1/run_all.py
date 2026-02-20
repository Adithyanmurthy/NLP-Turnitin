#!/usr/bin/env python3
"""
Person 1 — Training Script
Checks if all setup is complete, then trains everything automatically.

BEFORE running this, run:  python setup_all.py

Usage (from person_1/ directory):
    python run_all.py
"""

import sys
import subprocess
from pathlib import Path

from config import CHECKPOINT_DIR, SPLITS_DIR, RAW_DIR, MODELS, DATASETS


# ─── Readiness Checks ───────────────────────────────────

def check_requirements():
    """Check if key packages are installed."""
    missing = []
    packages = {
        "torch": "torch",
        "transformers": "transformers",
        "datasets": "datasets",
        "sklearn": "scikit-learn",
        "numpy": "numpy",
        "tqdm": "tqdm",
        "accelerate": "accelerate",
        "joblib": "joblib",
    }
    for import_name, pip_name in packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pip_name)
    return missing


def check_datasets():
    """Check which raw datasets are downloaded."""
    downloaded = []
    missing = []
    for name, info in DATASETS.items():
        raw_path = RAW_DIR / name
        if raw_path.exists() and any(raw_path.iterdir()):
            downloaded.append(name)
        else:
            missing.append(name)
    return downloaded, missing


def check_splits():
    """Check which datasets have been preprocessed into splits."""
    ready = []
    missing = []
    for name in DATASETS:
        split_dir = SPLITS_DIR / name
        if split_dir.exists() and (split_dir / "train.jsonl").exists():
            ready.append(name)
        else:
            missing.append(name)
    return ready, missing


def check_pretrained_models():
    """Check if pre-trained models are cached in HuggingFace cache."""
    from transformers import AutoTokenizer
    cached = []
    missing = []
    for key, cfg in MODELS.items():
        model_name = cfg["name"]
        try:
            AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            cached.append(model_name)
        except Exception:
            missing.append(model_name)
    return cached, missing


def check_training_data_for_models():
    """Check if each model has at least some of its required training data."""
    issues = []
    for key, cfg in MODELS.items():
        model_name = cfg["name"]
        required_datasets = cfg["datasets"]
        available = []
        for ds_name in required_datasets:
            split_dir = SPLITS_DIR / ds_name
            if split_dir.exists() and (split_dir / "train.jsonl").exists():
                available.append(ds_name)
        if not available:
            issues.append(
                f"  {key} ({model_name}): NO training data available. "
                f"Needs at least one of: {required_datasets}"
            )
    return issues


# ─── Training ────────────────────────────────────────────

def run_command(script, description):
    """Run a Python script and handle errors."""
    print("\n" + "=" * 70)
    print(f"  {description}")
    print("=" * 70)
    try:
        subprocess.run([sys.executable, script], check=True, capture_output=False)
        print(f"\n  ✓ {description} — completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n  ✗ {description} — failed (exit code {e.returncode})")
        return False
    except Exception as e:
        print(f"\n  ✗ {description} — error: {e}")
        return False


def checkpoint_exists(name):
    return (CHECKPOINT_DIR / name).exists()


# ─── Main ────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  PERSON 1 — READINESS CHECK & TRAINING")
    print("=" * 70)

    all_ok = True

    # Check 1: Requirements
    print("\n[CHECK 1] Python packages...")
    missing_pkgs = check_requirements()
    if missing_pkgs:
        print(f"  ✗ MISSING packages: {', '.join(missing_pkgs)}")
        print(f"  → Run: pip install -r requirements.txt")
        all_ok = False
    else:
        print(f"  ✓ All required packages installed")

    # Check 2: Raw datasets
    print("\n[CHECK 2] Raw datasets...")
    downloaded, missing_raw = check_datasets()
    print(f"  ✓ Downloaded: {len(downloaded)} datasets")
    if missing_raw:
        # Separate manual vs auto
        manual_missing = [d for d in missing_raw if DATASETS[d].get("manual_download")]
        auto_missing = [d for d in missing_raw if not DATASETS[d].get("manual_download")]
        if auto_missing:
            print(f"  ✗ MISSING (auto-downloadable): {', '.join(auto_missing)}")
            print(f"  → Run: python scripts/download_datasets.py")
            all_ok = False
        if manual_missing:
            print(f"  ⚠ MISSING (manual download): {', '.join(manual_missing)}")
            for d in manual_missing:
                print(f"      {d}: place files in {RAW_DIR / d}")
            # Manual datasets are optional — don't block training
    else:
        print(f"  ✓ All datasets downloaded")

    # Check 3: Preprocessed splits
    print("\n[CHECK 3] Preprocessed splits...")
    splits_ready, splits_missing = check_splits()
    print(f"  ✓ Ready: {len(splits_ready)} datasets")
    if splits_missing:
        # Only flag as error if auto-downloadable datasets are missing splits
        critical_missing = [
            d for d in splits_missing
            if d in downloaded  # downloaded but not preprocessed
        ]
        if critical_missing:
            print(f"  ✗ NOT preprocessed (but downloaded): {', '.join(critical_missing)}")
            print(f"  → Run: python scripts/preprocess.py")
            all_ok = False
        optional_missing = [d for d in splits_missing if d not in downloaded]
        if optional_missing:
            print(f"  ⚠ Skipped (not downloaded): {', '.join(optional_missing)}")

    # Check 4: Pre-trained models
    print("\n[CHECK 4] Pre-trained models...")
    cached_models, missing_models = check_pretrained_models()
    if cached_models:
        print(f"  ✓ Cached: {', '.join(cached_models)}")
    if missing_models:
        print(f"  ✗ NOT cached: {', '.join(missing_models)}")
        print(f"  → Run: python scripts/download_models.py")
        all_ok = False

    # Check 5: Training data availability per model
    print("\n[CHECK 5] Training data per model...")
    training_issues = check_training_data_for_models()
    if training_issues:
        for issue in training_issues:
            print(f"  ✗ {issue}")
        all_ok = False
    else:
        print(f"  ✓ All models have training data available")

    # ─── Decision ────────────────────────────────────────
    print("\n" + "=" * 70)
    if not all_ok:
        print("  ✗ SETUP INCOMPLETE — cannot start training")
        print()
        print("  Fix the issues above, or run:  python setup_all.py")
        print("  Then re-run:                   python run_all.py")
        print("=" * 70)
        sys.exit(1)

    print("  ✓ ALL CHECKS PASSED — starting training")
    print("=" * 70)

    # ─── Train ───────────────────────────────────────────
    models = [
        ("deberta_ai_detector", "train_deberta.py", "Train DeBERTa-v3-large"),
        ("roberta_ai_detector", "train_roberta.py", "Train RoBERTa-large"),
        ("longformer_ai_detector", "train_longformer.py", "Train Longformer-base"),
        ("xlm_roberta_ai_detector", "train_xlm_roberta.py", "Train XLM-RoBERTa-large"),
    ]

    for ckpt_name, script, desc in models:
        if not checkpoint_exists(ckpt_name):
            success = run_command(script, desc)
            if not success:
                print(f"  ⚠ {ckpt_name} training failed — continuing with next model")
        else:
            print(f"\n  [SKIP] {ckpt_name} already trained")

    # Meta-classifier
    if not checkpoint_exists("meta_classifier.joblib"):
        run_command("train_meta_classifier.py", "Train Meta-Classifier (Ensemble)")
    else:
        print(f"\n  [SKIP] Meta-classifier already trained")

    # Evaluate
    run_command("evaluate.py", "Evaluate Full Ensemble")

    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE")
    print("=" * 70)
    print()
    print("  Checkpoints saved to:", CHECKPOINT_DIR)
    print()
    print("  Usage:")
    print("    from person_1.ai_detector import AIDetector")
    print("    detector = AIDetector()")
    print("    score = detector.detect('your text here')  # 0.0=human, 1.0=AI")
    print()


if __name__ == "__main__":
    main()
