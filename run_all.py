#!/usr/bin/env python3
"""
UNIFIED TRAINING — Checks everything, then trains Person 1 → 2 → 3 in order.

BEFORE running this, run:  python setup_all.py

Usage:
    python run_all.py

What it does:
  1. Checks all requirements, datasets, splits, and pre-trained models
  2. Reports anything missing (and stops if critical things are missing)
  3. Trains Person 1: DeBERTa, RoBERTa, Longformer, XLM-RoBERTa, Meta-classifier
  4. Trains Person 2: Sentence-BERT, Cross-Encoder
  5. Trains Person 3: Flan-T5-XL, PEGASUS-large, Mistral-7B (QLoRA)
  6. Evaluates Person 1 ensemble
  7. Prints summary + next steps (Person 4 integration)

All checkpoints are saved under each person's checkpoints/ folder.
"""

import sys
import subprocess
from pathlib import Path

ROOT_DIR = Path(__file__).parent


# ═══════════════════════════════════════════════════════════
#  READINESS CHECKS
# ═══════════════════════════════════════════════════════════

def check_packages():
    """Check if all key packages are installed."""
    missing = []
    packages = {
        "torch": "torch",
        "transformers": "transformers",
        "datasets": "datasets",
        "sentence_transformers": "sentence-transformers",
        "sklearn": "scikit-learn",
        "numpy": "numpy",
        "tqdm": "tqdm",
        "accelerate": "accelerate",
        "joblib": "joblib",
        "peft": "peft",
        "fastapi": "fastapi",
    }
    for import_name, pip_name in packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pip_name)
    return missing


def check_person1_data():
    """Check Person 1 datasets and splits."""
    splits_dir = ROOT_DIR / "person_1" / "data" / "splits"
    raw_dir = ROOT_DIR / "person_1" / "data" / "raw"

    # Key datasets that Person 1 training needs
    # m4 is manual-download only and not available — skip it
    critical_datasets = ["raid", "hc3", "gpt2_output", "faidset"]
    downloaded = []
    missing_raw = []
    missing_splits = []

    for name in critical_datasets:
        raw_path = raw_dir / name
        split_path = splits_dir / name / "train.jsonl"
        if raw_path.exists() and any(raw_path.iterdir()):
            downloaded.append(name)
        else:
            missing_raw.append(name)
        if not split_path.exists():
            missing_splits.append(name)

    return downloaded, missing_raw, missing_splits


def check_person3_data():
    """Check Person 3 datasets."""
    data_dir = ROOT_DIR / "person_3" / "data"
    train_file = data_dir / "train.jsonl"
    # Person 3 can also fall back to Person 1's splits
    p1_splits = ROOT_DIR / "person_1" / "data" / "splits"
    p3_paraphrase_datasets = ["paws", "qqp", "mrpc", "paranmt", "wikisplit"]

    if train_file.exists():
        return True, "Person 3 own data"

    for ds in p3_paraphrase_datasets:
        if (p1_splits / ds / "train.jsonl").exists():
            return True, "Person 1 splits (fallback)"

    return False, "No data found"


def check_pretrained_models():
    """Check if pre-trained models are cached."""
    cached = []
    missing = []

    models = [
        ("P1: deberta-v3-large", "microsoft/deberta-v3-large"),
        ("P1: roberta-large", "roberta-large"),
        ("P1: longformer-base", "allenai/longformer-base-4096"),
        ("P1: xlm-roberta-large", "xlm-roberta-large"),
        ("P2: all-mpnet-base-v2", "sentence-transformers/all-mpnet-base-v2"),
        ("P2: nli-deberta-v3-large", "cross-encoder/nli-deberta-v3-large"),
        ("P3: flan-t5-xl", "google/flan-t5-xl"),
        ("P3: pegasus-large", "google/pegasus-large"),
    ]

    from transformers import AutoTokenizer
    for label, model_name in models:
        try:
            AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            cached.append(label)
        except Exception:
            missing.append(label)

    return cached, missing


# ═══════════════════════════════════════════════════════════
#  TRAINING HELPERS
# ═══════════════════════════════════════════════════════════

def run_command(script, description, cwd=None):
    """Run a Python script and handle errors."""
    print("\n" + "=" * 70)
    print(f"  {description}")
    print("=" * 70)
    try:
        subprocess.run(
            [sys.executable, script],
            check=True,
            capture_output=False,
            cwd=cwd,
        )
        print(f"\n  ✓ {description} — completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n  ✗ {description} — failed (exit code {e.returncode})")
        return False
    except Exception as e:
        print(f"\n  ✗ {description} — error: {e}")
        return False


def checkpoint_exists(person_dir, name):
    return (ROOT_DIR / person_dir / "checkpoints" / name).exists()


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  UNIFIED TRAINING PIPELINE")
    print("  Person 1 (AI Detection) → Person 2 (Plagiarism) → Person 3 (Humanization)")
    print("=" * 70)

    all_ok = True
    py = sys.executable

    # ─── CHECK 1: Packages ───────────────────────────────
    print("\n[CHECK 1] Python packages...")
    missing_pkgs = check_packages()
    if missing_pkgs:
        print(f"  ✗ MISSING: {', '.join(missing_pkgs)}")
        print(f"  → Run: python setup_all.py")
        all_ok = False
    else:
        print(f"  ✓ All required packages installed")

    # ─── CHECK 2: Person 1 data ──────────────────────────
    print("\n[CHECK 2] Person 1 datasets...")
    downloaded, missing_raw, missing_splits = check_person1_data()
    if downloaded:
        print(f"  ✓ Downloaded: {', '.join(downloaded)}")
    if missing_raw:
        print(f"  ✗ NOT downloaded: {', '.join(missing_raw)}")
        print(f"  → Run: python setup_all.py")
        all_ok = False
    if missing_splits and not missing_raw:
        print(f"  ✗ NOT preprocessed: {', '.join(missing_splits)}")
        print(f"  → Run: python setup_all.py")
        all_ok = False

    # ─── CHECK 3: Person 3 data ──────────────────────────
    print("\n[CHECK 3] Person 3 datasets...")
    p3_ok, p3_source = check_person3_data()
    if p3_ok:
        print(f"  ✓ Data available ({p3_source})")
    else:
        print(f"  ✗ No training data for Person 3")
        print(f"  → Run: python setup_all.py")
        all_ok = False

    # ─── CHECK 4: Pre-trained models ─────────────────────
    print("\n[CHECK 4] Pre-trained models...")
    cached, missing_models = check_pretrained_models()
    if cached:
        for m in cached:
            print(f"  ✓ {m}")
    if missing_models:
        for m in missing_models:
            print(f"  ✗ {m} — NOT cached")
        print(f"  → Run: python setup_all.py")
        all_ok = False

    # ─── DECISION ────────────────────────────────────────
    print("\n" + "=" * 70)
    if not all_ok:
        print("  ✗ SETUP INCOMPLETE — cannot start training")
        print()
        print("  Run first:  python setup_all.py")
        print("  Then retry: python run_all.py")
        print("=" * 70)
        sys.exit(1)

    print("  ✓ ALL CHECKS PASSED — starting training")
    print("=" * 70)

    p1_dir = str(ROOT_DIR / "person_1")
    p2_dir = str(ROOT_DIR / "person_2")
    p3_dir = str(ROOT_DIR / "person_3")

    # ═════════════════════════════════════════════════════
    #  PERSON 1 — AI Detection (4 models + meta-classifier)
    # ═════════════════════════════════════════════════════
    print("\n\n" + "█" * 70)
    print("  PERSON 1 — AI DETECTION TRAINING")
    print("█" * 70)

    p1_models = [
        ("deberta_ai_detector", "train_deberta.py", "P1: Train DeBERTa-v3-large"),
        ("roberta_ai_detector", "train_roberta.py", "P1: Train RoBERTa-large"),
        ("longformer_ai_detector", "train_longformer.py", "P1: Train Longformer-base"),
        ("xlm_roberta_ai_detector", "train_xlm_roberta.py", "P1: Train XLM-RoBERTa-large"),
    ]

    for ckpt_name, script, desc in p1_models:
        if not checkpoint_exists("person_1", ckpt_name):
            success = run_command(script, desc, cwd=p1_dir)
            if not success:
                print(f"  ⚠ {ckpt_name} failed — continuing with next model")
        else:
            print(f"\n  [SKIP] {ckpt_name} already trained")

    # Meta-classifier
    if not checkpoint_exists("person_1", "meta_classifier.joblib"):
        run_command("train_meta_classifier.py", "P1: Train Meta-Classifier (Ensemble)", cwd=p1_dir)
    else:
        print(f"\n  [SKIP] P1 meta-classifier already trained")

    # Evaluate Person 1
    run_command("evaluate.py", "P1: Evaluate Full Ensemble", cwd=p1_dir)

    # ═════════════════════════════════════════════════════
    #  PERSON 2 — Plagiarism Detection (2 models)
    # ═════════════════════════════════════════════════════
    print("\n\n" + "█" * 70)
    print("  PERSON 2 — PLAGIARISM DETECTION TRAINING")
    print("█" * 70)

    # Build reference index (if corpus data exists)
    p2_data = ROOT_DIR / "person_2" / "data"
    p2_index = ROOT_DIR / "person_2" / "reference_index"
    if not (p2_index / "metadata.json").exists():
        if p2_data.exists() and any(p2_data.iterdir()):
            run_command(
                str(ROOT_DIR / "person_2" / "scripts" / "build_index.py")
                + f" --corpus_path {p2_data} --output_path {p2_index}",
                "P2: Build MinHash/LSH Reference Index",
                cwd=p2_dir,
            )
        else:
            print("\n  [SKIP] No corpus data in person_2/data/ — skipping index build")
    else:
        print(f"\n  [SKIP] P2 reference index already exists")

    # Train Sentence-BERT
    sbert_path = ROOT_DIR / "person_2" / "checkpoints" / "sbert"
    if not sbert_path.exists():
        run_command(
            "models/train_sentence_bert.py",
            "P2: Train Sentence-BERT (all-mpnet-base-v2)",
            cwd=p2_dir,
        )
    else:
        print(f"\n  [SKIP] P2 Sentence-BERT already trained")

    # Train Cross-Encoder
    ce_path = ROOT_DIR / "person_2" / "checkpoints" / "cross_encoder"
    if not ce_path.exists():
        run_command(
            "models/train_cross_encoder.py",
            "P2: Train Cross-Encoder (DeBERTa-v3)",
            cwd=p2_dir,
        )
    else:
        print(f"\n  [SKIP] P2 Cross-Encoder already trained")

    # ═════════════════════════════════════════════════════
    #  PERSON 3 — Humanization (3 models + optional DIPPER)
    # ═════════════════════════════════════════════════════
    print("\n\n" + "█" * 70)
    print("  PERSON 3 — HUMANIZATION TRAINING")
    print("█" * 70)

    # Flan-T5-XL
    if not checkpoint_exists("person_3", "flan_t5_xl_final"):
        run_command("train_flan_t5.py", "P3: Train Flan-T5-XL", cwd=p3_dir)
    else:
        print(f"\n  [SKIP] P3 Flan-T5-XL already trained")

    # PEGASUS-large
    if not checkpoint_exists("person_3", "pegasus_large_final"):
        run_command("train_pegasus.py", "P3: Train PEGASUS-large", cwd=p3_dir)
    else:
        print(f"\n  [SKIP] P3 PEGASUS-large already trained")

    # Mistral-7B QLoRA
    if not checkpoint_exists("person_3", "mistral_7b_qlora_final"):
        run_command("train_mistral.py", "P3: Train Mistral-7B (QLoRA)", cwd=p3_dir)
    else:
        print(f"\n  [SKIP] P3 Mistral-7B already trained")

    # DIPPER (pretrained, just download and save to checkpoints)
    if not checkpoint_exists("person_3", "dipper_xxl"):
        run_command("setup_dipper.py", "P3: Setup DIPPER (11B paraphraser)", cwd=p3_dir)
    else:
        print(f"\n  [SKIP] P3 DIPPER already set up")

    # ═════════════════════════════════════════════════════
    #  DONE
    # ═════════════════════════════════════════════════════
    print("\n\n" + "█" * 70)
    print("  ALL TRAINING COMPLETE")
    print("█" * 70)
    print()
    print("  Checkpoints saved in:")
    print(f"    Person 1 (AI Detection):      person_1/checkpoints/")
    print(f"    Person 2 (Plagiarism):         person_2/checkpoints/")
    print(f"    Person 3 (Humanization):       person_3/checkpoints/")
    print()
    print("  What was trained:")
    print("    P1: DeBERTa-v3-large, RoBERTa-large, Longformer-base,")
    print("        XLM-RoBERTa-large, Meta-classifier (ensemble)")
    print("    P2: Sentence-BERT, Cross-Encoder")
    print("    P3: Flan-T5-XL, PEGASUS-large, Mistral-7B (QLoRA), DIPPER (11B)")
    print()
    print("  NEXT STEP:")
    print("    Person 4 integration is ready to use. Run:")
    print("      python person_4/main.py --input \"your text here\" --full")
    print()
    print("    Or start the web API:")
    print("      python person_4/run_server.py")
    print()
    print("    Person 4 automatically loads trained checkpoints from P1, P2, P3.")
    print("█" * 70)


if __name__ == "__main__":
    main()
