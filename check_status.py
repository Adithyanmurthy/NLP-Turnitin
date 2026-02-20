#!/usr/bin/env python3
"""
STATUS CHECK — Shows exactly what's downloaded, preprocessed, and trained.

Usage:
    python check_status.py
"""

import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent


def get_size(path):
    """Get size of a file or directory."""
    path = Path(path)
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total


def fmt_size(bytes_val):
    """Format bytes to human readable."""
    if bytes_val == 0:
        return "0 B"
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_val < 1024:
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024
    return f"{bytes_val:.1f} PB"


def count_files(path):
    """Count files in a directory."""
    path = Path(path)
    if not path.exists():
        return 0
    return sum(1 for f in path.rglob("*") if f.is_file())


def check_mark(condition):
    return "✓" if condition else "✗"


def main():
    print("=" * 70)
    print("  PROJECT STATUS CHECK")
    print("=" * 70)

    total_done = 0
    total_items = 0

    # ─── 1. PIP PACKAGES ────────────────────────────────
    print("\n[1] PIP PACKAGES")
    print("─" * 50)
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
        "bitsandbytes": "bitsandbytes",
        "fastapi": "fastapi",
        "nltk": "nltk",
        "wandb": "wandb",
        "sentencepiece": "sentencepiece",
    }
    pkg_installed = 0
    for import_name, pip_name in packages.items():
        try:
            __import__(import_name)
            print(f"  ✓ {pip_name}")
            pkg_installed += 1
        except ImportError:
            print(f"  ✗ {pip_name} — NOT installed")
    total_done += pkg_installed
    total_items += len(packages)
    print(f"  → {pkg_installed}/{len(packages)} packages installed")

    # ─── 2. RAW DATASETS (Person 1) ─────────────────────
    print("\n[2] RAW DATASETS (person_1/data/raw/)")
    print("─" * 50)
    raw_dir = ROOT_DIR / "person_1" / "data" / "raw"
    datasets_expected = {
        "raid": "RAID benchmark",
        "hc3": "HC3 (Human vs ChatGPT)",
        "m4": "M4 (Multi-generator)",
        "gpt2_output": "GPT-wiki-intro",
        "faidset": "FAIDSet (Fine-grained AI detection)",
        "pan_author_id": "PAN Author ID [MANUAL]",
        "pan_plagiarism": "PAN Plagiarism [MANUAL]",
        "clough_stevenson": "Clough & Stevenson",
        "webis_crowd_paraphrase": "Webis Crowd Paraphrase",
        "wikisplit": "WikiSplit",
        "sts_benchmark": "STS Benchmark",
        "paws": "PAWS",
        "paranmt": "ParaNMT (ChatGPT paraphrases)",
        "qqp": "QQP",
        "mrpc": "MRPC",
        "bea_2019_gec": "BEA-2019 GEC",
    }
    ds_downloaded = 0
    for name, desc in datasets_expected.items():
        path = raw_dir / name
        exists = path.exists() and any(path.iterdir()) if path.exists() else False
        size = get_size(path) if exists else 0
        mark = check_mark(exists)
        size_str = fmt_size(size) if exists else "—"
        print(f"  {mark} {name:<25} {size_str:>10}  ({desc})")
        if exists:
            ds_downloaded += 1
    total_done += ds_downloaded
    total_items += len(datasets_expected)
    print(f"  → {ds_downloaded}/{len(datasets_expected)} datasets downloaded")
    if raw_dir.exists():
        print(f"  → Total raw data size: {fmt_size(get_size(raw_dir))}")

    # ─── 3. PREPROCESSED SPLITS (Person 1) ───────────────
    print("\n[3] PREPROCESSED SPLITS (person_1/data/splits/)")
    print("─" * 50)
    splits_dir = ROOT_DIR / "person_1" / "data" / "splits"
    splits_ready = 0
    for name in datasets_expected:
        split_path = splits_dir / name
        has_train = (split_path / "train.jsonl").exists()
        has_val = (split_path / "val.jsonl").exists()
        has_test = (split_path / "test.jsonl").exists()
        ready = has_train and has_val and has_test
        mark = check_mark(ready)
        if ready:
            train_lines = sum(1 for _ in open(split_path / "train.jsonl"))
            size = get_size(split_path)
            print(f"  {mark} {name:<25} {fmt_size(size):>10}  ({train_lines:,} train samples)")
            splits_ready += 1
        else:
            parts = []
            if has_train: parts.append("train")
            if has_val: parts.append("val")
            if has_test: parts.append("test")
            status = f"has: {', '.join(parts)}" if parts else "not preprocessed"
            print(f"  {mark} {name:<25} {'—':>10}  ({status})")
    total_done += splits_ready
    total_items += len(datasets_expected)
    print(f"  → {splits_ready}/{len(datasets_expected)} datasets preprocessed")

    # ─── 4. PERSON 3 DATASETS ───────────────────────────
    print("\n[4] PERSON 3 DATASETS (person_3/data/)")
    print("─" * 50)
    p3_data = ROOT_DIR / "person_3" / "data"
    p3_files = {"train.jsonl": False, "validation.jsonl": False, "test.jsonl": False, "metadata.json": False}
    p3_count = 0
    for fname in p3_files:
        fpath = p3_data / fname
        exists = fpath.exists()
        p3_files[fname] = exists
        size = fmt_size(get_size(fpath)) if exists else "—"
        print(f"  {check_mark(exists)} {fname:<25} {size:>10}")
        if exists:
            p3_count += 1
    total_done += p3_count
    total_items += len(p3_files)
    print(f"  → {p3_count}/{len(p3_files)} Person 3 data files ready")

    # ─── 5. PRE-TRAINED MODELS (HF Cache) ───────────────
    print("\n[5] PRE-TRAINED MODELS (HuggingFace cache)")
    print("─" * 50)

    hf_home = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    print(f"  Cache location: {hf_home}")
    print(f"  Cache size: {fmt_size(get_size(hf_home))}")
    print()

    models_expected = [
        ("P1", "microsoft/deberta-v3-large", "DeBERTa-v3-large"),
        ("P1", "roberta-large", "RoBERTa-large"),
        ("P1", "allenai/longformer-base-4096", "Longformer-base"),
        ("P1", "xlm-roberta-large", "XLM-RoBERTa-large"),
        ("P2", "sentence-transformers/all-mpnet-base-v2", "Sentence-BERT"),
        ("P2", "cross-encoder/nli-deberta-v3-large", "Cross-Encoder"),
        ("P3", "google/flan-t5-xl", "Flan-T5-XL"),
        ("P3", "google/pegasus-large", "PEGASUS-large"),
        ("P3", "mistralai/Mistral-7B-v0.3", "Mistral-7B"),
        ("P3", "kalpeshk2011/dipper-paraphraser-xxl", "DIPPER (11B)"),
    ]

    models_cached = 0
    for person, model_id, label in models_expected:
        try:
            from transformers import AutoTokenizer
            AutoTokenizer.from_pretrained(model_id, local_files_only=True)
            print(f"  ✓ [{person}] {label:<30} ({model_id})")
            models_cached += 1
        except Exception:
            print(f"  ✗ [{person}] {label:<30} ({model_id}) — NOT cached")
    total_done += models_cached
    total_items += len(models_expected)
    print(f"  → {models_cached}/{len(models_expected)} models cached")

    # ─── 6. TRAINED CHECKPOINTS ──────────────────────────
    print("\n[6] TRAINED CHECKPOINTS")
    print("─" * 50)

    checkpoints = [
        ("P1", "person_1", "deberta_ai_detector", "DeBERTa AI Detector"),
        ("P1", "person_1", "roberta_ai_detector", "RoBERTa AI Detector"),
        ("P1", "person_1", "longformer_ai_detector", "Longformer AI Detector"),
        ("P1", "person_1", "xlm_roberta_ai_detector", "XLM-RoBERTa AI Detector"),
        ("P1", "person_1", "meta_classifier.joblib", "Meta-Classifier (Ensemble)"),
        ("P2", "person_2", "sbert", "Sentence-BERT (fine-tuned)"),
        ("P2", "person_2", "cross_encoder", "Cross-Encoder (fine-tuned)"),
        ("P3", "person_3", "flan_t5_xl_final", "Flan-T5-XL (fine-tuned)"),
        ("P3", "person_3", "pegasus_large_final", "PEGASUS-large (fine-tuned)"),
        ("P3", "person_3", "mistral_7b_qlora_final", "Mistral-7B QLoRA (fine-tuned)"),
        ("P3", "person_3", "dipper_xxl", "DIPPER (pretrained, saved)"),
    ]

    ckpts_done = 0
    for person, folder, name, label in checkpoints:
        path = ROOT_DIR / folder / "checkpoints" / name
        exists = path.exists()
        size = fmt_size(get_size(path)) if exists else "—"
        print(f"  {check_mark(exists)} [{person}] {label:<35} {size:>10}")
        if exists:
            ckpts_done += 1
    total_done += ckpts_done
    total_items += len(checkpoints)
    print(f"  → {ckpts_done}/{len(checkpoints)} checkpoints ready")

    # ─── 7. EVALUATION REPORTS ───────────────────────────
    print("\n[7] EVALUATION")
    print("─" * 50)
    eval_report = ROOT_DIR / "person_1" / "checkpoints" / "evaluation_report.json"
    if eval_report.exists():
        print(f"  ✓ P1 evaluation report exists ({fmt_size(get_size(eval_report))})")
        total_done += 1
    else:
        print(f"  ✗ P1 evaluation not run yet")
    total_items += 1

    # ─── SUMMARY ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  OVERALL PROGRESS")
    print("=" * 70)
    pct = round(total_done / total_items * 100) if total_items > 0 else 0

    bar_len = 40
    filled = int(bar_len * pct / 100)
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"\n  [{bar}] {pct}%  ({total_done}/{total_items} items)")

    # Total disk usage
    print(f"\n  Disk usage:")
    project_size = get_size(ROOT_DIR)
    print(f"    Project folder: {fmt_size(project_size)}")
    print(f"    HF cache:       {fmt_size(get_size(hf_home))}")
    print(f"    TOTAL:          {fmt_size(project_size + get_size(hf_home))}")

    # What to do next
    print(f"\n  NEXT STEPS:")
    if pkg_installed < len(packages):
        print(f"    → Install packages:    python setup_all.py")
    if ds_downloaded < len(datasets_expected):
        print(f"    → Download datasets:   python setup_all.py")
    if splits_ready < ds_downloaded:
        print(f"    → Preprocess data:     python setup_all.py")
    if models_cached < len(models_expected):
        print(f"    → Download models:     python setup_all.py")
    if ckpts_done < len(checkpoints):
        print(f"    → Train models:        python run_all.py")
    if pct == 100:
        print(f"    → Everything done! Run: python person_4/main.py --input \"text\" --full")

    print("=" * 70)


if __name__ == "__main__":
    main()
