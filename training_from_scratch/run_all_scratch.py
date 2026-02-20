#!/usr/bin/env python3
"""
Training From Scratch — Master Script
Runs the complete from-scratch training pipeline.

Usage:
    python run_all_scratch.py                  # Run everything
    python run_all_scratch.py --check          # Status check only
    python run_all_scratch.py --only tok       # Only train tokenizer
    python run_all_scratch.py --only ai        # Only train AI detector
    python run_all_scratch.py --only plag      # Only train plagiarism detector
    python run_all_scratch.py --only human     # Only train humanizer
    python run_all_scratch.py --only eval      # Only evaluate
    python run_all_scratch.py --skip-eval      # Train all, skip evaluation

Multi-GPU (launch individual scripts directly):
    torchrun --nproc_per_node=8 train_ai_detector_scratch.py
    torchrun --nproc_per_node=8 train_plagiarism_detector_scratch.py
    torchrun --nproc_per_node=8 train_humanizer_scratch.py

Environment variables:
    SCRATCH_NUM_WORKERS=4    # DataLoader workers (default: 0 for Windows)
"""

import sys
import os
import time
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))


def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m {int(s)}s"
    else:
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        return f"{int(h)}h {int(m)}m"


def fmt_size(bytes_val):
    if bytes_val == 0:
        return "—"
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_val < 1024:
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024
    return f"{bytes_val:.1f} TB"


def get_size(path):
    path = Path(path)
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    try:
        for f in path.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
    except (PermissionError, OSError):
        pass
    return total


def check_status():
    """Check readiness: datasets, tokenizer, checkpoints, GPU."""
    from config_scratch import (
        SPLITS_DIR, VOCAB_DIR, CHECKPOINT_DIR,
        AI_DETECTOR_DATASETS, PLAGIARISM_DATASETS, HUMANIZER_DATASETS, ALL_DATASETS,
    )

    print("\n" + "=" * 70)
    print("  TRAINING FROM SCRATCH — STATUS CHECK")
    print("=" * 70)

    # ─── 1. GPU ──────────────────────────────────────────
    print("\n  [1] GPU")
    print("  " + "─" * 50)
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                vram = torch.cuda.get_device_properties(i).total_mem / 1e9
                print(f"    ✓ GPU {i}: {name} ({vram:.1f} GB)")
            print(f"    → {torch.cuda.device_count()} GPU(s) available")
        else:
            print("    ✗ No GPU detected (CPU only — training will be very slow)")
    except ImportError:
        print("    ✗ PyTorch not installed")

    # ─── 2. Dependencies ─────────────────────────────────
    print("\n  [2] DEPENDENCIES")
    print("  " + "─" * 50)
    deps = {"torch": "torch", "tokenizers": "tokenizers", "numpy": "numpy"}
    for import_name, pip_name in deps.items():
        try:
            __import__(import_name)
            print(f"    ✓ {pip_name}")
        except ImportError:
            print(f"    ✗ {pip_name} — NOT installed")

    # ─── 3. Datasets ─────────────────────────────────────
    print("\n  [3] PREPROCESSED DATASETS (from person_1/data/splits/)")
    print("  " + "─" * 50)
    ds_ok, ds_missing = [], []
    for name in ALL_DATASETS:
        split_path = SPLITS_DIR / name
        has_all = all((split_path / f"{s}.jsonl").exists() for s in ["train", "val", "test"])
        if has_all:
            size = fmt_size(get_size(split_path))
            print(f"    ✓ {name:<28} {size:>10}")
            ds_ok.append(name)
        else:
            print(f"    ✗ {name:<28} {'—':>10}")
            ds_missing.append(name)
    print(f"    → {len(ds_ok)}/{len(ALL_DATASETS)} datasets available")

    # Per-model coverage
    for label, ds_list in [("AI Detector", AI_DETECTOR_DATASETS),
                           ("Plagiarism Detector", PLAGIARISM_DATASETS),
                           ("Humanizer", HUMANIZER_DATASETS)]:
        available = sum(1 for d in ds_list if d in ds_ok)
        print(f"    → {label}: {available}/{len(ds_list)} datasets")

    # ─── 4. Tokenizer ────────────────────────────────────
    print("\n  [4] TOKENIZER")
    print("  " + "─" * 50)
    tok_path = VOCAB_DIR / "tokenizer.json"
    if tok_path.exists():
        print(f"    ✓ Tokenizer trained ({fmt_size(get_size(tok_path))})")
    else:
        print(f"    ✗ Tokenizer not trained yet")

    # ─── 5. Checkpoints ──────────────────────────────────
    print("\n  [5] TRAINED CHECKPOINTS")
    print("  " + "─" * 50)
    checkpoint_names = [
        ("AI Detector (pretrain)", "ai_detector_scratch_pretrain"),
        ("AI Detector (finetune)", "ai_detector_scratch_finetune"),
        ("Plagiarism Det. (pretrain)", "plagiarism_detector_scratch_pretrain"),
        ("Plagiarism Det. (finetune)", "plagiarism_detector_scratch_finetune"),
        ("Humanizer (pretrain)", "humanizer_scratch_pretrain"),
        ("Humanizer (finetune)", "humanizer_scratch_finetune"),
    ]
    ckpts_ok = 0
    for label, name in checkpoint_names:
        best_path = CHECKPOINT_DIR / name / "best" / "checkpoint.pt"
        if best_path.exists():
            size = fmt_size(get_size(best_path))
            print(f"    ✓ {label:<35} {size:>10}")
            ckpts_ok += 1
        else:
            print(f"    ✗ {label:<35} {'—':>10}")
    print(f"    → {ckpts_ok}/{len(checkpoint_names)} checkpoints ready")

    # ─── Summary ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    if not tok_path.exists():
        print("  → Next step: python run_all_scratch.py --only tok")
    elif ckpts_ok == 0:
        print("  → Tokenizer ready. Next: train models")
        print("    Single GPU:  python run_all_scratch.py")
        print("    Multi-GPU:   torchrun --nproc_per_node=8 train_ai_detector_scratch.py")
    elif ckpts_ok < len(checkpoint_names):
        print(f"  → {ckpts_ok}/{len(checkpoint_names)} models trained. Continue training.")
    else:
        print("  → All models trained! Run evaluation:")
        print("    python run_all_scratch.py --only eval")
    print()


def print_banner():
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║                                                                    ║")
    print("║   Training From Scratch — Custom Transformer Models                ║")
    print("║   ─────────────────────────────────────────                         ║")
    print("║   No pretrained weights. Everything learned from our datasets.     ║")
    print("║                                                                    ║")
    print("║   Models:                                                          ║")
    print("║     1. AI Detector      — 12-layer Transformer Encoder (~85M)      ║")
    print("║     2. Plagiarism Det.  — Siamese Transformer Encoder  (~85M)      ║")
    print("║     3. Humanizer        — Encoder-Decoder Transformer  (~60M)      ║")
    print("║                                                                    ║")
    print("║   Multi-GPU:                                                       ║")
    print("║     torchrun --nproc_per_node=N train_<model>_scratch.py           ║")
    print("║                                                                    ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()


def run_step(name, func):
    print(f"\n{'━' * 70}")
    print(f"  ▶ {name}")
    print(f"{'━' * 70}")
    start = time.time()
    try:
        func()
        elapsed = time.time() - start
        print(f"\n  ✓ {name} — completed in {format_time(elapsed)}")
        return True, elapsed
    except Exception as e:
        elapsed = time.time() - start
        print(f"\n  ✗ {name} — failed: {e}")
        import traceback
        traceback.print_exc()
        return False, elapsed


def main():
    parser = argparse.ArgumentParser(description="Train all models from scratch")
    parser.add_argument("--check", action="store_true", help="Status check only")
    parser.add_argument("--only", choices=["tok", "ai", "plag", "human", "eval"],
                        help="Run only a specific step")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip evaluation after training")
    args = parser.parse_args()

    print_banner()

    if args.check:
        check_status()
        return

    # Always show status first
    check_status()

    total_start = time.time()
    results = {}

    if args.only in (None, "tok"):
        from train_tokenizer import train_tokenizer
        ok, t = run_step("Step 1: Train BPE Tokenizer", train_tokenizer)
        results["Tokenizer"] = (ok, t)

    if args.only in (None, "ai"):
        from train_ai_detector_scratch import main as train_ai
        ok, t = run_step("Step 2: Train AI Detector (MLM + Classification)", train_ai)
        results["AI Detector"] = (ok, t)

    if args.only in (None, "plag"):
        from train_plagiarism_detector_scratch import main as train_plag
        ok, t = run_step("Step 3: Train Plagiarism Detector (MLM + Similarity)", train_plag)
        results["Plagiarism Detector"] = (ok, t)

    if args.only in (None, "human"):
        from train_humanizer_scratch import main as train_human
        ok, t = run_step("Step 4: Train Humanizer (Denoising + Paraphrase)", train_human)
        results["Humanizer"] = (ok, t)

    if not args.skip_eval and args.only in (None, "eval"):
        from evaluate_scratch import main as eval_all
        ok, t = run_step("Step 5: Evaluate All Models", eval_all)
        results["Evaluation"] = (ok, t)

    # Summary
    total_time = time.time() - total_start
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║                    TRAINING FROM SCRATCH — SUMMARY                 ║")
    print("╠══════════════════════════════════════════════════════════════════════╣")
    for name, (ok, t) in results.items():
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"║  {status}  {name:<30s}  {format_time(t):>10s}              ║")
    print("╠══════════════════════════════════════════════════════════════════════╣")
    print(f"║  Total time: {format_time(total_time):>10s}                                       ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()


if __name__ == "__main__":
    main()
