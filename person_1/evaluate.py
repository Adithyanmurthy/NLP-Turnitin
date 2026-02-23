from __future__ import annotations
"""
Person 1 — Full Evaluation Script
Evaluates the complete AI detection ensemble on held-out test sets.
Generates a comprehensive evaluation report.
"""

import json
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from config import CHECKPOINT_DIR, MODELS
from data_loader import DataLoader
from ai_detector import AIDetector
from train_utils import compute_metrics, set_seed


def evaluate_on_dataset(
    detector: AIDetector,
    records: list[dict],
    dataset_name: str,
    batch_size: int = 32,
) -> dict:
    """Evaluate the ensemble on a single dataset's test split."""
    texts = [r["text"] for r in records]
    labels = np.array([r["label"] for r in records])

    start_time = time.time()
    scores = detector.detect_batch(texts, batch_size=batch_size)
    elapsed = time.time() - start_time

    preds = np.array(scores)
    metrics = compute_metrics(preds, labels)
    metrics["dataset"] = dataset_name
    metrics["n_samples"] = len(records)
    metrics["time_seconds"] = round(elapsed, 2)
    metrics["samples_per_second"] = round(len(records) / elapsed, 1)

    return metrics


def main():
    print("=" * 60)
    print("  AI DETECTION ENSEMBLE — FULL EVALUATION")
    print("=" * 60)

    set_seed()
    loader = DataLoader()

    # Load detector
    print("\nInitializing AI Detector...")
    detector = AIDetector()

    # Evaluate on each dataset's test split
    test_datasets = ["raid", "hc3", "gpt2_output", "faidset"]
    all_results = []

    for ds_name in test_datasets:
        print(f"\n{'─' * 40}")
        print(f"Evaluating on: {ds_name}")
        try:
            records = loader.load(ds_name, split="test")
            if len(records) > 50_000:
                import random
                random.seed(42)
                records = random.sample(records, 50_000)

            # Small batch size to avoid OOM on Longformer (4096 tokens)
            metrics = evaluate_on_dataset(detector, records, ds_name, batch_size=4)
            all_results.append(metrics)

            print(f"  Samples:    {metrics['n_samples']:,}")
            print(f"  Accuracy:   {metrics['accuracy']:.4f}")
            print(f"  F1:         {metrics['f1']:.4f}")
            print(f"  Precision:  {metrics['precision']:.4f}")
            print(f"  Recall:     {metrics['recall']:.4f}")
            print(f"  AUROC:      {metrics['auroc']:.4f}")
            print(f"  Speed:      {metrics['samples_per_second']:.1f} samples/sec")
        except FileNotFoundError:
            print(f"  [SKIP] Test data not found for {ds_name}")

    # Combined evaluation
    print(f"\n{'─' * 40}")
    print("Combined evaluation (all datasets)...")
    try:
        combined_records = loader.load_combined(
            test_datasets, split="test", max_per_dataset=20_000
        )
        if combined_records:
            combined_metrics = evaluate_on_dataset(
                detector, combined_records, "combined", batch_size=4
            )
            all_results.append(combined_metrics)
            print(f"  Combined Accuracy: {combined_metrics['accuracy']:.4f}")
            print(f"  Combined F1:       {combined_metrics['f1']:.4f}")
            print(f"  Combined AUROC:    {combined_metrics['auroc']:.4f}")
    except Exception as e:
        print(f"  [ERROR] Combined evaluation failed: {e}")

    # Per-model breakdown (detailed analysis)
    print(f"\n{'─' * 40}")
    print("Per-model detailed breakdown (sample)...")
    try:
        sample_records = loader.load("hc3", split="test")[:100]
        if sample_records:
            for record in sample_records[:5]:
                text_preview = record["text"][:80] + "..."
                details = detector.detect_detailed(record["text"])
                true_label = "AI" if record["label"] == 1 else "Human"
                print(f"\n  Text: {text_preview}")
                print(f"  True: {true_label}")
                for model_name, score in details.items():
                    print(f"    {model_name}: {score:.4f}")
    except FileNotFoundError:
        pass

    # Save report
    report = {
        "evaluation_results": all_results,
        "model_checkpoints": list(AIDetector.MODEL_CONFIGS.keys()),
        "meta_classifier": str(CHECKPOINT_DIR / "meta_classifier.joblib"),
    }

    report_path = CHECKPOINT_DIR / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved to {report_path}")

    # Print summary table
    print(f"\n{'=' * 60}")
    print("  EVALUATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"  {'Dataset':<15} {'Acc':>8} {'F1':>8} {'AUROC':>8} {'N':>10}")
    print(f"  {'─' * 51}")
    for r in all_results:
        print(
            f"  {r['dataset']:<15} "
            f"{r['accuracy']:>8.4f} "
            f"{r['f1']:>8.4f} "
            f"{r['auroc']:>8.4f} "
            f"{r['n_samples']:>10,}"
        )
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
