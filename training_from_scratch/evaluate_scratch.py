#!/usr/bin/env python3
"""
Training From Scratch — Evaluation
Evaluates all three from-scratch models on test sets.
Reports accuracy/F1 for classification, correlation for similarity, BLEU for generation.

Usage:  python evaluate_scratch.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import json
import torch
import numpy as np
from collections import Counter

from config_scratch import (
    AI_DETECTOR_CONFIG, PLAGIARISM_DETECTOR_CONFIG, HUMANIZER_CONFIG,
    AI_DETECTOR_DATASETS, PLAGIARISM_DATASETS, HUMANIZER_DATASETS,
    CHECKPOINT_DIR, LOG_DIR, NUM_WORKERS,
)
from models import AIDetectorFromScratch, PlagiarismDetectorFromScratch, HumanizerFromScratch
from data_utils import (
    load_tokenizer, load_dataset_splits, dataset_available,
    extract_texts_and_labels, ClassificationDataset, SentencePairDataset,
    Seq2SeqDataset, create_dataloader,
)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def compute_classification_metrics(predictions, labels):
    preds = np.array(predictions)
    labs = np.array(labels)
    accuracy = (preds == labs).mean()
    tp = ((preds == 1) & (labs == 1)).sum()
    fp = ((preds == 1) & (labs == 0)).sum()
    fn = ((preds == 0) & (labs == 1)).sum()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {
        "accuracy": float(accuracy), "precision": float(precision),
        "recall": float(recall), "f1": float(f1), "total_samples": len(labels),
    }


def compute_bleu(ref_tokens, hyp_tokens, max_n=4):
    scores = []
    for n in range(1, max_n + 1):
        ref_ng = Counter(tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens) - n + 1))
        hyp_ng = Counter(tuple(hyp_tokens[i:i+n]) for i in range(len(hyp_tokens) - n + 1))
        matches = sum(min(hyp_ng[ng], ref_ng[ng]) for ng in hyp_ng)
        total = max(sum(hyp_ng.values()), 1)
        scores.append(matches / total)
    if any(s == 0 for s in scores):
        return 0.0
    log_avg = sum(np.log(s) for s in scores) / len(scores)
    bp = min(1.0, np.exp(1 - len(ref_tokens) / max(len(hyp_tokens), 1)))
    return float(bp * np.exp(log_avg))


@torch.no_grad()
def evaluate_ai_detector():
    print("\n" + "=" * 60)
    print("  Evaluating AI Detector (From Scratch)")
    print("=" * 60)

    config = AI_DETECTOR_CONFIG
    device = get_device()
    tokenizer = load_tokenizer()

    model = AIDetectorFromScratch(config)
    ckpt_path = CHECKPOINT_DIR / "ai_detector_scratch_finetune" / "best" / "checkpoint.pt"
    if not ckpt_path.exists():
        print("  [SKIP] No trained checkpoint found.")
        return None

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    all_texts, all_labels = [], []
    for ds_name in AI_DETECTOR_DATASETS:
        if not dataset_available(ds_name):
            continue
        data = load_dataset_splits(ds_name).get("test", [])
        if data:
            texts, labels = extract_texts_and_labels(data, "classification")
            all_texts.extend(texts)
            all_labels.extend(labels)

    if not all_texts:
        print("  [SKIP] No test data.")
        return None

    print(f"  Test samples: {len(all_texts):,}")
    test_ds = ClassificationDataset(all_texts, all_labels, tokenizer,
                                    config["max_position_embeddings"])
    test_loader = create_dataloader(test_ds, 64, shuffle=False)

    all_preds = []
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        all_preds.extend(outputs["logits"].argmax(dim=-1).cpu().tolist())

    metrics = compute_classification_metrics(all_preds, all_labels)
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    return metrics


@torch.no_grad()
def evaluate_plagiarism_detector():
    print("\n" + "=" * 60)
    print("  Evaluating Plagiarism Detector (From Scratch)")
    print("=" * 60)

    config = PLAGIARISM_DETECTOR_CONFIG
    device = get_device()
    tokenizer = load_tokenizer()

    model = PlagiarismDetectorFromScratch(config)
    ckpt_path = CHECKPOINT_DIR / "plagiarism_detector_scratch_finetune" / "best" / "checkpoint.pt"
    if not ckpt_path.exists():
        print("  [SKIP] No trained checkpoint found.")
        return None

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    all_a, all_b, all_scores = [], [], []
    for ds_name in PLAGIARISM_DATASETS:
        if not dataset_available(ds_name):
            continue
        data = load_dataset_splits(ds_name).get("test", [])
        if data:
            result = extract_texts_and_labels(data, "paraphrase")
            if len(result) == 3:
                a, b, s = result
                all_a.extend(a)
                all_b.extend(b)
                all_scores.extend(s)

    if not all_a:
        print("  [SKIP] No test data.")
        return None

    print(f"  Test pairs: {len(all_a):,}")
    test_ds = SentencePairDataset(all_a, all_b, all_scores, tokenizer,
                                  config["max_position_embeddings"])
    test_loader = create_dataloader(test_ds, 64, shuffle=False)

    all_pred_sims, all_true = [], []
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(
            input_ids_a=batch["input_ids_a"], attention_mask_a=batch["attention_mask_a"],
            input_ids_b=batch["input_ids_b"], attention_mask_b=batch["attention_mask_b"],
        )
        all_pred_sims.extend(outputs["similarity"].cpu().tolist())
        all_true.extend(batch["labels"].cpu().tolist())

    pred_arr = np.array(all_pred_sims)
    true_arr = np.array(all_true)
    mse = float(np.mean((pred_arr - true_arr) ** 2))
    correlation = float(np.corrcoef(pred_arr, true_arr)[0, 1]) if np.std(pred_arr) > 0 and np.std(true_arr) > 0 else 0.0
    binary_acc = float(((pred_arr > 0.5).astype(int) == (true_arr > 0.5).astype(int)).mean())

    metrics = {"mse": mse, "pearson_correlation": correlation,
               "binary_accuracy": binary_acc, "total_pairs": len(all_a)}
    print(f"  MSE:                 {mse:.4f}")
    print(f"  Pearson Correlation: {correlation:.4f}")
    print(f"  Binary Accuracy:     {binary_acc:.4f}")
    return metrics


@torch.no_grad()
def evaluate_humanizer():
    print("\n" + "=" * 60)
    print("  Evaluating Humanizer (From Scratch)")
    print("=" * 60)

    config = HUMANIZER_CONFIG
    device = get_device()
    tokenizer = load_tokenizer()

    model = HumanizerFromScratch(config)
    ckpt_path = CHECKPOINT_DIR / "humanizer_scratch_finetune" / "best" / "checkpoint.pt"
    if not ckpt_path.exists():
        print("  [SKIP] No trained checkpoint found.")
        return None

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    all_src, all_tgt = [], []
    for ds_name in HUMANIZER_DATASETS:
        if not dataset_available(ds_name):
            continue
        data = load_dataset_splits(ds_name).get("test", [])
        if data:
            result = extract_texts_and_labels(data, "paraphrase")
            if len(result) == 3:
                a, b, _ = result
                all_src.extend(a)
                all_tgt.extend(b)

    if not all_src:
        print("  [SKIP] No test data.")
        return None

    eval_size = min(len(all_src), 500)
    all_src = all_src[:eval_size]
    all_tgt = all_tgt[:eval_size]
    print(f"  Test samples: {eval_size}")

    bos_id = tokenizer.token_to_id("[BOS]")
    eos_id = tokenizer.token_to_id("[EOS]")

    bleu_scores = []
    for i, (src, tgt) in enumerate(zip(all_src, all_tgt)):
        enc = tokenizer.encode(src)
        src_ids = torch.tensor([enc.ids[:config["max_position_embeddings"]]],
                               dtype=torch.long, device=device)
        src_mask = torch.ones_like(src_ids)
        generated = model.generate(src_ids, src_mask, max_length=256,
                                   bos_token_id=bos_id, eos_token_id=eos_id)
        gen_text = tokenizer.decode(generated[0].cpu().tolist(), skip_special_tokens=True)
        ref_tokens = tgt.split()
        hyp_tokens = gen_text.split()
        if hyp_tokens:
            bleu_scores.append(compute_bleu(ref_tokens, hyp_tokens))
        if i < 3:
            print(f"\n    Example {i+1}:")
            print(f"    Source:    {src[:100]}...")
            print(f"    Target:    {tgt[:100]}...")
            print(f"    Generated: {gen_text[:100]}...")

    avg_bleu = float(np.mean(bleu_scores)) if bleu_scores else 0.0
    metrics = {"avg_bleu": avg_bleu, "total_samples": eval_size}
    print(f"\n  Average BLEU: {avg_bleu:.4f}")
    return metrics


def main():
    print("\n" + "═" * 70)
    print("  EVALUATION — All From-Scratch Models")
    print("═" * 70)

    results = {}
    ai = evaluate_ai_detector()
    if ai:
        results["ai_detector"] = ai
    plag = evaluate_plagiarism_detector()
    if plag:
        results["plagiarism_detector"] = plag
    human = evaluate_humanizer()
    if human:
        results["humanizer"] = human

    report_path = LOG_DIR / "evaluation_report_scratch.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Report saved to: {report_path}")

    print("\n" + "═" * 70)
    print("  EVALUATION SUMMARY")
    print("═" * 70)
    for model_name, metrics in results.items():
        print(f"\n  {model_name}:")
        for k, v in metrics.items():
            print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")


if __name__ == "__main__":
    main()
