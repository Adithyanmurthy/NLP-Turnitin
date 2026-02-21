"""
Person 1 — Shared Training Utilities
Common training loop, evaluation, and checkpoint management.
"""

import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader as TorchDataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm

from config import CHECKPOINT_DIR, SEED


def set_seed(seed: int = SEED):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_metrics(preds: np.ndarray, labels: np.ndarray) -> dict:
    """Compute all evaluation metrics."""
    pred_labels = (preds >= 0.5).astype(int)
    metrics = {
        "accuracy": accuracy_score(labels, pred_labels),
        "f1": f1_score(labels, pred_labels, average="binary", zero_division=0),
        "precision": precision_score(labels, pred_labels, average="binary", zero_division=0),
        "recall": recall_score(labels, pred_labels, average="binary", zero_division=0),
    }
    try:
        metrics["auroc"] = roc_auc_score(labels, preds)
    except ValueError:
        metrics["auroc"] = 0.0
    return metrics


def train_classifier(
    model_name: str,
    train_dataloader: TorchDataLoader,
    val_dataloader: TorchDataLoader,
    num_epochs: int = 3,
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    gradient_accumulation_steps: int = 1,
    checkpoint_name: Optional[str] = None,
    num_labels: int = 2,
) -> tuple:
    """
    Full training loop for a sequence classification model.
    Returns (model, tokenizer, best_metrics).
    """
    set_seed()
    device = get_device()
    print(f"\nDevice: {device}")

    # Load model and tokenizer
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )
    model.to(device)

    # Optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_params = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_params, lr=learning_rate)

    # Scheduler
    total_steps = len(train_dataloader) * num_epochs // gradient_accumulation_steps
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # FP16 mixed precision — halves GPU memory usage
    use_fp16 = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
    print(f"Mixed precision (fp16): {use_fp16}")

    # Training loop
    best_val_f1 = 0.0
    best_metrics = {}
    save_dir = CHECKPOINT_DIR / (checkpoint_name or model_name.split("/")[-1])

    for epoch in range(num_epochs):
        # --- Train ---
        model.train()
        total_loss = 0
        optimizer.zero_grad()

        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]")
        for step, batch in enumerate(pbar):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=use_fp16):
                outputs = model(**batch)
                loss = outputs.loss / gradient_accumulation_steps
            scaler.scale(loss).backward()
            total_loss += loss.item()

            if (step + 1) % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            pbar.set_postfix({"loss": f"{total_loss / (step + 1):.4f}"})

        avg_train_loss = total_loss / len(train_dataloader)

        # --- Validate ---
        val_metrics = evaluate_classifier(model, val_dataloader, device)
        print(
            f"  Epoch {epoch + 1}: "
            f"train_loss={avg_train_loss:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f} | "
            f"val_auroc={val_metrics['auroc']:.4f}"
        )

        # Save best model
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_metrics = val_metrics
            save_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            print(f"  → Saved best model to {save_dir} (F1={best_val_f1:.4f})")

    return model, tokenizer, best_metrics


def evaluate_classifier(
    model, dataloader: TorchDataLoader, device: torch.device
) -> dict:
    """Evaluate a classifier on a dataloader. Returns metrics dict."""
    model.eval()
    all_preds = []
    all_labels = []
    use_fp16 = device.type == "cuda"

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=use_fp16):
                outputs = model(**batch)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            labels = batch["labels"].cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(labels)

    return compute_metrics(np.array(all_preds), np.array(all_labels))
