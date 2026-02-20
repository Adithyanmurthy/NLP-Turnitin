"""
Person 1 — Train DeBERTa-v3-large
Primary AI classifier. Trained on RAID + HC3 + M4 + FAIDSet.
"""

import sys
from pathlib import Path

from config import MODELS, CHECKPOINT_DIR
from data_loader import DataLoader
from train_utils import train_classifier, set_seed, get_device

import torch
from torch.utils.data import DataLoader as TorchDataLoader


def main():
    cfg = MODELS["deberta"]
    model_name = cfg["name"]
    print("=" * 60)
    print(f"  Training: {model_name}")
    print(f"  Datasets: {cfg['datasets']}")
    print("=" * 60)

    set_seed()
    loader = DataLoader()

    # Load combined training data
    print("\nLoading training data...")
    train_dataset = loader.get_combined_torch_dataset(
        dataset_names=cfg["datasets"],
        split="train",
        tokenizer_name=model_name,
        max_length=cfg["max_length"],
        max_per_dataset=500_000,  # Cap per dataset to balance
    )

    print("Loading validation data...")
    val_dataset = loader.get_combined_torch_dataset(
        dataset_names=cfg["datasets"],
        split="val",
        tokenizer_name=model_name,
        max_length=cfg["max_length"],
        max_per_dataset=50_000,
    )

    train_dl = TorchDataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_dl = TorchDataLoader(
        val_dataset,
        batch_size=cfg["batch_size"] * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    print(f"\nTrain samples: {len(train_dataset):,}")
    print(f"Val samples:   {len(val_dataset):,}")

    # Train
    model, tokenizer, best_metrics = train_classifier(
        model_name=model_name,
        train_dataloader=train_dl,
        val_dataloader=val_dl,
        num_epochs=cfg["epochs"],
        learning_rate=cfg["learning_rate"],
        warmup_ratio=cfg["warmup_ratio"],
        weight_decay=cfg["weight_decay"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        checkpoint_name="deberta_ai_detector",
    )

    print(f"\n{'=' * 60}")
    print(f"  DeBERTa Training Complete")
    print(f"  Best Metrics: {best_metrics}")
    print(f"  Checkpoint:   {CHECKPOINT_DIR / 'deberta_ai_detector'}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
