"""
Person 1 — Train Longformer-base
Long document AI detection (up to 4096 tokens). Trained on RAID long documents.
"""

from config import MODELS, CHECKPOINT_DIR
from data_loader import DataLoader
from train_utils import train_classifier, set_seed

from torch.utils.data import DataLoader as TorchDataLoader


def main():
    cfg = MODELS["longformer"]
    model_name = cfg["name"]
    print("=" * 60)
    print(f"  Training: {model_name}")
    print(f"  Datasets: {cfg['datasets']}")
    print(f"  Max Length: {cfg['max_length']} tokens")
    print("=" * 60)

    set_seed()
    loader = DataLoader()

    # For Longformer, we specifically want long documents
    # Filter to keep only texts > 512 tokens worth of content (~2000 chars)
    print("\nLoading training data (long documents)...")
    raw_train = loader.load_combined(
        dataset_names=cfg["datasets"],
        split="train",
        max_per_dataset=20_000,  # Cap per dataset — keeps training ~2-3h per model
    )
    # Keep only long texts (>2000 chars ≈ >512 tokens) for Longformer's advantage
    raw_train = [r for r in raw_train if len(r.get("text", "")) > 2000]
    print(f"  Filtered to {len(raw_train):,} long documents (>2000 chars)")

    from data_loader import TextClassificationDataset
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset = TextClassificationDataset(raw_train, tokenizer, cfg["max_length"])

    raw_val = loader.load_combined(
        dataset_names=cfg["datasets"],
        split="val",
        max_per_dataset=5_000,
    )
    raw_val = [r for r in raw_val if len(r.get("text", "")) > 2000]
    print(f"  Filtered to {len(raw_val):,} long val documents")
    val_dataset = TextClassificationDataset(raw_val, tokenizer, cfg["max_length"])

    train_dl = TorchDataLoader(
        train_dataset, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=0, pin_memory=True,
    )
    val_dl = TorchDataLoader(
        val_dataset, batch_size=cfg["batch_size"] * 2, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    print(f"\nTrain samples: {len(train_dataset):,}")
    print(f"Val samples:   {len(val_dataset):,}")

    model, tokenizer, best_metrics = train_classifier(
        model_name=model_name,
        train_dataloader=train_dl,
        val_dataloader=val_dl,
        num_epochs=cfg["epochs"],
        learning_rate=cfg["learning_rate"],
        warmup_ratio=cfg["warmup_ratio"],
        weight_decay=cfg["weight_decay"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        checkpoint_name="longformer_ai_detector",
    )

    print(f"\n{'=' * 60}")
    print(f"  Longformer Training Complete")
    print(f"  Best Metrics: {best_metrics}")
    print(f"  Checkpoint:   {CHECKPOINT_DIR / 'longformer_ai_detector'}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
