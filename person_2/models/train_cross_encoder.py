"""
Training script for Cross-Encoder (RoBERTa-large based)
Fine-tunes on PAWS + MRPC + STS for pairwise similarity verification.
Uses num_labels=1 (regression/sigmoid) to avoid size mismatch with pretrained checkpoints.
"""

import os
import argparse
from sentence_transformers import CrossEncoder, InputExample
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
from torch.utils.data import DataLoader
from datasets import load_dataset


def load_paws_for_cross_encoder():
    """Load PAWS dataset for cross-encoder training."""
    print("Loading PAWS...")
    dataset = load_dataset("paws", "labeled_final")
    train_examples = [
        InputExample(texts=[item['sentence1'], item['sentence2']], label=float(item['label']))
        for item in dataset['train']
    ]
    val_examples = [
        InputExample(texts=[item['sentence1'], item['sentence2']], label=float(item['label']))
        for item in dataset['validation']
    ]
    return train_examples, val_examples


def load_mrpc_for_cross_encoder():
    """Load MRPC dataset for cross-encoder training."""
    print("Loading MRPC...")
    dataset = load_dataset("glue", "mrpc")
    train_examples = [
        InputExample(texts=[item['sentence1'], item['sentence2']], label=float(item['label']))
        for item in dataset['train']
    ]
    val_examples = [
        InputExample(texts=[item['sentence1'], item['sentence2']], label=float(item['label']))
        for item in dataset['validation']
    ]
    return train_examples, val_examples


def load_sts_for_cross_encoder():
    """Load STS Benchmark for cross-encoder training."""
    print("Loading STS Benchmark...")
    dataset = load_dataset("mteb/stsbenchmark-sts")
    train_examples = [
        InputExample(
            texts=[item['sentence1'], item['sentence2']],
            label=1.0 if item['score'] >= 3.0 else 0.0
        )
        for item in dataset['train']
    ]
    val_examples = [
        InputExample(
            texts=[item['sentence1'], item['sentence2']],
            label=1.0 if item['score'] >= 3.0 else 0.0
        )
        for item in dataset['validation']
    ]
    return train_examples, val_examples


def train_cross_encoder(
    output_path: str,
    base_model: str = "cross-encoder/quora-roberta-large",
    batch_size: int = 8,
    epochs: int = 3,
    warmup_steps: int = 500,
    use_paws: bool = True,
    use_mrpc: bool = True,
    use_sts: bool = True
):
    """Train Cross-Encoder model using num_labels=1 (sigmoid regression)."""
    # num_labels=1 matches the pretrained checkpoint shape — no size mismatch
    # With 1 label, CrossEncoder uses sigmoid: output in [0,1], perfect for binary similarity
    print(f"Loading base model: {base_model} (num_labels=1, sigmoid mode)")
    model = CrossEncoder(base_model, num_labels=1)

    # Load datasets
    train_examples, val_examples = [], []

    if use_paws:
        t, v = load_paws_for_cross_encoder()
        train_examples.extend(t)
        val_examples.extend(v)
        print(f"  PAWS: {len(t)} train, {len(v)} val")

    if use_mrpc:
        t, v = load_mrpc_for_cross_encoder()
        train_examples.extend(t)
        val_examples.extend(v)
        print(f"  MRPC: {len(t)} train, {len(v)} val")

    if use_sts:
        t, v = load_sts_for_cross_encoder()
        train_examples.extend(t)
        val_examples.extend(v)
        print(f"  STS:  {len(t)} train, {len(v)} val")

    print(f"\nTotal: {len(train_examples)} train, {len(val_examples)} val")

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    evaluator = CEBinaryClassificationEvaluator.from_input_examples(val_examples, name='validation')

    print(f"\nTraining: batch_size={batch_size}, epochs={epochs}, warmup={warmup_steps}")
    print(f"Output: {output_path}\n")

    model.fit(
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=output_path,
        evaluation_steps=1000,
        save_best_model=True,
        show_progress_bar=True
    )

    print(f"\nTraining complete! Model saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Cross-Encoder for plagiarism verification")
    parser.add_argument("--output_path", type=str, default="../checkpoints/cross_encoder")
    parser.add_argument("--base_model", type=str, default="cross-encoder/quora-roberta-large")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--no_paws", action="store_true")
    parser.add_argument("--no_mrpc", action="store_true")
    parser.add_argument("--no_sts", action="store_true")

    args = parser.parse_args()
    train_cross_encoder(
        output_path=args.output_path,
        base_model=args.base_model,
        batch_size=args.batch_size,
        epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        use_paws=not args.no_paws,
        use_mrpc=not args.no_mrpc,
        use_sts=not args.no_sts
    )
