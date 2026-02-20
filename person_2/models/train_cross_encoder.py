"""
Training script for DeBERTa-v3 Cross-Encoder
Fine-tunes on PAWS + MRPC + STS for pairwise similarity verification
"""

import os
import argparse
from sentence_transformers import CrossEncoder, InputExample
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator, CESoftmaxAccuracyEvaluator
from torch.utils.data import DataLoader
from datasets import load_dataset


def load_paws_for_cross_encoder():
    """Load PAWS dataset for cross-encoder training."""
    print("Loading PAWS...")
    dataset = load_dataset("paws", "labeled_final")
    
    train_examples = []
    for item in dataset['train']:
        train_examples.append(InputExample(
            texts=[item['sentence1'], item['sentence2']],
            label=int(item['label'])
        ))
    
    val_examples = []
    for item in dataset['validation']:
        val_examples.append(InputExample(
            texts=[item['sentence1'], item['sentence2']],
            label=int(item['label'])
        ))
    
    return train_examples, val_examples


def load_mrpc_for_cross_encoder():
    """Load MRPC dataset for cross-encoder training."""
    print("Loading MRPC...")
    dataset = load_dataset("glue", "mrpc")
    
    train_examples = []
    for item in dataset['train']:
        train_examples.append(InputExample(
            texts=[item['sentence1'], item['sentence2']],
            label=int(item['label'])
        ))
    
    val_examples = []
    for item in dataset['validation']:
        val_examples.append(InputExample(
            texts=[item['sentence1'], item['sentence2']],
            label=int(item['label'])
        ))
    
    return train_examples, val_examples


def load_sts_for_cross_encoder():
    """Load STS Benchmark for cross-encoder training."""
    print("Loading STS Benchmark...")
    dataset = load_dataset("mteb/stsbenchmark-sts")
    
    train_examples = []
    for item in dataset['train']:
        # Convert to binary: score >= 3.0 means similar
        label = 1 if item['score'] >= 3.0 else 0
        train_examples.append(InputExample(
            texts=[item['sentence1'], item['sentence2']],
            label=label
        ))
    
    val_examples = []
    for item in dataset['validation']:
        label = 1 if item['score'] >= 3.0 else 0
        val_examples.append(InputExample(
            texts=[item['sentence1'], item['sentence2']],
            label=label
        ))
    
    return train_examples, val_examples


def train_cross_encoder(
    output_path: str,
    base_model: str = "cross-encoder/nli-deberta-v3-large",
    batch_size: int = 16,
    epochs: int = 3,
    warmup_steps: int = 500,
    use_paws: bool = True,
    use_mrpc: bool = True,
    use_sts: bool = True
):
    """
    Train Cross-Encoder model.
    
    Args:
        output_path: Path to save trained model
        base_model: Base cross-encoder model
        batch_size: Training batch size
        epochs: Number of training epochs
        warmup_steps: Number of warmup steps
        use_paws: Include PAWS
        use_mrpc: Include MRPC
        use_sts: Include STS
    """
    # Load model
    print(f"Loading base model: {base_model}")
    model = CrossEncoder(base_model, num_labels=2)
    
    # Load datasets
    train_examples = []
    val_examples = []
    
    if use_paws:
        paws_train, paws_val = load_paws_for_cross_encoder()
        train_examples.extend(paws_train)
        val_examples.extend(paws_val)
        print(f"Added {len(paws_train)} PAWS training examples")
    
    if use_mrpc:
        mrpc_train, mrpc_val = load_mrpc_for_cross_encoder()
        train_examples.extend(mrpc_train)
        val_examples.extend(mrpc_val)
        print(f"Added {len(mrpc_train)} MRPC training examples")
    
    if use_sts:
        sts_train, sts_val = load_sts_for_cross_encoder()
        train_examples.extend(sts_train)
        val_examples.extend(sts_val)
        print(f"Added {len(sts_train)} STS training examples")
    
    print(f"\nTotal training examples: {len(train_examples)}")
    print(f"Total validation examples: {len(val_examples)}")
    
    # Create DataLoader
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=batch_size
    )
    
    # Create evaluator
    evaluator = CEBinaryClassificationEvaluator.from_input_examples(
        val_examples,
        name='validation'
    )
    
    # Training configuration
    print(f"\nTraining configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Output path: {output_path}")
    
    # Train
    print("\nStarting training...")
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
    parser.add_argument("--output_path", type=str, default="../checkpoints/cross_encoder",
                        help="Path to save trained model")
    parser.add_argument("--base_model", type=str, default="cross-encoder/nli-deberta-v3-large",
                        help="Base cross-encoder model")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Training batch size")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--warmup_steps", type=int, default=500,
                        help="Number of warmup steps")
    parser.add_argument("--no_paws", action="store_true",
                        help="Exclude PAWS")
    parser.add_argument("--no_mrpc", action="store_true",
                        help="Exclude MRPC")
    parser.add_argument("--no_sts", action="store_true",
                        help="Exclude STS")
    
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
