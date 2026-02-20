"""
Training script for Sentence-BERT on similarity datasets
Fine-tunes all-mpnet-base-v2 on STS Benchmark + PAWS + QQP
"""

import os
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader
from datasets import load_dataset
import argparse


def load_sts_benchmark():
    """Load STS Benchmark dataset."""
    print("Loading STS Benchmark...")
    dataset = load_dataset("mteb/stsbenchmark-sts")
    
    train_examples = []
    for item in dataset['train']:
        score = float(item['score']) / 5.0  # Normalize to [0, 1]
        train_examples.append(InputExample(
            texts=[item['sentence1'], item['sentence2']],
            label=score
        ))
    
    val_examples = []
    for item in dataset['validation']:
        score = float(item['score']) / 5.0
        val_examples.append(InputExample(
            texts=[item['sentence1'], item['sentence2']],
            label=score
        ))
    
    return train_examples, val_examples


def load_paws():
    """Load PAWS dataset."""
    print("Loading PAWS...")
    dataset = load_dataset("paws", "labeled_final")
    
    train_examples = []
    for item in dataset['train']:
        # PAWS has binary labels (0 or 1)
        label = float(item['label'])
        train_examples.append(InputExample(
            texts=[item['sentence1'], item['sentence2']],
            label=label
        ))
    
    val_examples = []
    for item in dataset['validation']:
        label = float(item['label'])
        val_examples.append(InputExample(
            texts=[item['sentence1'], item['sentence2']],
            label=label
        ))
    
    return train_examples, val_examples


def load_qqp(max_samples=100000):
    """Load QQP dataset (sampled)."""
    print(f"Loading QQP (sampling {max_samples} examples)...")
    dataset = load_dataset("glue", "qqp")
    
    train_examples = []
    for i, item in enumerate(dataset['train']):
        if i >= max_samples:
            break
        if item['question1'] and item['question2']:
            label = float(item['label'])
            train_examples.append(InputExample(
                texts=[item['question1'], item['question2']],
                label=label
            ))
    
    val_examples = []
    for i, item in enumerate(dataset['validation']):
        if i >= max_samples // 10:
            break
        if item['question1'] and item['question2']:
            label = float(item['label'])
            val_examples.append(InputExample(
                texts=[item['question1'], item['question2']],
                label=label
            ))
    
    return train_examples, val_examples


def train_sentence_bert(
    output_path: str,
    base_model: str = "sentence-transformers/all-mpnet-base-v2",
    batch_size: int = 16,
    epochs: int = 3,
    warmup_steps: int = 1000,
    use_sts: bool = True,
    use_paws: bool = True,
    use_qqp: bool = True
):
    """
    Train Sentence-BERT model.
    
    Args:
        output_path: Path to save trained model
        base_model: Base model to fine-tune
        batch_size: Training batch size
        epochs: Number of training epochs
        warmup_steps: Number of warmup steps
        use_sts: Include STS Benchmark
        use_paws: Include PAWS
        use_qqp: Include QQP
    """
    # Load model
    print(f"Loading base model: {base_model}")
    model = SentenceTransformer(base_model)
    
    # Load datasets
    train_examples = []
    val_examples = []
    
    if use_sts:
        sts_train, sts_val = load_sts_benchmark()
        train_examples.extend(sts_train)
        val_examples.extend(sts_val)
        print(f"Added {len(sts_train)} STS training examples")
    
    if use_paws:
        paws_train, paws_val = load_paws()
        train_examples.extend(paws_train)
        val_examples.extend(paws_val)
        print(f"Added {len(paws_train)} PAWS training examples")
    
    if use_qqp:
        qqp_train, qqp_val = load_qqp()
        train_examples.extend(qqp_train)
        val_examples.extend(qqp_val)
        print(f"Added {len(qqp_train)} QQP training examples")
    
    print(f"\nTotal training examples: {len(train_examples)}")
    print(f"Total validation examples: {len(val_examples)}")
    
    # Create DataLoader
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=batch_size
    )
    
    # Define loss function (Cosine Similarity Loss)
    train_loss = losses.CosineSimilarityLoss(model)
    
    # Create evaluator
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
        val_examples,
        name='validation'
    )
    
    # Training configuration
    num_training_steps = len(train_dataloader) * epochs
    
    print(f"\nTraining configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Training steps: {num_training_steps}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Output path: {output_path}")
    
    # Train
    print("\nStarting training...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
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
    parser = argparse.ArgumentParser(description="Train Sentence-BERT for plagiarism detection")
    parser.add_argument("--output_path", type=str, default="../checkpoints/sbert",
                        help="Path to save trained model")
    parser.add_argument("--base_model", type=str, default="sentence-transformers/all-mpnet-base-v2",
                        help="Base model to fine-tune")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Training batch size")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--warmup_steps", type=int, default=1000,
                        help="Number of warmup steps")
    parser.add_argument("--no_sts", action="store_true",
                        help="Exclude STS Benchmark")
    parser.add_argument("--no_paws", action="store_true",
                        help="Exclude PAWS")
    parser.add_argument("--no_qqp", action="store_true",
                        help="Exclude QQP")
    
    args = parser.parse_args()
    
    train_sentence_bert(
        output_path=args.output_path,
        base_model=args.base_model,
        batch_size=args.batch_size,
        epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        use_sts=not args.no_sts,
        use_paws=not args.no_paws,
        use_qqp=not args.no_qqp
    )
