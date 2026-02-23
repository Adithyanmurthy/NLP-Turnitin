"""
Training script for Cross-Encoder (RoBERTa-large based)
Fine-tunes on PAWS + MRPC + STS for pairwise similarity verification.
Uses manual model loading with use_fast=False to avoid tokenizer parsing bugs
on older tokenizers library versions (Python 3.8 / tokenizers 0.15.x).
"""

import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from datasets import load_dataset
from tqdm import tqdm


class PairDataset(Dataset):
    """Simple dataset for sentence pairs."""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return item


def load_all_data(use_paws=True, use_mrpc=True, use_sts=True):
    """Load and combine all datasets, return (train_pairs, val_pairs)."""
    train_pairs, val_pairs = [], []

    if use_paws:
        print("Loading PAWS...")
        ds = load_dataset("paws", "labeled_final")
        for item in ds['train']:
            train_pairs.append((item['sentence1'], item['sentence2'], float(item['label'])))
        for item in ds['validation']:
            val_pairs.append((item['sentence1'], item['sentence2'], float(item['label'])))
        print(f"  PAWS: {len(ds['train'])} train, {len(ds['validation'])} val")

    if use_mrpc:
        print("Loading MRPC...")
        ds = load_dataset("glue", "mrpc")
        for item in ds['train']:
            train_pairs.append((item['sentence1'], item['sentence2'], float(item['label'])))
        for item in ds['validation']:
            val_pairs.append((item['sentence1'], item['sentence2'], float(item['label'])))
        print(f"  MRPC: {len(ds['train'])} train, {len(ds['validation'])} val")

    if use_sts:
        print("Loading STS Benchmark...")
        ds = load_dataset("mteb/stsbenchmark-sts")
        for item in ds['train']:
            label = 1.0 if item['score'] >= 3.0 else 0.0
            train_pairs.append((item['sentence1'], item['sentence2'], label))
        for item in ds['validation']:
            label = 1.0 if item['score'] >= 3.0 else 0.0
            val_pairs.append((item['sentence1'], item['sentence2'], label))
        print(f"  STS: {len(ds['train'])} train, {len(ds['validation'])} val")

    print(f"\nTotal: {len(train_pairs)} train, {len(val_pairs)} val")
    return train_pairs, val_pairs


def encode_pairs(tokenizer, pairs, max_length=512):
    """Tokenize sentence pairs."""
    texts_a = [p[0] for p in pairs]
    texts_b = [p[1] for p in pairs]
    labels = [p[2] for p in pairs]
    encodings = tokenizer(texts_a, texts_b, truncation=True, padding=True,
                          max_length=max_length, return_tensors='pt')
    return encodings, labels


def evaluate(model, dataloader, device):
    """Evaluate model on validation set."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop('labels')
            outputs = model(**batch)
            # 2-class output — use softmax on class 1 (similar)
            probs = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    pred_binary = (all_preds >= 0.5).astype(int)
    label_binary = all_labels.astype(int)

    acc = accuracy_score(label_binary, pred_binary)
    f1 = f1_score(label_binary, pred_binary, zero_division=0)
    try:
        auc = roc_auc_score(label_binary, all_preds)
    except ValueError:
        auc = 0.0
    return acc, f1, auc


def train_cross_encoder(
    output_path=None,
    base_model="roberta-large",
    batch_size=8,
    epochs=3,
    warmup_steps=500,
    use_paws=True,
    use_mrpc=True,
    use_sts=True
):
    """
    Train Cross-Encoder using manual training loop.
    Uses roberta-large directly with use_fast=False tokenizer.
    num_labels=2 with CrossEntropyLoss for proper binary classification.
    """
    # Default output path: person_2/checkpoints/cross_encoder
    if output_path is None:
        output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                   "checkpoints", "cross_encoder")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load tokenizer with use_fast=False to avoid tokenizers library bugs
    print(f"Loading model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=2, ignore_mismatched_sizes=True
    )
    model.to(device)

    # Load data
    train_pairs, val_pairs = load_all_data(use_paws, use_mrpc, use_sts)

    # Encode
    print("Tokenizing...")
    train_enc, train_labels = encode_pairs(tokenizer, train_pairs)
    val_enc, val_labels = encode_pairs(tokenizer, val_pairs)

    train_dataset = PairDataset(train_enc, train_labels)
    val_dataset = PairDataset(val_enc, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Differential learning rate: higher for classifier head, lower for backbone
    classifier_params = [p for n, p in model.named_parameters() if 'classifier' in n]
    backbone_params = [p for n, p in model.named_parameters() if 'classifier' not in n]
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': 2e-5},
        {'params': classifier_params, 'lr': 1e-3},  # 50x higher for new head
    ], weight_decay=0.01)

    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # FP16
    use_fp16 = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    best_f1 = 0.0
    os.makedirs(output_path, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop('labels')
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_fp16):
                outputs = model(**batch)
                loss = outputs.loss if outputs.loss is not None else \
                    torch.nn.CrossEntropyLoss()(outputs.logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{total_loss / (pbar.n + 1):.4f}"})

        # Evaluate
        acc, f1, auc = evaluate(model, val_loader, device)
        print(f"  Epoch {epoch+1}: acc={acc:.4f}, f1={f1:.4f}, auc={auc:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            model.save_pretrained(output_path)
            tokenizer.save_pretrained(output_path)
            print(f"  -> Saved best model (F1={best_f1:.4f})")

    print(f"\nTraining complete! Best F1={best_f1:.4f}, saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--base_model", type=str, default="roberta-large")
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
