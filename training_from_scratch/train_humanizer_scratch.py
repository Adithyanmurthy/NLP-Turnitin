#!/usr/bin/env python3
"""
Training From Scratch — Humanizer (Encoder-Decoder Transformer)
Full seq2seq model trained from zero for AI text humanization.

Phase 1: Pre-train as denoising autoencoder (corrupt text → reconstruct)
Phase 2: Fine-tune on paraphrase pairs (AI text → human-like text)

Single GPU:  python train_humanizer_scratch.py
Multi-GPU:   torchrun --nproc_per_node=8 train_humanizer_scratch.py
"""

import sys
import os
import random
sys.path.insert(0, os.path.dirname(__file__))

import torch
from torch.utils.data import Dataset
from config_scratch import HUMANIZER_CONFIG, HUMANIZER_DATASETS, SEED
from models import HumanizerFromScratch
from data_utils import (
    load_tokenizer, collect_all_texts, load_dataset_splits, dataset_available,
    extract_texts_and_labels, Seq2SeqDataset, create_dataloader,
)
from train_engine import Trainer, setup_distributed, cleanup_distributed, is_main_process

torch.manual_seed(SEED)
random.seed(SEED)


class DenoisingDataset(Dataset):
    """
    Denoising autoencoder dataset for pre-training.
    Corrupts input text (mask spans) and trains the model to reconstruct.
    """

    def __init__(self, texts, tokenizer, max_length=512,
                 noise_density=0.15, mean_span_length=3.0):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.noise_density = noise_density
        self.mean_span_length = mean_span_length
        self.pad_id = tokenizer.token_to_id("[PAD]")
        self.mask_id = tokenizer.token_to_id("[MASK]")
        self.bos_id = tokenizer.token_to_id("[BOS]")
        self.eos_id = tokenizer.token_to_id("[EOS]")

    def __len__(self):
        return len(self.texts)

    def _corrupt(self, token_ids):
        length = len(token_ids)
        num_to_mask = max(1, int(length * self.noise_density))
        corrupted = list(token_ids)
        masked_positions = set()
        i = 0
        while len(masked_positions) < num_to_mask and i < length:
            if random.random() < self.noise_density:
                span_len = max(1, int(random.expovariate(1.0 / self.mean_span_length)))
                span_len = min(span_len, length - i, num_to_mask - len(masked_positions))
                for j in range(i, min(i + span_len, length)):
                    masked_positions.add(j)
                    corrupted[j] = self.mask_id
                i += span_len
            else:
                i += 1
        return corrupted, list(token_ids)

    def _pad(self, ids, max_len):
        ids = ids[:max_len]
        mask = [1] * len(ids)
        pad_len = max_len - len(ids)
        return ids + [self.pad_id] * pad_len, mask + [0] * pad_len

    def __getitem__(self, idx):
        encoding = self.tokenizer.encode(self.texts[idx])
        token_ids = encoding.ids[:self.max_length - 2]
        corrupted, original = self._corrupt(token_ids)
        src_ids, src_mask = self._pad(corrupted, self.max_length)
        tgt_ids, tgt_mask = self._pad([self.bos_id] + original, self.max_length)
        label_ids, _ = self._pad(original + [self.eos_id], self.max_length)
        return {
            "src_ids": torch.tensor(src_ids, dtype=torch.long),
            "src_mask": torch.tensor(src_mask, dtype=torch.long),
            "tgt_ids": torch.tensor(tgt_ids, dtype=torch.long),
            "tgt_mask": torch.tensor(tgt_mask, dtype=torch.long),
            "labels": torch.tensor(label_ids, dtype=torch.long),
        }


def pretrain_denoising(rank, world_size, device):
    """Phase 1: Pre-train as denoising autoencoder."""
    if is_main_process(rank):
        print("=" * 70)
        print("  PHASE 1: Denoising Pre-training — Humanizer Encoder-Decoder")
        print("=" * 70)

    config = HUMANIZER_CONFIG
    tokenizer = load_tokenizer()

    all_texts = collect_all_texts(HUMANIZER_DATASETS, split="train")
    if is_main_process(rank):
        print(f"  Collected {len(all_texts):,} text samples")

    if not all_texts:
        if is_main_process(rank):
            print("  [ERROR] No data found. Run setup_all.py first.")
        return None

    val_size = min(len(all_texts) // 10, 50000)
    val_texts = all_texts[:val_size]
    train_texts = all_texts[val_size:]
    if is_main_process(rank):
        print(f"  Train: {len(train_texts):,} | Val: {len(val_texts):,}")

    train_ds = DenoisingDataset(train_texts, tokenizer, config["max_position_embeddings"],
                                config["noise_density"], config["mean_noise_span_length"])
    val_ds = DenoisingDataset(val_texts, tokenizer, config["max_position_embeddings"],
                              config["noise_density"], config["mean_noise_span_length"])

    distributed = world_size > 1
    train_loader = create_dataloader(train_ds, config["pretrain_batch_size"],
                                     distributed=distributed, rank=rank, world_size=world_size)
    val_loader = create_dataloader(val_ds, config["pretrain_batch_size"], shuffle=False,
                                   distributed=distributed, rank=rank, world_size=world_size)

    model = HumanizerFromScratch(config)
    trainer = Trainer(model, "humanizer_scratch_pretrain", config, device, rank, world_size)
    trainer.load_checkpoint()

    def seq2seq_forward(model, batch):
        outputs = model(
            src_ids=batch["src_ids"], src_mask=batch["src_mask"],
            tgt_ids=batch["tgt_ids"], tgt_mask=batch["tgt_mask"],
            labels=batch["labels"],
        )
        return outputs["loss"]

    trainer.train_loop(
        train_loader, val_loader,
        epochs=config["pretrain_epochs"],
        lr=config["pretrain_lr"],
        grad_accum_steps=config["gradient_accumulation_steps"],
        max_grad_norm=config["max_grad_norm"],
        warmup_ratio=config["warmup_ratio"],
        forward_fn=seq2seq_forward,
        max_steps=config["pretrain_max_steps"],
    )
    return trainer._raw_model


def finetune_paraphrase(rank, world_size, device, pretrained_model=None):
    """Phase 2: Fine-tune on paraphrase pairs for humanization."""
    if is_main_process(rank):
        print("\n" + "=" * 70)
        print("  PHASE 2: Paraphrase Fine-tuning — Humanizer")
        print("=" * 70)

    config = HUMANIZER_CONFIG
    tokenizer = load_tokenizer()

    if pretrained_model is not None:
        model = pretrained_model
    else:
        model = HumanizerFromScratch(config)
        temp_trainer = Trainer(model, "humanizer_scratch_pretrain", config, device, rank, world_size)
        if not temp_trainer.load_checkpoint("best"):
            if is_main_process(rank):
                print("  [WARN] No pretrained checkpoint. Training from random init.")

    if is_main_process(rank):
        print("\n  Loading paraphrase datasets...")
    all_train_src, all_train_tgt = [], []
    all_val_src, all_val_tgt = [], []

    for ds_name in HUMANIZER_DATASETS:
        if not dataset_available(ds_name):
            if is_main_process(rank):
                print(f"    {ds_name}: not available, skipping")
            continue
        splits = load_dataset_splits(ds_name)
        for split_name, src_list, tgt_list in [
            ("train", all_train_src, all_train_tgt),
            ("val", all_val_src, all_val_tgt),
        ]:
            data = splits.get(split_name, [])
            if data:
                result = extract_texts_and_labels(data, "paraphrase")
                if len(result) == 3:
                    texts_a, texts_b, _ = result
                    src_list.extend(texts_a)
                    tgt_list.extend(texts_b)
                    if is_main_process(rank):
                        print(f"    {ds_name}/{split_name}: {len(texts_a):,} pairs")

    if is_main_process(rank):
        print(f"\n  Total — Train: {len(all_train_src):,} | Val: {len(all_val_src):,}")

    if not all_train_src:
        if is_main_process(rank):
            print("  [ERROR] No paraphrase data found.")
        return None

    train_ds = Seq2SeqDataset(all_train_src, all_train_tgt, tokenizer,
                              config["max_position_embeddings"])
    val_ds = Seq2SeqDataset(all_val_src, all_val_tgt, tokenizer,
                            config["max_position_embeddings"])

    distributed = world_size > 1
    train_loader = create_dataloader(train_ds, config["finetune_batch_size"],
                                     distributed=distributed, rank=rank, world_size=world_size)
    val_loader = create_dataloader(val_ds, config["finetune_batch_size"], shuffle=False,
                                   distributed=distributed, rank=rank, world_size=world_size)

    def seq2seq_forward(model, batch):
        outputs = model(
            src_ids=batch["src_ids"], src_mask=batch["src_mask"],
            tgt_ids=batch["tgt_ids"], tgt_mask=batch["tgt_mask"],
            labels=batch["labels"],
        )
        return outputs["loss"]

    trainer = Trainer(model, "humanizer_scratch_finetune", config, device, rank, world_size)
    trainer.train_loop(
        train_loader, val_loader,
        epochs=config["finetune_epochs"],
        lr=config["finetune_lr"],
        grad_accum_steps=config["gradient_accumulation_steps"],
        max_grad_norm=config["max_grad_norm"],
        warmup_ratio=config["warmup_ratio"],
        forward_fn=seq2seq_forward,
    )
    return trainer._raw_model


def main():
    rank, world_size, device = setup_distributed()

    if is_main_process(rank):
        print("\n" + "═" * 70)
        print("  HUMANIZER — Training From Scratch")
        print("═" * 70)

    model = pretrain_denoising(rank, world_size, device)
    if model is not None:
        finetune_paraphrase(rank, world_size, device, model)

    if is_main_process(rank):
        print("\n  Humanizer training complete!")

    cleanup_distributed()


if __name__ == "__main__":
    main()
