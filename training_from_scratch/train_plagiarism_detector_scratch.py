#!/usr/bin/env python3
"""
Training From Scratch — Plagiarism Detector
Siamese transformer encoder trained from zero for plagiarism detection.

Phase 1: Pre-train with MLM on all similarity/paraphrase texts
Phase 2: Fine-tune with contrastive + similarity regression loss

Single GPU:  python train_plagiarism_detector_scratch.py
Multi-GPU:   torchrun --nproc_per_node=8 train_plagiarism_detector_scratch.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
from config_scratch import PLAGIARISM_DETECTOR_CONFIG, PLAGIARISM_DATASETS, SEED
from models import PlagiarismDetectorFromScratch
from data_utils import (
    load_tokenizer, collect_all_texts, load_dataset_splits, dataset_available,
    extract_texts_and_labels, MLMDataset, SentencePairDataset, create_dataloader,
)
from train_engine import Trainer, setup_distributed, cleanup_distributed, is_main_process

torch.manual_seed(SEED)


def pretrain_mlm(rank, world_size, device):
    """Phase 1: Pre-train the shared encoder with MLM."""
    if is_main_process(rank):
        print("=" * 70)
        print("  PHASE 1: MLM Pre-training — Plagiarism Detector Encoder")
        print("=" * 70)

    config = PLAGIARISM_DETECTOR_CONFIG
    tokenizer = load_tokenizer()

    all_texts = collect_all_texts(PLAGIARISM_DATASETS, split="train")
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

    train_ds = MLMDataset(train_texts, tokenizer, config["max_position_embeddings"],
                          config["mlm_probability"])
    val_ds = MLMDataset(val_texts, tokenizer, config["max_position_embeddings"],
                        config["mlm_probability"])

    distributed = world_size > 1
    train_loader = create_dataloader(train_ds, config["pretrain_batch_size"],
                                     distributed=distributed, rank=rank, world_size=world_size)
    val_loader = create_dataloader(val_ds, config["pretrain_batch_size"], shuffle=False,
                                   distributed=distributed, rank=rank, world_size=world_size)

    model = PlagiarismDetectorFromScratch(config)
    trainer = Trainer(model, "plagiarism_detector_scratch_pretrain", config, device, rank, world_size)
    trainer.load_checkpoint()

    def mlm_forward(model, batch):
        outputs = model.forward_mlm(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
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
        forward_fn=mlm_forward,
        max_steps=config["pretrain_max_steps"],
    )
    return trainer._raw_model


def finetune_similarity(rank, world_size, device, pretrained_model=None):
    """Phase 2: Fine-tune with contrastive/similarity loss on sentence pairs."""
    if is_main_process(rank):
        print("\n" + "=" * 70)
        print("  PHASE 2: Similarity Fine-tuning — Plagiarism Detector")
        print("=" * 70)

    config = PLAGIARISM_DETECTOR_CONFIG
    tokenizer = load_tokenizer()

    if pretrained_model is not None:
        model = pretrained_model
    else:
        model = PlagiarismDetectorFromScratch(config)
        temp_trainer = Trainer(model, "plagiarism_detector_scratch_pretrain", config, device, rank, world_size)
        if not temp_trainer.load_checkpoint("best"):
            if is_main_process(rank):
                print("  [WARN] No pretrained checkpoint. Training from random init.")

    if is_main_process(rank):
        print("\n  Loading sentence pair datasets...")
    all_train_a, all_train_b, all_train_scores = [], [], []
    all_val_a, all_val_b, all_val_scores = [], [], []

    for ds_name in PLAGIARISM_DATASETS:
        if not dataset_available(ds_name):
            if is_main_process(rank):
                print(f"    {ds_name}: not available, skipping")
            continue
        splits = load_dataset_splits(ds_name)
        for split_name, ta, tb, ts in [
            ("train", all_train_a, all_train_b, all_train_scores),
            ("val", all_val_a, all_val_b, all_val_scores),
        ]:
            data = splits.get(split_name, [])
            if data:
                result = extract_texts_and_labels(data, "paraphrase")
                if len(result) == 3:
                    texts_a, texts_b, scores = result
                    ta.extend(texts_a)
                    tb.extend(texts_b)
                    ts.extend(scores)
                    if is_main_process(rank):
                        print(f"    {ds_name}/{split_name}: {len(texts_a):,} pairs")

    if is_main_process(rank):
        print(f"\n  Total — Train: {len(all_train_a):,} | Val: {len(all_val_a):,}")

    if not all_train_a:
        if is_main_process(rank):
            print("  [ERROR] No pair data found.")
        return None

    train_ds = SentencePairDataset(all_train_a, all_train_b, all_train_scores,
                                   tokenizer, config["max_position_embeddings"])
    val_ds = SentencePairDataset(all_val_a, all_val_b, all_val_scores,
                                 tokenizer, config["max_position_embeddings"])

    distributed = world_size > 1
    train_loader = create_dataloader(train_ds, config["finetune_batch_size"],
                                     distributed=distributed, rank=rank, world_size=world_size)
    val_loader = create_dataloader(val_ds, config["finetune_batch_size"], shuffle=False,
                                   distributed=distributed, rank=rank, world_size=world_size)

    def pair_forward(model, batch):
        outputs = model(
            input_ids_a=batch["input_ids_a"],
            attention_mask_a=batch["attention_mask_a"],
            input_ids_b=batch["input_ids_b"],
            attention_mask_b=batch["attention_mask_b"],
            labels=batch["labels"],
        )
        return outputs["loss"]

    trainer = Trainer(model, "plagiarism_detector_scratch_finetune", config, device, rank, world_size)
    trainer.train_loop(
        train_loader, val_loader,
        epochs=config["finetune_epochs"],
        lr=config["finetune_lr"],
        grad_accum_steps=config["gradient_accumulation_steps"],
        max_grad_norm=config["max_grad_norm"],
        warmup_ratio=config["warmup_ratio"],
        forward_fn=pair_forward,
    )
    return trainer._raw_model


def main():
    rank, world_size, device = setup_distributed()

    if is_main_process(rank):
        print("\n" + "═" * 70)
        print("  PLAGIARISM DETECTOR — Training From Scratch")
        print("═" * 70)

    model = pretrain_mlm(rank, world_size, device)
    if model is not None:
        finetune_similarity(rank, world_size, device, model)

    if is_main_process(rank):
        print("\n  Plagiarism Detector training complete!")

    cleanup_distributed()


if __name__ == "__main__":
    main()
