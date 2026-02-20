#!/usr/bin/env python3
"""
Training From Scratch — AI Detector
Transformer encoder trained from zero for AI-generated text detection.

Phase 1: Pre-train with MLM on all classification texts
Phase 2: Fine-tune on AI detection datasets (binary: human=0, AI=1)

Single GPU:  python train_ai_detector_scratch.py
Multi-GPU:   torchrun --nproc_per_node=8 train_ai_detector_scratch.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
from config_scratch import AI_DETECTOR_CONFIG, AI_DETECTOR_DATASETS, SEED
from models import AIDetectorFromScratch
from data_utils import (
    load_tokenizer, collect_all_texts, load_dataset_splits, dataset_available,
    extract_texts_and_labels, MLMDataset, ClassificationDataset, create_dataloader,
)
from train_engine import Trainer, setup_distributed, cleanup_distributed, is_main_process

torch.manual_seed(SEED)


def pretrain_mlm(rank, world_size, device):
    """Phase 1: Pre-train the encoder with Masked Language Modeling."""
    if is_main_process(rank):
        print("=" * 70)
        print("  PHASE 1: MLM Pre-training — AI Detector Encoder")
        print("=" * 70)

    config = AI_DETECTOR_CONFIG
    tokenizer = load_tokenizer()

    if is_main_process(rank):
        print("\n  Loading texts for MLM pre-training...")
    all_texts = collect_all_texts(AI_DETECTOR_DATASETS, split="train")
    if is_main_process(rank):
        print(f"  Collected {len(all_texts):,} text samples")

    if not all_texts:
        if is_main_process(rank):
            print("  [ERROR] No training data found. Run setup_all.py first.")
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

    model = AIDetectorFromScratch(config)
    trainer = Trainer(model, "ai_detector_scratch_pretrain", config, device, rank, world_size)
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


def finetune_classification(rank, world_size, device, pretrained_model=None):
    """Phase 2: Fine-tune for AI detection classification."""
    if is_main_process(rank):
        print("\n" + "=" * 70)
        print("  PHASE 2: Classification Fine-tuning — AI Detector")
        print("=" * 70)

    config = AI_DETECTOR_CONFIG
    tokenizer = load_tokenizer()

    if pretrained_model is not None:
        model = pretrained_model
    else:
        model = AIDetectorFromScratch(config)
        temp_trainer = Trainer(model, "ai_detector_scratch_pretrain", config, device, rank, world_size)
        if not temp_trainer.load_checkpoint("best"):
            if is_main_process(rank):
                print("  [WARN] No pretrained checkpoint. Training from random init.")

    if is_main_process(rank):
        print("\n  Loading classification datasets...")
    all_train_texts, all_train_labels = [], []
    all_val_texts, all_val_labels = [], []

    for ds_name in AI_DETECTOR_DATASETS:
        if not dataset_available(ds_name):
            if is_main_process(rank):
                print(f"    {ds_name}: not available, skipping")
            continue
        splits = load_dataset_splits(ds_name)
        for split_name, t_texts, t_labels in [
            ("train", all_train_texts, all_train_labels),
            ("val", all_val_texts, all_val_labels),
        ]:
            data = splits.get(split_name, [])
            if data:
                texts, labels = extract_texts_and_labels(data, "classification")
                t_texts.extend(texts)
                t_labels.extend(labels)
                if is_main_process(rank):
                    print(f"    {ds_name}/{split_name}: {len(texts):,} samples")

    if is_main_process(rank):
        print(f"\n  Total — Train: {len(all_train_texts):,} | Val: {len(all_val_texts):,}")

    if not all_train_texts:
        if is_main_process(rank):
            print("  [ERROR] No classification data found.")
        return None

    train_ds = ClassificationDataset(all_train_texts, all_train_labels, tokenizer,
                                     config["max_position_embeddings"])
    val_ds = ClassificationDataset(all_val_texts, all_val_labels, tokenizer,
                                   config["max_position_embeddings"])

    distributed = world_size > 1
    train_loader = create_dataloader(train_ds, config["finetune_batch_size"],
                                     distributed=distributed, rank=rank, world_size=world_size)
    val_loader = create_dataloader(val_ds, config["finetune_batch_size"], shuffle=False,
                                   distributed=distributed, rank=rank, world_size=world_size)

    trainer = Trainer(model, "ai_detector_scratch_finetune", config, device, rank, world_size)
    trainer.train_loop(
        train_loader, val_loader,
        epochs=config["finetune_epochs"],
        lr=config["finetune_lr"],
        grad_accum_steps=config["gradient_accumulation_steps"],
        max_grad_norm=config["max_grad_norm"],
        warmup_ratio=config["warmup_ratio"],
    )
    return trainer._raw_model


def main():
    rank, world_size, device = setup_distributed()

    if is_main_process(rank):
        print("\n" + "═" * 70)
        print("  AI DETECTOR — Training From Scratch")
        print("═" * 70)

    model = pretrain_mlm(rank, world_size, device)
    if model is not None:
        finetune_classification(rank, world_size, device, model)

    if is_main_process(rank):
        print("\n  AI Detector training complete!")

    cleanup_distributed()


if __name__ == "__main__":
    main()
