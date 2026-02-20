from __future__ import annotations
"""
Person 1 — Shared Data Loading Utility
Used by Person 1, 2, 3, and 4 to load preprocessed datasets.

Usage:
    from person_1.data_loader import DataLoader

    loader = DataLoader()
    train_data = loader.load("raid", split="train")
    all_ai_data = loader.load_combined(["raid", "hc3", "m4"], split="train")
"""

import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader as TorchDataLoader
from transformers import AutoTokenizer

from config import SPLITS_DIR, DATASETS


class TextClassificationDataset(TorchDataset):
    """PyTorch dataset for classification tasks (text → label)."""

    def __init__(self, records: list[dict], tokenizer, max_length: int = 512):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        encoding = self.tokenizer(
            record["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(record["label"], dtype=torch.long),
        }


class ParaphraseDataset(TorchDataset):
    """PyTorch dataset for paraphrase tasks (input → output)."""

    def __init__(self, records: list[dict], tokenizer, max_length: int = 512):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        input_enc = self.tokenizer(
            record["input"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        output_enc = self.tokenizer(
            record["output"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": input_enc["input_ids"].squeeze(0),
            "attention_mask": input_enc["attention_mask"].squeeze(0),
            "labels": output_enc["input_ids"].squeeze(0),
        }


class SimilarityDataset(TorchDataset):
    """PyTorch dataset for similarity tasks (text_a, text_b → score)."""

    def __init__(self, records: list[dict], tokenizer, max_length: int = 512):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        encoding = self.tokenizer(
            record["text_a"],
            record["text_b"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(record["score"], dtype=torch.float),
        }


class DataLoader:
    """
    Unified data loader for all preprocessed datasets.
    Loads JSONL files from the splits directory.
    """

    def __init__(self, splits_dir: Optional[Path] = None):
        self.splits_dir = splits_dir or SPLITS_DIR

    def load(self, dataset_name: str, split: str = "train") -> list[dict]:
        """Load a single dataset split as a list of dicts."""
        path = self.splits_dir / dataset_name / f"{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset split not found: {path}\n"
                f"Run 'python scripts/download_datasets.py' and 'python scripts/preprocess.py' first."
            )
        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def load_combined(
        self,
        dataset_names: list[str],
        split: str = "train",
        max_per_dataset: Optional[int] = None,
    ) -> list[dict]:
        """Load and combine multiple datasets into one list."""
        combined = []
        for name in dataset_names:
            try:
                records = self.load(name, split)
                if max_per_dataset and len(records) > max_per_dataset:
                    import random
                    random.seed(42)
                    records = random.sample(records, max_per_dataset)
                combined.extend(records)
                print(f"  Loaded {name}/{split}: {len(records):,} records")
            except FileNotFoundError:
                print(f"  [SKIP] {name}/{split} not found")
        return combined

    def get_torch_dataset(
        self,
        dataset_name: str,
        split: str,
        tokenizer_name: str,
        max_length: int = 512,
        data_type: Optional[str] = None,
    ) -> TorchDataset:
        """Get a PyTorch-ready dataset for training."""
        records = self.load(dataset_name, split)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        if data_type is None:
            data_type = DATASETS.get(dataset_name, {}).get("type", "classification")

        if data_type == "classification":
            return TextClassificationDataset(records, tokenizer, max_length)
        elif data_type == "paraphrase":
            return ParaphraseDataset(records, tokenizer, max_length)
        elif data_type == "similarity":
            return SimilarityDataset(records, tokenizer, max_length)
        else:
            raise ValueError(f"Unknown data type: {data_type}")

    def get_torch_dataloader(
        self,
        dataset_name: str,
        split: str,
        tokenizer_name: str,
        batch_size: int = 16,
        max_length: int = 512,
        shuffle: bool = True,
        data_type: Optional[str] = None,
    ) -> TorchDataLoader:
        """Get a PyTorch DataLoader ready for training."""
        dataset = self.get_torch_dataset(
            dataset_name, split, tokenizer_name, max_length, data_type
        )
        return TorchDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,  # 0 for Windows compatibility
            pin_memory=True,
        )

    def get_combined_torch_dataset(
        self,
        dataset_names: list[str],
        split: str,
        tokenizer_name: str,
        max_length: int = 512,
        max_per_dataset: Optional[int] = None,
        data_type: str = "classification",
    ) -> TorchDataset:
        """Get a combined PyTorch dataset from multiple sources."""
        records = self.load_combined(dataset_names, split, max_per_dataset)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        if data_type == "classification":
            return TextClassificationDataset(records, tokenizer, max_length)
        elif data_type == "paraphrase":
            return ParaphraseDataset(records, tokenizer, max_length)
        elif data_type == "similarity":
            return SimilarityDataset(records, tokenizer, max_length)
        else:
            raise ValueError(f"Unknown data type: {data_type}")

    def dataset_info(self, dataset_name: str) -> dict:
        """Get info about a dataset's available splits and sizes."""
        ds_dir = self.splits_dir / dataset_name
        info = {"name": dataset_name, "splits": {}}
        if ds_dir.exists():
            for split_file in ds_dir.glob("*.jsonl"):
                split_name = split_file.stem
                count = sum(1 for _ in open(split_file))
                info["splits"][split_name] = count
        return info

    def list_available(self) -> list[str]:
        """List all available preprocessed datasets."""
        available = []
        for d in self.splits_dir.iterdir():
            if d.is_dir() and any(d.glob("*.jsonl")):
                available.append(d.name)
        return sorted(available)
