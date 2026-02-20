"""
Training From Scratch — Data Loading Utilities
Loads preprocessed datasets and prepares them for each training phase.
Gracefully skips datasets that aren't available (manual download required).
"""

import json
import random
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from pathlib import Path

from config_scratch import SPLITS_DIR, RAW_DIR, VOCAB_DIR, SEED, NUM_WORKERS

random.seed(SEED)


# ═══════════════════════════════════════════════════════════
#  TOKENIZER
# ═══════════════════════════════════════════════════════════

def load_tokenizer():
    """Load the custom-trained BPE tokenizer."""
    from tokenizers import Tokenizer
    tok_path = VOCAB_DIR / "tokenizer.json"
    if not tok_path.exists():
        raise FileNotFoundError(
            f"Tokenizer not found at {tok_path}. Run: python run_all_scratch.py --only tok"
        )
    return Tokenizer.from_file(str(tok_path))


# ═══════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════

def load_jsonl(path: Path, max_samples: int = None):
    """Load a JSONL file, return list of dicts."""
    data = []
    if not path.exists():
        return data
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return data


def load_dataset_splits(dataset_name: str, max_per_split: int = None):
    """Load train/val/test splits for a dataset. Returns empty splits if missing."""
    splits = {}
    for split in ["train", "val", "test"]:
        path = SPLITS_DIR / dataset_name / f"{split}.jsonl"
        splits[split] = load_jsonl(path, max_per_split)
    return splits


def dataset_available(dataset_name: str) -> bool:
    """Check if a dataset has preprocessed splits available."""
    train_path = SPLITS_DIR / dataset_name / "train.jsonl"
    return train_path.exists() and train_path.stat().st_size > 0


def extract_texts_and_labels(data: list, dataset_type: str):
    """
    Extract data from dataset records based on type.
    classification → (texts, labels)
    paraphrase/similarity → (text_a, text_b, scores)
    """
    if dataset_type == "classification":
        texts, labels = [], []
        for item in data:
            text = item.get("text", item.get("content", item.get("document", "")))
            label = item.get("label", 0)
            if text:
                texts.append(text)
                labels.append(int(label))
        return texts, labels

    elif dataset_type in ("paraphrase", "similarity"):
        texts_a, texts_b, scores = [], [], []
        for item in data:
            a = item.get("input", item.get("text_a", item.get("sentence1",
                item.get("question1", ""))))
            b = item.get("output", item.get("text_b", item.get("sentence2",
                item.get("question2", ""))))
            s = item.get("score", item.get("label", 1.0))
            if a and b:
                texts_a.append(a)
                texts_b.append(b)
                scores.append(float(s))
        return texts_a, texts_b, scores

    return [], []


# ═══════════════════════════════════════════════════════════
#  DATASET CLASSES
# ═══════════════════════════════════════════════════════════

class MLMDataset(Dataset):
    """Dataset for Masked Language Modeling pre-training."""

    def __init__(self, texts: list, tokenizer, max_length: int = 512,
                 mlm_prob: float = 0.15):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mlm_prob = mlm_prob
        self.vocab_size = tokenizer.get_vocab_size()
        self.mask_id = tokenizer.token_to_id("[MASK]")
        self.pad_id = tokenizer.token_to_id("[PAD]")
        self.special_ids = {
            tokenizer.token_to_id(t)
            for t in ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[BOS]", "[EOS]"]
            if tokenizer.token_to_id(t) is not None
        }

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer.encode(self.texts[idx])
        input_ids = encoding.ids[:self.max_length]
        attention_mask = encoding.attention_mask[:self.max_length]

        pad_len = self.max_length - len(input_ids)
        input_ids = input_ids + [self.pad_id] * pad_len
        attention_mask = attention_mask + [0] * pad_len

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)

        # Create MLM labels
        labels = input_ids.clone()
        prob_matrix = torch.full(labels.shape, self.mlm_prob)
        for sid in self.special_ids:
            prob_matrix[labels == sid] = 0.0
        prob_matrix[attention_mask == 0] = 0.0

        masked_indices = torch.bernoulli(prob_matrix).bool()
        labels[~masked_indices] = -100

        # 80% [MASK], 10% random, 10% keep
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.mask_id
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices & ~indices_replaced
        )
        random_words = torch.randint(self.vocab_size, labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


class ClassificationDataset(Dataset):
    """Dataset for binary classification fine-tuning."""

    def __init__(self, texts: list, labels: list, tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_id = tokenizer.token_to_id("[PAD]")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer.encode(self.texts[idx])
        input_ids = encoding.ids[:self.max_length]
        attention_mask = encoding.attention_mask[:self.max_length]
        pad_len = self.max_length - len(input_ids)
        input_ids = input_ids + [self.pad_id] * pad_len
        attention_mask = attention_mask + [0] * pad_len
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class SentencePairDataset(Dataset):
    """Dataset for sentence-pair similarity/contrastive training."""

    def __init__(self, texts_a: list, texts_b: list, scores: list,
                 tokenizer, max_length: int = 512):
        self.texts_a = texts_a
        self.texts_b = texts_b
        self.scores = scores
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_id = tokenizer.token_to_id("[PAD]")

    def __len__(self):
        return len(self.texts_a)

    def _encode(self, text):
        encoding = self.tokenizer.encode(text)
        ids = encoding.ids[:self.max_length]
        mask = encoding.attention_mask[:self.max_length]
        pad_len = self.max_length - len(ids)
        return ids + [self.pad_id] * pad_len, mask + [0] * pad_len

    def __getitem__(self, idx):
        ids_a, mask_a = self._encode(self.texts_a[idx])
        ids_b, mask_b = self._encode(self.texts_b[idx])
        return {
            "input_ids_a": torch.tensor(ids_a, dtype=torch.long),
            "attention_mask_a": torch.tensor(mask_a, dtype=torch.long),
            "input_ids_b": torch.tensor(ids_b, dtype=torch.long),
            "attention_mask_b": torch.tensor(mask_b, dtype=torch.long),
            "labels": torch.tensor(self.scores[idx], dtype=torch.float),
        }


class Seq2SeqDataset(Dataset):
    """Dataset for encoder-decoder seq2seq training."""

    def __init__(self, sources: list, targets: list, tokenizer, max_length: int = 512):
        self.sources = sources
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_id = tokenizer.token_to_id("[PAD]")
        self.bos_id = tokenizer.token_to_id("[BOS]")
        self.eos_id = tokenizer.token_to_id("[EOS]")

    def __len__(self):
        return len(self.sources)

    def _encode(self, text, add_bos=False, add_eos=False):
        encoding = self.tokenizer.encode(text)
        ids = encoding.ids[:self.max_length - int(add_bos) - int(add_eos)]
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        mask = [1] * len(ids)
        pad_len = self.max_length - len(ids)
        return ids + [self.pad_id] * pad_len, mask + [0] * pad_len

    def __getitem__(self, idx):
        src_ids, src_mask = self._encode(self.sources[idx])
        tgt_ids, tgt_mask = self._encode(self.targets[idx], add_bos=True)
        label_ids, _ = self._encode(self.targets[idx], add_eos=True)
        return {
            "src_ids": torch.tensor(src_ids, dtype=torch.long),
            "src_mask": torch.tensor(src_mask, dtype=torch.long),
            "tgt_ids": torch.tensor(tgt_ids, dtype=torch.long),
            "tgt_mask": torch.tensor(tgt_mask, dtype=torch.long),
            "labels": torch.tensor(label_ids, dtype=torch.long),
        }


# ═══════════════════════════════════════════════════════════
#  DATA LOADING HELPERS
# ═══════════════════════════════════════════════════════════

def collect_all_texts(dataset_names: list, split: str = "train",
                      max_per_dataset: int = None) -> list:
    """Collect all raw texts from multiple datasets for MLM pre-training.
    Silently skips datasets that aren't available."""
    all_texts = []
    for name in dataset_names:
        if not dataset_available(name):
            continue
        splits = load_dataset_splits(name, max_per_dataset)
        data = splits.get(split, [])
        for item in data:
            for key in ["text", "input", "output", "sentence", "sentence1",
                        "sentence2", "text_a", "text_b", "content", "document",
                        "question1", "question2"]:
                if key in item and isinstance(item[key], str) and len(item[key]) > 20:
                    all_texts.append(item[key])
    random.shuffle(all_texts)
    return all_texts


def create_dataloader(dataset: Dataset, batch_size: int, shuffle: bool = True,
                      distributed: bool = False, rank: int = 0,
                      world_size: int = 1) -> DataLoader:
    """Create a DataLoader. Uses DistributedSampler when distributed=True."""
    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                     shuffle=shuffle)
        shuffle = False  # sampler handles shuffling

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=distributed or shuffle,
    )
