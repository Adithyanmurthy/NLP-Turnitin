"""
Training From Scratch — Master Configuration
All hyperparameters, paths, dataset mappings, and model architectures.

Nothing pretrained. Everything learned from our datasets.
Multi-GPU ready via PyTorch DistributedDataParallel.
"""

import os
from pathlib import Path

# ─── Paths ───────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
PROJECT_DIR = BASE_DIR.parent

# Data comes from person_1's preprocessed splits (shared across pipelines)
DATA_DIR = PROJECT_DIR / "person_1" / "data"
RAW_DIR = DATA_DIR / "raw"
SPLITS_DIR = DATA_DIR / "splits"

# From-scratch outputs stay in our own folder
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
LOG_DIR = BASE_DIR / "logs"
VOCAB_DIR = BASE_DIR / "vocab"

for d in [CHECKPOINT_DIR, LOG_DIR, VOCAB_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Global Training Settings ────────────────────────────
SEED = 42
MAX_SEQ_LENGTH = 512

# Mixed precision: bf16 on A100/H100, fp16 on older GPUs, auto-detected at runtime
# Set to "auto" to let the engine pick based on GPU capability
DTYPE = "auto"

# torch.compile for 10-20% speedup (PyTorch 2.0+, disable if issues)
COMPILE_MODEL = False  # safer default; enable on cluster if stable

# DataLoader workers: 0 for Windows compatibility, increase on Linux cluster
NUM_WORKERS = int(os.environ.get("SCRATCH_NUM_WORKERS", "0"))

# ─── Tokenizer (trained from scratch on our data) ────────
TOKENIZER_CONFIG = {
    "vocab_size": 32_000,
    "min_frequency": 2,
    "special_tokens": ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[BOS]", "[EOS]"],
    "model_type": "BPE",
    "max_length": MAX_SEQ_LENGTH,
}

# Special token IDs (assigned by order in special_tokens list above)
PAD_ID = 0
UNK_ID = 1
CLS_ID = 2
SEP_ID = 3
MASK_ID = 4
BOS_ID = 5
EOS_ID = 6

# ─── Dataset → Task Mapping ─────────────────────────────
# Only datasets that are auto-downloadable or already preprocessed.
# Manual-only datasets (m4, clough_stevenson, webis_crowd_paraphrase,
# pan_author_id, pan_plagiarism) are included IF their splits exist,
# but won't cause errors if missing.

AI_DETECTOR_DATASETS = [
    "raid",           # RAID benchmark — primary AI detection
    "hc3",            # Human vs ChatGPT
    "gpt2_output",    # GPT-wiki-intro (human vs GPT)
    "faidset",        # Fine-grained AI detection
    # Optional (manual download required):
    "m4",             # Multi-generator multilingual
    "pan_author_id",  # Authorship / stylometry
]

PLAGIARISM_DATASETS = [
    "sts_benchmark",           # STS similarity scores
    "paws",                    # Adversarial paraphrase pairs
    "qqp",                     # Quora question pairs
    "mrpc",                    # Microsoft paraphrase corpus
    "wikisplit",               # WikiSplit sentence pairs
    # Optional (manual download required):
    "pan_plagiarism",          # PAN plagiarism corpora
    "clough_stevenson",        # Clough & Stevenson
    "webis_crowd_paraphrase",  # Webis CPC-11
]

HUMANIZER_DATASETS = [
    "paranmt",         # ChatGPT paraphrases (ParaNMT replacement)
    "paws",            # Adversarial paraphrases
    "qqp",             # Quora question pairs
    "mrpc",            # Microsoft paraphrase corpus
    "bea_2019_gec",    # Grammatical error correction
    "hc3",             # AI→Human pairs for style transfer
]

# All unique datasets (for tokenizer training — union of all three)
ALL_DATASETS = sorted(set(
    AI_DETECTOR_DATASETS + PLAGIARISM_DATASETS + HUMANIZER_DATASETS
))


# ─── Model 1: AI Detector (Transformer Encoder) ─────────
AI_DETECTOR_CONFIG = {
    # Architecture (BERT-base scale, trained from zero)
    "vocab_size": 32_000,
    "hidden_size": 768,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "intermediate_size": 3072,       # FFN inner dim = 4 * hidden
    "hidden_dropout_prob": 0.1,
    "attention_probs_dropout_prob": 0.1,
    "max_position_embeddings": MAX_SEQ_LENGTH,
    "num_labels": 2,                 # binary: human vs AI
    "layer_norm_eps": 1e-12,
    "initializer_range": 0.02,

    # Pre-training (MLM on all text data)
    "pretrain_epochs": 10,
    "pretrain_lr": 1e-4,
    "pretrain_batch_size": 64,
    "pretrain_max_steps": 500_000,
    "mlm_probability": 0.15,

    # Fine-tuning (classification)
    "finetune_epochs": 5,
    "finetune_lr": 3e-5,
    "finetune_batch_size": 32,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "gradient_accumulation_steps": 4,
}

# ─── Model 2: Plagiarism Detector (Siamese Encoder) ─────
PLAGIARISM_DETECTOR_CONFIG = {
    # Architecture (shared-weight dual encoder)
    "vocab_size": 32_000,
    "hidden_size": 768,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "intermediate_size": 3072,
    "hidden_dropout_prob": 0.1,
    "attention_probs_dropout_prob": 0.1,
    "max_position_embeddings": MAX_SEQ_LENGTH,
    "pooling_mode": "mean",
    "similarity_function": "cosine",
    "layer_norm_eps": 1e-12,
    "initializer_range": 0.02,

    # Pre-training (MLM)
    "pretrain_epochs": 10,
    "pretrain_lr": 1e-4,
    "pretrain_batch_size": 64,
    "pretrain_max_steps": 300_000,
    "mlm_probability": 0.15,

    # Fine-tuning (contrastive + similarity regression)
    "finetune_epochs": 5,
    "finetune_lr": 3e-5,
    "finetune_batch_size": 32,
    "contrastive_temperature": 0.05,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "gradient_accumulation_steps": 4,
}

# ─── Model 3: Humanizer (Encoder-Decoder Transformer) ───
HUMANIZER_CONFIG = {
    # Architecture (seq2seq, T5-base scale)
    "vocab_size": 32_000,
    "d_model": 512,
    "encoder_layers": 6,
    "decoder_layers": 6,
    "num_attention_heads": 8,
    "d_ff": 2048,
    "dropout": 0.1,
    "max_position_embeddings": MAX_SEQ_LENGTH,
    "layer_norm_eps": 1e-6,
    "initializer_range": 0.02,

    # Pre-training (denoising autoencoder)
    "pretrain_epochs": 10,
    "pretrain_lr": 1e-4,
    "pretrain_batch_size": 32,
    "pretrain_max_steps": 500_000,
    "noise_density": 0.15,
    "mean_noise_span_length": 3.0,

    # Fine-tuning (paraphrase generation)
    "finetune_epochs": 5,
    "finetune_lr": 3e-5,
    "finetune_batch_size": 16,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "gradient_accumulation_steps": 8,
    "label_smoothing": 0.1,
    "beam_size": 4,
    "length_penalty": 1.0,
}
