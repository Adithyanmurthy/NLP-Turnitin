"""
Person 3 - Humanization Module Configuration
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
LOGS_DIR = BASE_DIR / "logs"

# Create directories
for dir_path in [DATA_DIR, MODELS_DIR, CHECKPOINTS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Dataset configurations
DATASETS = {
    "paranmt": {
        "name": "chatgpt-paraphrases",
        "url": "https://huggingface.co/datasets/humarin/chatgpt-paraphrases",
        "hf_path": "humarin/chatgpt-paraphrases",
        "sample_size": 500000,
        "format": "paraphrase"
    },
    "paws": {
        "name": "paws",
        "hf_path": "paws",
        "subset": "labeled_final",
        "format": "paraphrase"
    },
    "qqp": {
        "name": "qqp",
        "hf_path": "glue",
        "subset": "qqp",
        "format": "paraphrase"
    },
    "mrpc": {
        "name": "mrpc",
        "hf_path": "glue",
        "subset": "mrpc",
        "format": "paraphrase"
    },
    "bea_gec": {
        "name": "bea-2019-gec",
        "hf_path": "wi_locness",
        "format": "error_correction"
    },
    "sts": {
        "name": "sts-benchmark",
        "hf_path": "mteb/stsbenchmark-sts",
        "format": "similarity"
    },
    "hc3": {
        "name": "hc3",
        "hf_path": "Hello-SimpleAI/HC3",
        "format": "ai_human_pairs"
    }
}

# Model configurations
MODELS = {
    "dipper": {
        "name": "DIPPER",
        "hf_path": "kalpeshk2011/dipper-paraphraser-xxl",
        "type": "paraphraser",
        "params": "11B",
        "use_pretrained": True,  # Use pretrained, fine-tune lightly
        "batch_size": 1,
        "max_length": 512
    },
    "flan_t5": {
        "name": "Flan-T5-XL",
        "hf_path": "google/flan-t5-xl",
        "type": "seq2seq",
        "params": "3B",
        "batch_size": 1,
        "max_length": 512,
        "learning_rate": 1e-4,
        "epochs": 3
    },
    "pegasus": {
        "name": "PEGASUS-large",
        "hf_path": "google/pegasus-large",
        "type": "seq2seq",
        "params": "568M",
        "batch_size": 4,
        "max_length": 512,
        "learning_rate": 1e-4,
        "epochs": 3
    },
    "mistral": {
        "name": "Mistral-7B",
        "hf_path": "mistralai/Mistral-7B-v0.3",
        "type": "causal_lm",
        "params": "7B",
        "use_qlora": True,
        "batch_size": 2,
        "max_length": 512,
        "learning_rate": 2e-4,
        "epochs": 2,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05
    }
}

# Training configurations
TRAINING_CONFIG = {
    "train_split": 0.8,
    "val_split": 0.1,
    "test_split": 0.1,
    "seed": 42,
    "gradient_accumulation_steps": 4,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "logging_steps": 100,
    "save_steps": 1000,
    "eval_steps": 500,
    "fp16": True,
    "dataloader_num_workers": 0  # 0 for Windows compatibility
}

# Feedback loop configuration
FEEDBACK_CONFIG = {
    "initial_diversity": 60,
    "initial_reorder": 40,
    "diversity_increment": 10,
    "reorder_increment": 10,
    "max_diversity": 100,
    "max_reorder": 100,
    "target_ai_score": 0.05,  # Target AI detection score (≤5% = undetectable)
    "max_iterations": 10,     # More iterations for aggressive humanization
}

# Integration with Person 1 (AI Detector)
PERSON1_CONFIG = {
    "detector_path": str(BASE_DIR.parent / "person_1"),
    "checkpoint_path": str(BASE_DIR.parent / "person_1" / "checkpoints"),
    "available": False  # Will be set to True when Person 1's module is available
}

# API interface for Person 4
API_CONFIG = {
    "function_name": "humanize",
    "input_format": "str",
    "output_format": {
        "text": "str",
        "ai_score_before": "float",
        "ai_score_after": "float",
        "iterations": "int",
        "diversity_used": "int",
        "reorder_used": "int"
    }
}

# Evaluation metrics
EVALUATION_METRICS = [
    "bleu",
    "rouge",
    "meteor",
    "bertscore",
    "semantic_similarity",
    "ai_detection_score"
]
