"""
Person 1 — Configuration
All paths, hyperparameters, and constants in one place.
"""

from pathlib import Path

# ─── Paths ───────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"

for d in [RAW_DIR, PROCESSED_DIR, SPLITS_DIR, CHECKPOINT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Dataset identifiers (Hugging Face or local) ────────
DATASETS = {
    # === AI Detection (Person 1 primary) ===
    "raid": {
        "hf_name": "liamdugan/raid",
        "type": "classification",
        "description": "RAID benchmark — 11 LLMs, 11 genres, adversarial attacks",
    },
    "hc3": {
        "hf_name": "Hello-SimpleAI/HC3",
        "type": "classification",
        "description": "Human vs ChatGPT comparison corpus",
        "load_parquet": True,  # loading script no longer supported
    },
    "m4": {
        "hf_name": None,
        "type": "classification",
        "description": "Multi-generator, multi-domain, multi-lingual detection",
        "manual_download": True,
        "manual_note": "M4 dataset repo is no longer accessible on HuggingFace. Download manually from the paper authors or use RAID/HC3 as alternatives.",
    },
    "gpt2_output": {
        "hf_name": "aadityaubhat/GPT-wiki-intro",
        "type": "classification",
        "description": "GPT-wiki-intro — 150K human vs GPT-generated Wikipedia intros",
    },
    "faidset": {
        "hf_name": "ngocminhta/FAIDSet",
        "type": "classification",
        "description": "Fine-grained AI detection (human/AI/mixed) — 84K examples",
    },
    "pan_author_id": {
        "hf_name": None,
        "type": "classification",
        "description": "PAN Author Identification Corpora",
        "manual_download": True,
    },
    # === Plagiarism Detection (Person 2 — preprocessed by Person 1) ===
    "pan_plagiarism": {
        "hf_name": None,
        "type": "similarity",
        "description": "PAN Plagiarism Detection Corpora 2009-2015",
        "manual_download": True,
    },
    "clough_stevenson": {
        "hf_name": None,
        "type": "similarity",
        "description": "Clough & Stevenson plagiarism corpus",
        "manual_download": True,
        "manual_note": "Original URL (ir.shef.ac.uk) is dead. Small corpus (100 docs). Contact authors at University of Sheffield or check ResearchGate.",
    },
    "webis_crowd_paraphrase": {
        "hf_name": None,
        "type": "paraphrase",
        "description": "Webis Crowd Paraphrase Corpus 2011",
        "manual_download": True,
        "manual_note": "Download from https://webis.de/data/webis-cpc-11.html (follow Zenodo link). Small corpus (7,859 samples).",
    },
    "wikisplit": {
        "hf_name": "wiki_split",
        "type": "paraphrase",
        "description": "WikiSplit — 1M sentence splits",
        "load_parquet": True,  # loading script no longer supported
    },
    "sts_benchmark": {
        "hf_name": "mteb/stsbenchmark-sts",
        "type": "similarity",
        "description": "Semantic Textual Similarity Benchmark",
    },
    "paws": {
        "hf_name": "paws",
        "subset": "labeled_final",
        "type": "paraphrase",
        "description": "Adversarial paraphrase pairs",
    },
    # === Humanization (Person 3 — preprocessed by Person 1) ===
    "paranmt": {
        "hf_name": "humarin/chatgpt-paraphrases",
        "type": "paraphrase",
        "description": "ChatGPT paraphrases — modern ParaNMT alternative, 500K+ pairs",
    },
    "qqp": {
        "hf_name": "SetFit/qqp",
        "type": "paraphrase",
        "description": "Quora Question Pairs",
    },
    "mrpc": {
        "hf_name": "glue",
        "subset": "mrpc",
        "type": "paraphrase",
        "description": "Microsoft Research Paraphrase Corpus",
    },
    "bea_2019_gec": {
        "hf_name": "wi_locness",
        "type": "paraphrase",
        "description": "BEA-2019 GEC — Write & Improve + LOCNESS error correction corpus",
        "load_parquet": True,  # loading script no longer supported
    },
}

# ─── Model configs ───────────────────────────────────────
MODELS = {
    "deberta": {
        "name": "microsoft/deberta-v3-large",
        "max_length": 512,
        "learning_rate": 2e-5,
        "batch_size": 8,
        "gradient_accumulation_steps": 2,
        "epochs": 3,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "datasets": ["raid", "hc3", "faidset"],
    },
    "roberta": {
        "name": "roberta-large",
        "max_length": 512,
        "learning_rate": 2e-5,
        "batch_size": 8,
        "gradient_accumulation_steps": 2,
        "epochs": 3,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "datasets": ["raid", "hc3", "gpt2_output"],
    },
    "longformer": {
        "name": "allenai/longformer-base-4096",
        "max_length": 4096,
        "learning_rate": 2e-5,
        "batch_size": 2,
        "gradient_accumulation_steps": 8,
        "epochs": 3,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "datasets": ["raid"],
    },
    "xlm_roberta": {
        "name": "xlm-roberta-large",
        "max_length": 512,
        "learning_rate": 2e-5,
        "batch_size": 8,
        "gradient_accumulation_steps": 2,
        "epochs": 3,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "datasets": ["faidset"],
    },
}

# ─── Training ────────────────────────────────────────────
SEED = 42
SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
NUM_LABELS = 2  # binary: 0=human, 1=AI
META_CLASSIFIER_PATH = CHECKPOINT_DIR / "meta_classifier.joblib"

# ─── Evaluation ──────────────────────────────────────────
EVAL_METRICS = ["accuracy", "f1", "precision", "recall", "roc_auc"]
AI_THRESHOLD = 0.5  # Default threshold for binary decision
