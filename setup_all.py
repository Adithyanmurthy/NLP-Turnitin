#!/usr/bin/env python3
"""
UNIFIED SETUP — Downloads & prepares everything for Person 1, 2, 3, 4.

Usage:
    python setup_all.py              (check status, then do everything needed)
    python setup_all.py --check      (only show status, don't do anything)
    python setup_all.py --step 1     (pip install only)
    python setup_all.py --step 2     (download datasets only)
    python setup_all.py --step 3     (preprocess only)
    python setup_all.py --step 4     (person 3 datasets only)
    python setup_all.py --step 5     (download models only)

Flow:
  Phase 0: STATUS CHECK — scans everything, shows what's done / what's pending
  Phase 1: Install all pip requirements (P1 + P2 + P3 + P4)
  Phase 2: Download all datasets (16 datasets for the whole project)
  Phase 3: Preprocess all datasets into train/val/test splits
  Phase 4: Download Person 3 humanization datasets
  Phase 5: Download ALL pre-trained models (10 models, ~70+ GB)

After this completes, run:  python run_all.py
"""

import sys
import os
import subprocess
import time
import argparse
import json
from pathlib import Path

ROOT_DIR = Path(__file__).parent

MAX_RETRIES = 3
RETRY_DELAY = 10  # seconds between retries


# ══════════════════════════════════════════════════════════════════════
#  UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════

def fmt_size(bytes_val):
    """Format bytes into human-readable size."""
    if bytes_val == 0:
        return "—"
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_val < 1024:
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024
    return f"{bytes_val:.1f} PB"


def get_size(path):
    """Get total size of a file or directory."""
    path = Path(path)
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    try:
        for f in path.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
    except (PermissionError, OSError):
        pass
    return total


# ══════════════════════════════════════════════════════════════════════
#  PHASE 0: STATUS CHECK
# ══════════════════════════════════════════════════════════════════════

# All pip packages needed across all 4 persons (union)
ALL_PACKAGES = {
    "torch": "torch",
    "transformers": "transformers",
    "datasets": "datasets",
    "sentence_transformers": "sentence-transformers",
    "sklearn": "scikit-learn",
    "numpy": "numpy",
    "pandas": "pandas",
    "nltk": "nltk",
    "tqdm": "tqdm",
    "accelerate": "accelerate",
    "evaluate": "evaluate",
    "joblib": "joblib",
    "peft": "peft",
    "bitsandbytes": "bitsandbytes",
    "sentencepiece": "sentencepiece",
    "fastapi": "fastapi",
    "wandb": "wandb",
    "huggingface_hub": "huggingface-hub",
    "safetensors": "safetensors",
    "jsonlines": "jsonlines",
    "datasketch": "datasketch",
    "spacy": "spacy",
    "yaml": "pyyaml",
    "tensorboard": "tensorboard",
    "fitz": "PyMuPDF",
    "docx": "python-docx",
    "uvicorn": "uvicorn",
}

# All 16 datasets (Person 1 config)
ALL_DATASETS = {
    "raid": "RAID benchmark (11 LLMs, 11 genres)",
    "hc3": "HC3 (Human vs ChatGPT)",
    "m4": "M4 (Multi-generator, multilingual) [MANUAL]",
    "gpt2_output": "GPT-wiki-intro (150K examples)",
    "faidset": "FAIDSet (Fine-grained AI detection)",
    "pan_author_id": "PAN Author ID [MANUAL]",
    "pan_plagiarism": "PAN Plagiarism [MANUAL]",
    "clough_stevenson": "Clough & Stevenson [MANUAL]",
    "webis_crowd_paraphrase": "Webis Crowd Paraphrase [MANUAL]",
    "wikisplit": "WikiSplit",
    "sts_benchmark": "STS Benchmark",
    "paws": "PAWS (adversarial paraphrases)",
    "paranmt": "ParaNMT (ChatGPT paraphrases)",
    "qqp": "QQP (Quora Question Pairs)",
    "mrpc": "MRPC (Microsoft Paraphrase)",
    "bea_2019_gec": "BEA-2019 GEC (error correction)",
}

# Person 3 data files
P3_DATA_FILES = ["train.jsonl", "validation.jsonl", "test.jsonl", "metadata.json"]

# All 10 pre-trained models
ALL_MODELS = [
    ("P1", "microsoft/deberta-v3-large", "DeBERTa-v3-large", "classification", "~1.7 GB"),
    ("P1", "roberta-large", "RoBERTa-large", "classification", "~1.4 GB"),
    ("P1", "allenai/longformer-base-4096", "Longformer-base", "classification", "~0.6 GB"),
    ("P1", "xlm-roberta-large", "XLM-RoBERTa-large", "classification", "~2.2 GB"),
    ("P2", "sentence-transformers/all-mpnet-base-v2", "Sentence-BERT", "sentence-transformer", "~0.4 GB"),
    ("P2", "cross-encoder/nli-deberta-v3-large", "Cross-Encoder", "cross-encoder", "~1.7 GB"),
    ("P3", "google/pegasus-large", "PEGASUS-large", "seq2seq", "~2.3 GB"),
    ("P3", "google/flan-t5-xl", "Flan-T5-XL", "seq2seq", "~12 GB"),
    ("P3", "mistralai/Mistral-7B-v0.3", "Mistral-7B", "causal-lm", "~14 GB"),
    ("P3", "kalpeshk2011/dipper-paraphraser-xxl", "DIPPER (11B)", "seq2seq", "~44 GB"),
]

# Trained checkpoints (produced by run_all.py, not setup)
ALL_CHECKPOINTS = [
    ("P1", "person_1", "deberta_ai_detector", "DeBERTa AI Detector"),
    ("P1", "person_1", "roberta_ai_detector", "RoBERTa AI Detector"),
    ("P1", "person_1", "longformer_ai_detector", "Longformer AI Detector"),
    ("P1", "person_1", "xlm_roberta_ai_detector", "XLM-RoBERTa AI Detector"),
    ("P1", "person_1", "meta_classifier.joblib", "Meta-Classifier (Ensemble)"),
    ("P2", "person_2", "sbert", "Sentence-BERT (fine-tuned)"),
    ("P2", "person_2", "cross_encoder", "Cross-Encoder (fine-tuned)"),
    ("P3", "person_3", "flan_t5_xl_final", "Flan-T5-XL (fine-tuned)"),
    ("P3", "person_3", "pegasus_large_final", "PEGASUS-large (fine-tuned)"),
    ("P3", "person_3", "mistral_7b_qlora_final", "Mistral-7B QLoRA (fine-tuned)"),
    ("P3", "person_3", "dipper_xxl", "DIPPER (pretrained, saved)"),
]


def check_status():
    """
    Phase 0: Scan everything and return a status dict.
    Returns dict with counts and lists of what's missing.
    """
    status = {}

    print("\n" + "=" * 70)
    print("  PHASE 0: STATUS CHECK — Scanning everything...")
    print("=" * 70)

    # ─── 1. PIP PACKAGES ────────────────────────────────
    print("\n  [1] PIP PACKAGES")
    print("  " + "─" * 50)
    pkg_ok = []
    pkg_missing = []
    for import_name, pip_name in ALL_PACKAGES.items():
        try:
            __import__(import_name)
            print(f"    ✓ {pip_name}")
            pkg_ok.append(pip_name)
        except ImportError:
            print(f"    ✗ {pip_name} — NOT installed")
            pkg_missing.append(pip_name)
    print(f"    → {len(pkg_ok)}/{len(ALL_PACKAGES)} packages installed")
    status["packages"] = {"ok": pkg_ok, "missing": pkg_missing}

    # ─── 2. RAW DATASETS ────────────────────────────────
    print(f"\n  [2] RAW DATASETS (person_1/data/raw/)")
    print("  " + "─" * 50)
    raw_dir = ROOT_DIR / "person_1" / "data" / "raw"
    ds_ok = []
    ds_missing = []
    ds_manual = []
    for name, desc in ALL_DATASETS.items():
        path = raw_dir / name
        exists = path.exists() and any(path.iterdir()) if path.exists() else False
        if exists:
            size = fmt_size(get_size(path))
            print(f"    ✓ {name:<28} {size:>10}  ({desc})")
            ds_ok.append(name)
        elif name in ("pan_author_id", "pan_plagiarism", "m4", "clough_stevenson", "webis_crowd_paraphrase"):
            print(f"    ⚠ {name:<28} {'MANUAL':>10}  ({desc})")
            ds_manual.append(name)
        else:
            print(f"    ✗ {name:<28} {'—':>10}  ({desc})")
            ds_missing.append(name)
    total_ds = len(ALL_DATASETS) - len(ds_manual)  # don't count manual in denominator
    print(f"    → {len(ds_ok)}/{total_ds} auto-downloadable datasets ready")
    if ds_manual:
        print(f"    → {len(ds_manual)} datasets require manual download (PAN corpora)")
    status["datasets"] = {"ok": ds_ok, "missing": ds_missing, "manual": ds_manual}

    # ─── 3. PREPROCESSED SPLITS ─────────────────────────
    print(f"\n  [3] PREPROCESSED SPLITS (person_1/data/splits/)")
    print("  " + "─" * 50)
    splits_dir = ROOT_DIR / "person_1" / "data" / "splits"
    splits_ok = []
    splits_missing = []
    for name in ALL_DATASETS:
        split_path = splits_dir / name
        has_all = all((split_path / f"{s}.jsonl").exists() for s in ["train", "val", "test"])
        if has_all:
            size = fmt_size(get_size(split_path))
            print(f"    ✓ {name:<28} {size:>10}")
            splits_ok.append(name)
        else:
            # Can only preprocess if raw data exists
            raw_exists = name in ds_ok
            note = "(raw data available)" if raw_exists else "(raw data missing)"
            print(f"    ✗ {name:<28} {'—':>10}  {note}")
            splits_missing.append(name)
    print(f"    → {len(splits_ok)}/{len(ALL_DATASETS)} datasets preprocessed")
    status["splits"] = {"ok": splits_ok, "missing": splits_missing}

    # ─── 4. PERSON 3 DATA ───────────────────────────────
    print(f"\n  [4] PERSON 3 DATA (person_3/data/)")
    print("  " + "─" * 50)
    p3_data = ROOT_DIR / "person_3" / "data"
    p3_ok = []
    p3_missing = []
    for fname in P3_DATA_FILES:
        fpath = p3_data / fname
        if fpath.exists():
            print(f"    ✓ {fname:<28} {fmt_size(get_size(fpath)):>10}")
            p3_ok.append(fname)
        else:
            print(f"    ✗ {fname:<28} {'—':>10}")
            p3_missing.append(fname)
    print(f"    → {len(p3_ok)}/{len(P3_DATA_FILES)} Person 3 data files ready")
    status["p3_data"] = {"ok": p3_ok, "missing": p3_missing}

    # ─── 5. PRE-TRAINED MODELS ──────────────────────────
    print(f"\n  [5] PRE-TRAINED MODELS (HuggingFace cache)")
    print("  " + "─" * 50)
    hf_home = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    print(f"    Cache: {hf_home}")
    hf_size = get_size(hf_home)
    print(f"    Size:  {fmt_size(hf_size)}")
    print()

    models_ok = []
    models_missing = []
    for person, model_id, label, mtype, size in ALL_MODELS:
        cached = _is_model_cached(model_id)
        if cached:
            print(f"    ✓ [{person}] {label:<25} ({model_id})")
            models_ok.append(model_id)
        else:
            print(f"    ✗ [{person}] {label:<25} ({model_id}) — NOT cached  {size}")
            models_missing.append((person, model_id, label, mtype, size))
    print(f"    → {len(models_ok)}/{len(ALL_MODELS)} models cached")
    status["models"] = {"ok": models_ok, "missing": models_missing}

    # ─── 6. TRAINED CHECKPOINTS (info only) ─────────────
    print(f"\n  [6] TRAINED CHECKPOINTS (produced by run_all.py)")
    print("  " + "─" * 50)
    ckpts_ok = 0
    for person, folder, name, label in ALL_CHECKPOINTS:
        path = ROOT_DIR / folder / "checkpoints" / name
        exists = path.exists()
        mark = "✓" if exists else "✗"
        size = fmt_size(get_size(path)) if exists else "—"
        print(f"    {mark} [{person}] {label:<35} {size:>10}")
        if exists:
            ckpts_ok += 1
    print(f"    → {ckpts_ok}/{len(ALL_CHECKPOINTS)} checkpoints ready")
    print(f"    (Checkpoints are created by run_all.py, not setup)")
    status["checkpoints_done"] = ckpts_ok

    # ─── SUMMARY ────────────────────────────────────────
    # Count what setup needs to do (not counting checkpoints — that's training)
    total_items = len(ALL_PACKAGES) + (len(ALL_DATASETS) - len(ds_manual)) + len(ALL_DATASETS) + len(P3_DATA_FILES) + len(ALL_MODELS)
    done_items = len(pkg_ok) + len(ds_ok) + len(splits_ok) + len(p3_ok) + len(models_ok)
    pct = round(done_items / total_items * 100) if total_items > 0 else 0

    print("\n" + "=" * 70)
    print("  SETUP PROGRESS (pre-training)")
    print("=" * 70)
    bar_len = 40
    filled = int(bar_len * pct / 100)
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"\n  [{bar}] {pct}%  ({done_items}/{total_items} items)")

    # What needs to be done
    pending = []
    if pkg_missing:
        pending.append(f"  Step 1: Install {len(pkg_missing)} missing packages")
    if ds_missing:
        pending.append(f"  Step 2: Download {len(ds_missing)} missing datasets")
    if splits_missing:
        can_preprocess = [s for s in splits_missing if s in ds_ok]
        pending.append(f"  Step 3: Preprocess {len(can_preprocess)} datasets (have raw data)")
        cant = len(splits_missing) - len(can_preprocess)
        if cant > 0:
            pending.append(f"          ({cant} more after downloading their raw data)")
    if p3_missing:
        pending.append(f"  Step 4: Download {len(p3_missing)} Person 3 data files")
    if models_missing:
        total_model_size = sum(float(m[4].replace("~", "").replace(" GB", "")) for m in models_missing)
        pending.append(f"  Step 5: Download {len(models_missing)} models (~{total_model_size:.0f} GB)")

    if pending:
        print(f"\n  PENDING WORK:")
        for p in pending:
            print(f"    → {p}")
    else:
        print(f"\n  ✓ All setup complete! Ready for training: python run_all.py")

    print()
    status["pending_summary"] = pending
    return status


# ══════════════════════════════════════════════════════════════════════
#  HELPER: Check if model is cached
# ══════════════════════════════════════════════════════════════════════

def _is_model_cached(model_name):
    """Check if a model is already in the HuggingFace cache."""
    try:
        from huggingface_hub import try_to_load_from_cache
        result = try_to_load_from_cache(model_name, "config.json")
        return result is not None and isinstance(result, str)
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════════════
#  STEP 1: INSTALL PIP PACKAGES
# ══════════════════════════════════════════════════════════════════════

def step1_install_requirements(status):
    """Install all pip requirements from all 4 persons."""
    missing = status["packages"]["missing"]
    if not missing:
        print("\n  [Step 1] ✓ All packages already installed — skipping")
        return True

    print(f"\n  [Step 1] Installing {len(missing)} missing packages...")
    py = sys.executable
    req_files = [
        ROOT_DIR / "person_1" / "requirements.txt",
        ROOT_DIR / "person_2" / "requirements.txt",
        ROOT_DIR / "person_3" / "requirements.txt",
        ROOT_DIR / "person_4" / "requirements.txt",
    ]
    all_ok = True
    for req in req_files:
        if req.exists():
            print(f"\n    Installing from {req.parent.name}/requirements.txt ...")
            try:
                subprocess.run([py, "-m", "pip", "install", "-r", str(req)],
                               check=True, capture_output=False)
                print(f"    ✓ {req.parent.name} requirements done")
            except subprocess.CalledProcessError:
                print(f"    ✗ {req.parent.name} requirements failed")
                all_ok = False
    return all_ok


# ══════════════════════════════════════════════════════════════════════
#  STEP 2: DOWNLOAD DATASETS
# ══════════════════════════════════════════════════════════════════════

def step2_download_datasets(status):
    """Download only the missing datasets, with retry logic."""
    missing = status["datasets"]["missing"]
    if not missing:
        print("\n  [Step 2] ✓ All auto-downloadable datasets already present — skipping")
        return True

    print(f"\n  [Step 2] Downloading {len(missing)} missing datasets...")

    # Import person_1 config for dataset definitions
    sys.path.insert(0, str(ROOT_DIR / "person_1"))
    from config import RAW_DIR, DATASETS

    all_ok = True
    for name in missing:
        info = DATASETS.get(name)
        if info is None:
            print(f"\n    [SKIP] {name} — not in person_1 config")
            continue

        raw_path = RAW_DIR / name

        for attempt in range(1, MAX_RETRIES + 1):
            print(f"\n    [{name}] Downloading... (attempt {attempt}/{MAX_RETRIES})")
            try:
                raw_path.mkdir(parents=True, exist_ok=True)

                if info.get("url_download"):
                    _download_url(name, info, raw_path)
                else:
                    _download_hf(name, info, raw_path)

                print(f"    ✓ {name} downloaded")
                break
            except Exception as e:
                err = str(e)[:200]
                print(f"    ✗ {name} failed: {err}")
                if attempt < MAX_RETRIES:
                    wait = RETRY_DELAY * attempt
                    print(f"      Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"      All {MAX_RETRIES} attempts failed. Will skip {name}.")
                    all_ok = False

    return all_ok


def _download_hf(name, info, save_path):
    """Download a single HuggingFace dataset."""
    from datasets import load_dataset

    hf_name = info["hf_name"]
    subset = info.get("subset", None)
    use_parquet = info.get("load_parquet", False)

    if name == "faidset":
        # FAIDSet has inconsistent columns across splits
        ds = load_dataset(
            hf_name,
            data_files={"train": "train.jsonl", "test": "test.jsonl"},
        )
    elif use_parquet:
        # Datasets whose loading scripts are no longer supported —
        # load from the auto-converted parquet files on refs/convert/parquet
        if subset:
            ds = load_dataset(hf_name, subset, revision="refs/convert/parquet")
        else:
            ds = load_dataset(hf_name, revision="refs/convert/parquet")
    elif subset:
        ds = load_dataset(hf_name, subset)
    else:
        ds = load_dataset(hf_name)

    ds.save_to_disk(str(save_path))


def _download_url(name, info, save_path):
    """Download a dataset from a URL."""
    import requests
    import zipfile

    url = info["url_download"]
    resp = requests.get(url, timeout=300, allow_redirects=True)
    resp.raise_for_status()

    content_type = resp.headers.get("content-type", "")
    if "zip" in content_type or url.endswith(".zip"):
        zip_path = save_path / f"{name}.zip"
        zip_path.write_bytes(resp.content)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(save_path)
        zip_path.unlink()
    else:
        out_file = save_path / f"{name}_data.bin"
        out_file.write_bytes(resp.content)


# ══════════════════════════════════════════════════════════════════════
#  STEP 3: PREPROCESS DATASETS
# ══════════════════════════════════════════════════════════════════════

def step3_preprocess(status):
    """Preprocess datasets that have raw data but no splits yet."""
    # Only preprocess datasets that have raw data downloaded
    ds_ok = status["datasets"]["ok"]
    splits_ok = status["splits"]["ok"]
    need_preprocess = [name for name in ds_ok if name not in splits_ok]

    if not need_preprocess:
        print("\n  [Step 3] ✓ All downloaded datasets already preprocessed — skipping")
        return True

    print(f"\n  [Step 3] Preprocessing {len(need_preprocess)} datasets: {', '.join(need_preprocess)}")
    py = sys.executable
    try:
        subprocess.run(
            [py, "scripts/preprocess.py"],
            check=True, capture_output=False,
            cwd=str(ROOT_DIR / "person_1"),
        )
        print(f"\n    ✓ Preprocessing done")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n    ✗ Preprocessing failed (exit code {e.returncode})")
        return False


# ══════════════════════════════════════════════════════════════════════
#  STEP 4: PERSON 3 DATASETS
# ══════════════════════════════════════════════════════════════════════

def step4_person3_datasets(status):
    """Download Person 3 humanization datasets."""
    missing = status["p3_data"]["missing"]
    if not missing:
        print("\n  [Step 4] ✓ Person 3 data already present — skipping")
        return True

    print(f"\n  [Step 4] Downloading Person 3 datasets ({len(missing)} files missing)...")
    py = sys.executable
    try:
        subprocess.run(
            [py, "dataset_downloader.py"],
            check=True, capture_output=False,
            cwd=str(ROOT_DIR / "person_3"),
        )
        print(f"\n    ✓ Person 3 datasets done")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n    ✗ Person 3 datasets failed (exit code {e.returncode})")
        return False


# ══════════════════════════════════════════════════════════════════════
#  STEP 5: DOWNLOAD PRE-TRAINED MODELS
# ══════════════════════════════════════════════════════════════════════

def step5_download_models(status):
    """Download only the missing pre-trained models with retry logic."""
    missing = status["models"]["missing"]
    if not missing:
        print("\n  [Step 5] ✓ All models already cached — skipping")
        return True

    total_size = sum(float(m[4].replace("~", "").replace(" GB", "")) for m in missing)
    print(f"\n  [Step 5] Downloading {len(missing)} missing models (~{total_size:.0f} GB total)...")
    print(f"           Smaller models first, largest last.\n")

    all_ok = True
    for person, model_id, label, mtype, size in missing:
        for attempt in range(1, MAX_RETRIES + 1):
            print(f"    [{person}] {label} ({size})...", end="")
            if attempt > 1:
                print(f" (attempt {attempt}/{MAX_RETRIES})")
            else:
                print()
            try:
                _download_single_model(model_id, mtype)
                print(f"    ✓ {label} cached")
                break
            except Exception as e:
                err = str(e)[:200]
                print(f"    ✗ Failed: {err}")
                if attempt < MAX_RETRIES:
                    wait = RETRY_DELAY * attempt
                    print(f"      Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"      All {MAX_RETRIES} attempts failed for {label}.")
                    all_ok = False

    return all_ok


def _download_single_model(model_name, model_type):
    """Download a single model to HuggingFace cache."""
    if model_type == "sentence-transformer":
        from sentence_transformers import SentenceTransformer
        SentenceTransformer(model_name)
    elif model_type == "cross-encoder":
        from sentence_transformers import CrossEncoder
        CrossEncoder(model_name)
    elif model_type == "classification":
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        AutoTokenizer.from_pretrained(model_name)
        AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    elif model_type == "seq2seq":
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        AutoTokenizer.from_pretrained(model_name)
        AutoModelForSeq2SeqLM.from_pretrained(model_name)
    elif model_type == "causal-lm":
        from transformers import AutoTokenizer
        from huggingface_hub import snapshot_download
        AutoTokenizer.from_pretrained(model_name)
        # For Mistral: config/tokenizer only, full weights during QLoRA training
        snapshot_download(model_name, ignore_patterns=["*.bin", "*.safetensors"])
        print(f"      (Config/tokenizer cached. Full weights download during QLoRA training)")


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Unified Project Setup")
    parser.add_argument("--check", action="store_true",
                        help="Only show status, don't do anything")
    parser.add_argument("--step", type=int, choices=[1, 2, 3, 4, 5],
                        help="Run only a specific step (1-5)")
    args = parser.parse_args()

    print("=" * 70)
    print("  UNIFIED PROJECT SETUP")
    print("  Person 1 + Person 2 + Person 3 + Person 4")
    print("=" * 70)

    # ─── Phase 0: Always check status first ──────────────
    status = check_status()

    if args.check:
        print("\n  (--check mode: status only, no changes made)")
        print("=" * 70)
        return

    # ─── Check if there's anything to do ─────────────────
    nothing_to_do = (
        not status["packages"]["missing"]
        and not status["datasets"]["missing"]
        and not status["splits"]["missing"]
        and not status["p3_data"]["missing"]
        and not status["models"]["missing"]
    )

    if nothing_to_do:
        print("\n  ✓ Everything is already set up! Nothing to do.")
        print("  Next step: python run_all.py")
        print("=" * 70)
        return

    # ─── Execute steps ───────────────────────────────────
    print("\n" + "=" * 70)
    print("  EXECUTING SETUP STEPS...")
    print("=" * 70)

    results = {}
    steps = {
        1: ("Pip packages", step1_install_requirements),
        2: ("Datasets (P1)", step2_download_datasets),
        3: ("Preprocessing", step3_preprocess),
        4: ("Datasets (P3)", step4_person3_datasets),
        5: ("Models", step5_download_models),
    }

    if args.step:
        # Run single step
        label, func = steps[args.step]
        results[label] = func(status)
    else:
        # Run all steps that have pending work
        for step_num in sorted(steps.keys()):
            label, func = steps[step_num]
            results[label] = func(status)

    # ─── Final Summary ───────────────────────────────────
    print("\n" + "=" * 70)
    print("  SETUP RESULTS")
    print("=" * 70)
    for label, ok in results.items():
        mark = "✓ done" if ok else "✗ FAILED"
        print(f"    {label}: {mark}")

    all_ok = all(results.values())
    if all_ok:
        print("\n  All steps completed successfully.")
        print("  Next: python run_all.py")
    else:
        print("\n  Some steps had failures. Re-run to retry:")
        print("    python setup_all.py              (retry everything)")
        print("    python setup_all.py --step 2     (retry datasets only)")
        print("    python setup_all.py --step 5     (retry models only)")
    print("=" * 70)


if __name__ == "__main__":
    main()
