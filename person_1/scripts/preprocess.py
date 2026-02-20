"""
Person 1 — Data Preprocessing Pipeline
Converts all raw datasets into unified formats, cleans, and creates train/val/test splits.

Unified formats:
  - Classification: {"text": "...", "label": 0/1}
  - Paraphrase:     {"input": "...", "output": "..."}
  - Similarity:     {"text_a": "...", "text_b": "...", "score": 0.0-1.0}
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import hashlib
import re
from pathlib import Path
from typing import Generator

import numpy as np
from datasets import load_from_disk, Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from config import RAW_DIR, PROCESSED_DIR, SPLITS_DIR, DATASETS, SPLIT_RATIOS, SEED


# ─── Text Cleaning ───────────────────────────────────────

def clean_text(text: str) -> str:
    """Remove duplicates whitespace, fix encoding, strip extremes."""
    if not isinstance(text, str):
        return ""
    text = text.encode("utf-8", errors="ignore").decode("utf-8")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_valid_text(text: str, min_len: int = 20, max_len: int = 100_000) -> bool:
    """Filter out extremely short or long samples."""
    return min_len <= len(text) <= max_len


def deduplicate(records: list[dict], key: str = "text") -> list[dict]:
    """Remove exact duplicates based on text hash."""
    seen = set()
    unique = []
    for r in records:
        h = hashlib.md5(r.get(key, "").encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(r)
    return unique


# ─── Dataset-Specific Processors ─────────────────────────

def process_raid(raw_path: Path) -> list[dict]:
    """Process RAID dataset into classification format.
    RAID is huge (7M+ records). We cap at 500K to avoid OOM."""
    MAX_RECORDS = 500_000
    records = []
    try:
        ds = load_from_disk(str(raw_path))
        for split_name in ds:
            if len(records) >= MAX_RECORDS:
                break
            for row in tqdm(ds[split_name], desc=f"RAID/{split_name}"):
                text = clean_text(row.get("generation", row.get("text", "")))
                is_human = row.get("model", "") == "human" or row.get("label", 1) == 0
                label = 0 if is_human else 1
                if is_valid_text(text):
                    records.append({"text": text, "label": label})
                    if len(records) >= MAX_RECORDS:
                        print(f"  [CAP] Reached {MAX_RECORDS:,} records, stopping early")
                        break
            # Free the split from memory
            import gc; gc.collect()
    except Exception as e:
        print(f"  [WARN] RAID processing error: {e}")
    return records


def process_hc3(raw_path: Path) -> list[dict]:
    """Process HC3 — human vs ChatGPT answers."""
    records = []
    try:
        ds = load_from_disk(str(raw_path))
        for split_name in ds:
            for row in tqdm(ds[split_name], desc=f"HC3/{split_name}"):
                # HC3 has human_answers and chatgpt_answers lists
                for ans in row.get("human_answers", []):
                    text = clean_text(ans)
                    if is_valid_text(text):
                        records.append({"text": text, "label": 0})
                for ans in row.get("chatgpt_answers", []):
                    text = clean_text(ans)
                    if is_valid_text(text):
                        records.append({"text": text, "label": 1})
    except Exception as e:
        print(f"  [WARN] HC3 processing error: {e}")
    return records


def process_m4(raw_path: Path) -> list[dict]:
    """Process M4 multi-generator dataset."""
    records = []
    try:
        ds = load_from_disk(str(raw_path))
        for split_name in ds:
            for row in tqdm(ds[split_name], desc=f"M4/{split_name}"):
                text = clean_text(row.get("text", ""))
                label = 0 if row.get("label", 1) == 0 else 1  # 0=human
                if is_valid_text(text):
                    records.append({"text": text, "label": label})
    except Exception as e:
        print(f"  [WARN] M4 processing error: {e}")
    return records


def process_gpt2_output(raw_path: Path) -> list[dict]:
    """Process GPT-wiki-intro dataset (aadityaubhat/GPT-wiki-intro).
    Columns: wiki_intro (human), generated_intro (AI), title, etc."""
    records = []
    try:
        ds = load_from_disk(str(raw_path))
        for split_name in ds:
            for row in tqdm(ds[split_name], desc=f"GPT2/{split_name}"):
                # Human-written Wikipedia intro
                wiki = clean_text(row.get("wiki_intro", ""))
                if is_valid_text(wiki):
                    records.append({"text": wiki, "label": 0})
                # GPT-generated intro
                gen = clean_text(row.get("generated_intro", ""))
                if is_valid_text(gen):
                    records.append({"text": gen, "label": 1})
    except Exception as e:
        # Fallback: try jsonl or old format with text/label columns
        print(f"  [WARN] GPT2 Arrow load failed ({e}), trying JSONL fallback...")
        for jsonl_file in raw_path.glob("*.jsonl"):
            with open(jsonl_file, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    row = json.loads(line)
                    text = clean_text(row.get("text", ""))
                    label = int(row.get("label", row.get("ended", 0)))
                    if is_valid_text(text):
                        records.append({"text": text, "label": label})
    return records


def process_faidset(raw_path: Path) -> list[dict]:
    """Process FAIDSet — fine-grained AI detection (downloaded from HuggingFace)."""
    records = []
    # First try loading as HuggingFace Arrow dataset (auto-downloaded)
    try:
        ds = load_from_disk(str(raw_path))
        for split_name in ds:
            for row in tqdm(ds[split_name], desc=f"FAIDSet/{split_name}"):
                text = clean_text(row.get("text", ""))
                raw_label = str(row.get("label", "")).lower()
                if "human" in raw_label:
                    label = 0
                else:
                    label = 1
                if is_valid_text(text):
                    records.append({"text": text, "label": label})
        return records
    except Exception:
        pass
    # Fallback: try raw JSON/JSONL files (manual download)
    for f in raw_path.glob("*.json*"):
        with open(f, encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                    text = clean_text(row.get("text", ""))
                    raw_label = str(row.get("label", "")).lower()
                    if "human" in raw_label:
                        label = 0
                    else:
                        label = 1
                    if is_valid_text(text):
                        records.append({"text": text, "label": label})
                except json.JSONDecodeError:
                    continue
    return records


def process_generic_classification(raw_path: Path, name: str) -> list[dict]:
    """Generic processor for classification datasets."""
    records = []
    try:
        ds = load_from_disk(str(raw_path))
        for split_name in ds:
            for row in tqdm(ds[split_name], desc=f"{name}/{split_name}"):
                text = clean_text(row.get("text", row.get("sentence", "")))
                label = int(row.get("label", 0))
                if is_valid_text(text):
                    records.append({"text": text, "label": label})
    except Exception as e:
        print(f"  [WARN] {name} processing error: {e}")
    return records


# ─── Paraphrase / Similarity Processors ──────────────────

def process_paws(raw_path: Path) -> list[dict]:
    """Process PAWS into paraphrase format."""
    records = []
    try:
        ds = load_from_disk(str(raw_path))
        for split_name in ds:
            for row in tqdm(ds[split_name], desc=f"PAWS/{split_name}"):
                s1 = clean_text(row.get("sentence1", ""))
                s2 = clean_text(row.get("sentence2", ""))
                label = int(row.get("label", 0))
                if is_valid_text(s1) and is_valid_text(s2) and label == 1:
                    records.append({"input": s1, "output": s2})
    except Exception as e:
        print(f"  [WARN] PAWS processing error: {e}")
    return records


def process_sts(raw_path: Path) -> list[dict]:
    """Process STS Benchmark into similarity format."""
    records = []
    try:
        ds = load_from_disk(str(raw_path))
        for split_name in ds:
            for row in tqdm(ds[split_name], desc=f"STS/{split_name}"):
                a = clean_text(row.get("sentence1", ""))
                b = clean_text(row.get("sentence2", ""))
                score = float(row.get("score", row.get("label", 0)))
                # Normalize to 0-1 if on 0-5 scale
                if score > 1.0:
                    score = score / 5.0
                if a and b:
                    records.append({"text_a": a, "text_b": b, "score": round(score, 4)})
    except Exception as e:
        print(f"  [WARN] STS processing error: {e}")
    return records


def process_wikisplit(raw_path: Path) -> list[dict]:
    """Process WikiSplit into paraphrase format.
    Actual columns: complex_sentence, simple_sentence_1, simple_sentence_2.
    We pair complex_sentence with simple_sentence_1 (the primary simplification).
    Capped at 300K to avoid OOM."""
    MAX_RECORDS = 300_000
    records = []
    try:
        ds = load_from_disk(str(raw_path))
        for split_name in ds:
            if len(records) >= MAX_RECORDS:
                break
            cols = ds[split_name].column_names if hasattr(ds[split_name], 'column_names') else []
            print(f"  [DEBUG] WikiSplit {split_name} columns: {cols}")
            for row in tqdm(ds[split_name], desc=f"WikiSplit/{split_name}"):
                inp = clean_text(row.get("complex_sentence", row.get("source", "")))
                # Try all known column name variants
                out = ""
                for col in ["simple_sentence_1", "simple_sentences", "simple_sentence", "target"]:
                    val = row.get(col, "")
                    if val:
                        out = clean_text(val)
                        break
                if is_valid_text(inp, min_len=10) and is_valid_text(out, min_len=10):
                    records.append({"input": inp, "output": out})
                    if len(records) >= MAX_RECORDS:
                        print(f"  [CAP] Reached {MAX_RECORDS:,} records, stopping early")
                        break
    except Exception as e:
        print(f"  [WARN] WikiSplit processing error: {e}")
    return records


def process_qqp(raw_path: Path) -> list[dict]:
    """Process QQP into paraphrase format."""
    records = []
    try:
        ds = load_from_disk(str(raw_path))
        for split_name in ds:
            for row in tqdm(ds[split_name], desc=f"QQP/{split_name}"):
                q1 = clean_text(row.get("text1", row.get("question1", "")))
                q2 = clean_text(row.get("text2", row.get("question2", "")))
                label = int(row.get("label", 0))
                if label == 1 and is_valid_text(q1, min_len=10) and is_valid_text(q2, min_len=10):
                    records.append({"input": q1, "output": q2})
    except Exception as e:
        print(f"  [WARN] QQP processing error: {e}")
    return records


def process_mrpc(raw_path: Path) -> list[dict]:
    """Process MRPC into paraphrase format."""
    records = []
    try:
        ds = load_from_disk(str(raw_path))
        for split_name in ds:
            for row in tqdm(ds[split_name], desc=f"MRPC/{split_name}"):
                s1 = clean_text(row.get("sentence1", ""))
                s2 = clean_text(row.get("sentence2", ""))
                label = int(row.get("label", 0))
                if label == 1 and is_valid_text(s1, min_len=10) and is_valid_text(s2, min_len=10):
                    records.append({"input": s1, "output": s2})
    except Exception as e:
        print(f"  [WARN] MRPC processing error: {e}")
    return records


# ─── Manual-Download Dataset Processors ──────────────────

def process_pan_plagiarism(raw_path: Path) -> list[dict]:
    """Process PAN Plagiarism Detection Corpora into similarity format."""
    records = []
    # PAN corpora typically have source/suspicious document pairs with XML annotations
    for xml_file in raw_path.rglob("*.xml"):
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for feature in root.findall(".//feature"):
                src_ref = feature.get("source_reference", "")
                src_offset = int(feature.get("source_offset", 0))
                src_length = int(feature.get("source_length", 0))
                this_offset = int(feature.get("this_offset", 0))
                this_length = int(feature.get("this_length", 0))
                # Try to read the actual text files
                susp_file = xml_file.with_suffix(".txt")
                src_file = raw_path / "source-document" / src_ref
                if susp_file.exists() and src_file.exists():
                    with open(susp_file, encoding="utf-8", errors="ignore") as f:
                        susp_text = f.read()
                    with open(src_file, encoding="utf-8", errors="ignore") as f:
                        src_text = f.read()
                    text_a = clean_text(susp_text[this_offset:this_offset + this_length])
                    text_b = clean_text(src_text[src_offset:src_offset + src_length])
                    if text_a and text_b:
                        records.append({"text_a": text_a, "text_b": text_b, "score": 1.0})
        except Exception:
            continue
    # Also try plain text pairs in subdirectories
    for txt_file in raw_path.rglob("*.txt"):
        try:
            text = clean_text(txt_file.read_text(encoding="utf-8", errors="ignore"))
            if is_valid_text(text):
                # Store as similarity with self for index building (score=1.0 placeholder)
                records.append({"text_a": text, "text_b": text, "score": 1.0})
        except Exception:
            continue
    return records


def process_clough_stevenson(raw_path: Path) -> list[dict]:
    """Process Clough & Stevenson plagiarism corpus into similarity format."""
    records = []
    for txt_file in raw_path.rglob("*.txt"):
        try:
            text = clean_text(txt_file.read_text(encoding="utf-8", errors="ignore"))
            if is_valid_text(text):
                # Determine plagiarism level from filename or metadata
                fname = txt_file.stem.lower()
                if "near" in fname or "copy" in fname:
                    score = 0.95
                elif "light" in fname:
                    score = 0.75
                elif "heavy" in fname:
                    score = 0.5
                else:
                    score = 0.0  # non-plagiarism
                records.append({"text_a": text, "text_b": text, "score": score})
        except Exception:
            continue
    # Try CSV/TSV format
    for csv_file in raw_path.rglob("*.csv"):
        try:
            import csv
            with open(csv_file, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    text_a = clean_text(row.get("text_a", row.get("source", "")))
                    text_b = clean_text(row.get("text_b", row.get("suspicious", "")))
                    score = float(row.get("score", row.get("label", 0)))
                    if text_a and text_b:
                        records.append({"text_a": text_a, "text_b": text_b, "score": score})
        except Exception:
            continue
    return records


def process_webis_crowd_paraphrase(raw_path: Path) -> list[dict]:
    """Process Webis Crowd Paraphrase Corpus 2011 into paraphrase format.
    Downloaded from Zenodo — may be Arrow format or raw files."""
    records = []
    # First try HuggingFace Arrow format (if downloaded via HF)
    try:
        ds = load_from_disk(str(raw_path))
        for split_name in ds:
            for row in tqdm(ds[split_name], desc=f"Webis/{split_name}"):
                inp = clean_text(row.get("original", row.get("text1", row.get("input", row.get("sentence1", "")))))
                out = clean_text(row.get("paraphrase", row.get("text2", row.get("output", row.get("sentence2", "")))))
                if is_valid_text(inp, min_len=10) and is_valid_text(out, min_len=10):
                    records.append({"input": inp, "output": out})
        if records:
            return records
    except Exception:
        pass
    # Try JSON/JSONL files
    for f in raw_path.rglob("*.json*"):
        try:
            with open(f, encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    row = json.loads(line.strip())
                    inp = clean_text(row.get("original", row.get("text1", row.get("input", ""))))
                    out = clean_text(row.get("paraphrase", row.get("text2", row.get("output", ""))))
                    if is_valid_text(inp, min_len=10) and is_valid_text(out, min_len=10):
                        records.append({"input": inp, "output": out})
        except Exception:
            continue
    # Try TSV files
    for f in raw_path.rglob("*.tsv"):
        try:
            import csv
            with open(f, encoding="utf-8") as fh:
                reader = csv.reader(fh, delimiter="\t")
                for row in reader:
                    if len(row) >= 2:
                        inp = clean_text(row[0])
                        out = clean_text(row[1])
                        if is_valid_text(inp, min_len=10) and is_valid_text(out, min_len=10):
                            records.append({"input": inp, "output": out})
        except Exception:
            continue
    # Try CSV files
    for f in raw_path.rglob("*.csv"):
        try:
            import csv
            with open(f, encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    inp = clean_text(row.get("original", row.get("text1", row.get("sentence1", ""))))
                    out = clean_text(row.get("paraphrase", row.get("text2", row.get("sentence2", ""))))
                    if is_valid_text(inp, min_len=10) and is_valid_text(out, min_len=10):
                        records.append({"input": inp, "output": out})
        except Exception:
            continue
    # Try plain text files (one pair per file or tab-separated)
    for f in raw_path.rglob("*.txt"):
        try:
            with open(f, encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        inp = clean_text(parts[0])
                        out = clean_text(parts[1])
                        if is_valid_text(inp, min_len=10) and is_valid_text(out, min_len=10):
                            records.append({"input": inp, "output": out})
        except Exception:
            continue
    return records


def process_paranmt(raw_path: Path) -> list[dict]:
    """Process chatgpt-paraphrases (humarin/chatgpt-paraphrases) into paraphrase format.
    Capped at 300K to avoid OOM. Auto-detects column names."""
    records = []
    max_records = 300_000
    # First try HuggingFace Arrow format (auto-downloaded)
    try:
        ds = load_from_disk(str(raw_path))
        for split_name in ds:
            cols = ds[split_name].column_names if hasattr(ds[split_name], 'column_names') else []
            print(f"  [DEBUG] ParaNMT {split_name} columns: {cols}")
            for row in tqdm(ds[split_name], desc=f"ParaNMT/{split_name}"):
                # Try multiple column name patterns
                original = ""
                for col in ["text", "source", "input", "sentence", "original"]:
                    val = row.get(col, "")
                    if val:
                        original = clean_text(val)
                        break
                if not is_valid_text(original, min_len=10):
                    continue

                # Try to find paraphrases (could be a list or a single string)
                paraphrases = []
                for col in ["paraphrases", "paraphrase", "target", "output", "generation"]:
                    val = row.get(col, None)
                    if val is not None:
                        if isinstance(val, list):
                            paraphrases = val
                        elif isinstance(val, str) and val:
                            paraphrases = [val]
                        break

                # If no known paraphrase column, try all remaining string columns
                if not paraphrases:
                    for col in cols:
                        if col in ["text", "source", "input", "sentence", "original"]:
                            continue
                        val = row.get(col, None)
                        if isinstance(val, str) and len(val) > 10:
                            paraphrases = [val]
                            break
                        elif isinstance(val, list) and val and isinstance(val[0], str):
                            paraphrases = val
                            break

                for para in paraphrases:
                    out = clean_text(para) if isinstance(para, str) else ""
                    if is_valid_text(out, min_len=10):
                        records.append({"input": original, "output": out})
                        if len(records) >= max_records:
                            return records
        return records
    except Exception as e:
        print(f"  [WARN] ParaNMT Arrow loading error: {e}")
    # Fallback: tab-separated text files (old ParaNMT format)
    for f in raw_path.rglob("*.txt"):
        try:
            with open(f, encoding="utf-8", errors="ignore") as fh:
                for line in tqdm(fh, desc="ParaNMT"):
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        inp = clean_text(parts[0])
                        out = clean_text(parts[1])
                        if is_valid_text(inp, min_len=10) and is_valid_text(out, min_len=10):
                            records.append({"input": inp, "output": out})
                            if len(records) >= max_records:
                                return records
        except Exception:
            continue
    return records


def process_bea_2019_gec(raw_path: Path) -> list[dict]:
    """Process BEA-2019 GEC dataset into paraphrase format (error → corrected).
    When loaded from parquet, columns are: id, text, edits (or similar).
    Falls back to .m2 format or parallel text files."""
    records = []
    # First try HuggingFace Arrow format (parquet download)
    try:
        ds = load_from_disk(str(raw_path))
        for split_name in ds:
            # Debug: print column names on first split
            if hasattr(ds[split_name], 'column_names'):
                print(f"  [DEBUG] BEA-2019 {split_name} columns: {ds[split_name].column_names}")
            for row in tqdm(ds[split_name], desc=f"BEA2019/{split_name}"):
                # Try common column names from the parquet version
                text = clean_text(row.get("text", row.get("sentence", row.get("original", row.get("id", "")))))
                corrected = clean_text(row.get("corrected", row.get("target", row.get("correction", ""))))
                edits = row.get("edits", None)
                if text and corrected and text != corrected and is_valid_text(text, min_len=10):
                    records.append({"input": text, "output": corrected})
                elif text and is_valid_text(text, min_len=10):
                    # If no separate corrected column, use text as both (placeholder)
                    records.append({"input": text, "output": text})
        if records:
            return records
    except Exception as e:
        print(f"  [WARN] BEA-2019 Arrow load: {e}")
    # Fallback: .m2 format
    for m2_file in raw_path.rglob("*.m2"):
        try:
            current_source = None
            corrections = []
            with open(m2_file, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("S "):
                        if current_source and corrections:
                            corrected = current_source
                            inp = clean_text(current_source)
                            out = clean_text(corrected)
                            if is_valid_text(inp, min_len=10):
                                records.append({"input": inp, "output": out})
                        current_source = line[2:]
                        corrections = []
                    elif line.startswith("A "):
                        corrections.append(line[2:])
        except Exception:
            continue
    # Also try parallel text files
    src_files = list(raw_path.rglob("*source*")) + list(raw_path.rglob("*orig*"))
    tgt_files = list(raw_path.rglob("*target*")) + list(raw_path.rglob("*corrected*"))
    if src_files and tgt_files:
        try:
            with open(src_files[0], encoding="utf-8") as sf, open(tgt_files[0], encoding="utf-8") as tf:
                for src_line, tgt_line in zip(sf, tf):
                    inp = clean_text(src_line)
                    out = clean_text(tgt_line)
                    if is_valid_text(inp, min_len=10) and is_valid_text(out, min_len=10):
                        records.append({"input": inp, "output": out})
        except Exception:
            pass
    return records


# ─── Processor Registry ──────────────────────────────────

PROCESSORS = {
    "raid": process_raid,
    "hc3": process_hc3,
    "m4": process_m4,
    "gpt2_output": process_gpt2_output,
    "faidset": process_faidset,
    "pan_author_id": lambda p: process_generic_classification(p, "pan_author_id"),
    "pan_plagiarism": process_pan_plagiarism,
    "clough_stevenson": process_clough_stevenson,
    "webis_crowd_paraphrase": process_webis_crowd_paraphrase,
    "paws": process_paws,
    "sts_benchmark": process_sts,
    "wikisplit": process_wikisplit,
    "paranmt": process_paranmt,
    "qqp": process_qqp,
    "mrpc": process_mrpc,
    "bea_2019_gec": process_bea_2019_gec,
}


# ─── Split & Save ────────────────────────────────────────

def create_splits(records: list[dict], name: str, data_type: str) -> None:
    """Create stratified train/val/test splits and save as JSON Lines."""
    if not records:
        print(f"  [SKIP] No records for {name}")
        return

    np.random.seed(SEED)
    np.random.shuffle(records)

    n = len(records)
    train_end = int(n * SPLIT_RATIOS["train"])
    val_end = train_end + int(n * SPLIT_RATIOS["val"])

    splits = {
        "train": records[:train_end],
        "val": records[train_end:val_end],
        "test": records[val_end:],
    }

    # If classification, try stratified split
    if data_type == "classification" and len(records) > 100:
        labels = [r["label"] for r in records]
        texts_indices = list(range(len(records)))
        try:
            train_idx, temp_idx = train_test_split(
                texts_indices, test_size=0.2, stratify=labels, random_state=SEED
            )
            temp_labels = [labels[i] for i in temp_idx]
            val_idx, test_idx = train_test_split(
                temp_idx, test_size=0.5, stratify=temp_labels, random_state=SEED
            )
            splits = {
                "train": [records[i] for i in train_idx],
                "val": [records[i] for i in val_idx],
                "test": [records[i] for i in test_idx],
            }
        except ValueError:
            pass  # Fall back to random split if stratification fails

    save_dir = SPLITS_DIR / name
    save_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_data in splits.items():
        out_path = save_dir / f"{split_name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for record in split_data:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"    {split_name}: {len(split_data):,} records → {out_path}")


# ─── Main ────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  DATA PREPROCESSING PIPELINE")
    print("=" * 60)

    import gc

    for name, info in DATASETS.items():
        raw_path = RAW_DIR / name
        data_type = info["type"]

        # Skip if already preprocessed
        split_path = SPLITS_DIR / name
        if all((split_path / f"{s}.jsonl").exists() for s in ["train", "val", "test"]):
            print(f"\n  [SKIP] {name} — already preprocessed")
            continue

        print(f"\n{'─' * 40}")
        print(f"Processing: {name} ({info['description']})")
        print(f"  Type: {data_type}")
        print(f"  Raw path: {raw_path}")

        if not raw_path.exists():
            print(f"  [SKIP] Raw data not found. Download first.")
            continue

        # Get the appropriate processor
        processor = PROCESSORS.get(name)
        if processor is None:
            print(f"  [SKIP] No processor defined for {name}. Manual processing needed.")
            continue

        try:
            # Process
            records = processor(raw_path)
            print(f"  Raw records: {len(records):,}")

            # Deduplicate
            dedup_key = "text" if data_type == "classification" else "input" if data_type == "paraphrase" else "text_a"
            records = deduplicate(records, key=dedup_key)
            print(f"  After dedup: {len(records):,}")

            # Save processed
            processed_path = PROCESSED_DIR / f"{name}.jsonl"
            with open(processed_path, "w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"  Saved processed: {processed_path}")

            # Create splits
            create_splits(records, name, data_type)

        except Exception as e:
            print(f"  [ERROR] {name} failed: {e}")
            print(f"  Continuing with next dataset...")

        # Free memory between datasets
        try:
            del records
        except NameError:
            pass
        gc.collect()

    print(f"\n{'=' * 60}")
    print("  Preprocessing complete.")
    print(f"  Processed data: {PROCESSED_DIR}")
    print(f"  Split data:     {SPLITS_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
