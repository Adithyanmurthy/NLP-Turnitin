"""
Person 1 — Dataset Downloader
Downloads all 18 datasets for the entire team.
- HuggingFace datasets: downloaded automatically via `datasets` library
- URL datasets: downloaded automatically via requests/wget
- Manual datasets: prints instructions (only 2 remain)
"""

import sys
import os
import zipfile
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path
from datasets import load_dataset
from config import RAW_DIR, DATASETS


def download_hf_dataset(name: str, info: dict) -> None:
    """Download a dataset from Hugging Face Hub."""
    save_path = RAW_DIR / name
    if save_path.exists() and any(save_path.iterdir()):
        print(f"  [SKIP] {name} already downloaded at {save_path}")
        return

    save_path.mkdir(parents=True, exist_ok=True)
    hf_name = info["hf_name"]
    subset = info.get("subset", None)
    use_parquet = info.get("load_parquet", False)

    print(f"  [DOWNLOAD] {name} from '{hf_name}' (subset={subset})...")
    try:
        # FAIDSet has inconsistent columns across splits — load each file separately
        if name == "faidset":
            ds = load_dataset(
                hf_name,
                data_files={"train": "train.jsonl", "test": "test.jsonl"},
            )
        elif name == "hc3":
            # HC3 loading script is broken — load the "all" subset from parquet
            try:
                ds = load_dataset(hf_name, "all", revision="refs/convert/parquet")
            except Exception:
                ds = load_dataset(hf_name, "all", trust_remote_code=True)
        elif name == "bea_2019_gec":
            # wi_locness loading script is broken — try parquet, then trust_remote_code
            try:
                ds = load_dataset(hf_name, revision="refs/convert/parquet")
            except Exception:
                try:
                    ds = load_dataset(hf_name, trust_remote_code=True)
                except Exception:
                    ds = load_dataset(hf_name, "wi", trust_remote_code=True)
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
        print(f"  [OK] Saved to {save_path}")
    except Exception as e:
        print(f"  [ERROR] Failed to download {name}: {e}")
        print(f"  → You may need to download this manually.")


def download_url_dataset(name: str, info: dict) -> None:
    """Download a dataset from a direct URL."""
    import requests

    save_path = RAW_DIR / name
    if save_path.exists() and any(save_path.iterdir()):
        print(f"  [SKIP] {name} already downloaded at {save_path}")
        return

    save_path.mkdir(parents=True, exist_ok=True)
    url = info["url_download"]

    print(f"  [DOWNLOAD] {name} from {url}...")
    try:
        resp = requests.get(url, timeout=120, allow_redirects=True)
        resp.raise_for_status()

        # Determine file type and save
        content_type = resp.headers.get("content-type", "")
        if "zip" in content_type or url.endswith(".zip"):
            zip_path = save_path / f"{name}.zip"
            zip_path.write_bytes(resp.content)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(save_path)
            zip_path.unlink()
            print(f"  [OK] Downloaded and extracted to {save_path}")
        else:
            # Save the raw content (HTML page with links, or data file)
            out_file = save_path / f"{name}_data.bin"
            out_file.write_bytes(resp.content)
            print(f"  [OK] Saved to {out_file}")
            print(f"  [NOTE] You may need to manually extract/process files in {save_path}")
    except Exception as e:
        print(f"  [ERROR] Failed to download {name}: {e}")
        print(f"  → Try downloading manually from: {url}")
        print(f"  → Place files in: {save_path}")


def print_manual_instructions(name: str, info: dict) -> None:
    """Print manual download instructions for datasets not on HF Hub."""
    instructions = {
        "pan_author_id": (
            "PAN Author Identification Corpora:\n"
            "  1. Go to https://pan.webis.de/data.html\n"
            "  2. Download 'C10-Attribution' and 'C50-Attribution' from Zenodo\n"
            "  3. Extract the ZIP files\n"
            f"  Place data in: {RAW_DIR / name}"
        ),
        "pan_plagiarism": (
            "PAN Plagiarism Detection Corpora (2009-2015):\n"
            "  1. Go to https://pan.webis.de/data.html\n"
            "  2. Download PAN-PC-09, PAN-PC-10, PAN-PC-11 from Zenodo\n"
            "  3. May require free registration (fill short form)\n"
            f"  Place data in: {RAW_DIR / name}"
        ),
        "m4": (
            "M4 (Multi-Generator, Multi-Domain, Multi-Lingual) Dataset:\n"
            "  The original HuggingFace repo is no longer accessible.\n"
            "  1. Check the paper: https://arxiv.org/abs/2305.14902\n"
            "  2. Contact the authors at MBZUAI for access\n"
            "  3. Or use RAID + HC3 as alternatives (already downloaded)\n"
            f"  Place data in: {RAW_DIR / name}"
        ),
        "clough_stevenson": (
            "Clough & Stevenson Plagiarism Corpus:\n"
            "  Original URL (ir.shef.ac.uk) is no longer available.\n"
            "  1. Small corpus (100 documents, ~20KB)\n"
            "  2. Contact authors at University of Sheffield\n"
            "  3. Or check ResearchGate for the paper/data\n"
            f"  Place data in: {RAW_DIR / name}"
        ),
    }
    if name in instructions:
        print(f"\n  [MANUAL] {instructions[name]}")


def main():
    print("=" * 60)
    print("  DATASET DOWNLOADER — All Datasets")
    print("=" * 60)

    auto_hf = []
    auto_url = []
    manual = []

    for name, info in DATASETS.items():
        if info.get("url_download"):
            auto_url.append((name, info))
        elif info.get("manual_download", False) or info.get("hf_name") is None:
            manual.append((name, info))
        else:
            auto_hf.append((name, info))

    # --- HuggingFace auto downloads ---
    print(f"\n{'─' * 40}")
    print(f"Auto-downloading {len(auto_hf)} datasets from Hugging Face...")
    print(f"{'─' * 40}")
    for name, info in auto_hf:
        print(f"\n[{name}] {info['description']}")
        download_hf_dataset(name, info)

    # --- URL auto downloads ---
    if auto_url:
        print(f"\n{'─' * 40}")
        print(f"Auto-downloading {len(auto_url)} datasets from URLs...")
        print(f"{'─' * 40}")
        for name, info in auto_url:
            print(f"\n[{name}] {info['description']}")
            download_url_dataset(name, info)

    # --- Manual downloads ---
    if manual:
        print(f"\n{'─' * 40}")
        print(f"{len(manual)} datasets require manual download:")
        print(f"{'─' * 40}")
        for name, info in manual:
            print(f"\n[{name}] {info['description']}")
            print_manual_instructions(name, info)

    print(f"\n{'=' * 60}")
    print("  Download phase complete.")
    print(f"  Auto-downloaded (HuggingFace): {len(auto_hf)}")
    print(f"  Auto-downloaded (URL):         {len(auto_url)}")
    print(f"  Manual required:               {len(manual)}")
    print(f"  Data directory:                {RAW_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
