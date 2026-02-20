#!/usr/bin/env python3
"""
Fix Storage — Move HuggingFace cache from C: to D: and resume setup.

Problem: HuggingFace downloads models/datasets to C:\\Users\\<user>\\.cache\\huggingface\\
         which fills up the C: drive.

Solution:
  1. Find and report what HF downloaded to C:
  2. Move the entire HF cache to D:\\hf_cache\\
  3. Set environment variables so all future downloads go to D:
  4. Resume setup_all.py (skips what's already done)

Usage:
    python fix_storage.py
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path


def get_size_gb(path):
    """Get total size of a directory in GB."""
    total = 0
    path = Path(path)
    if not path.exists():
        return 0
    for f in path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return round(total / (1024 ** 3), 2)


def find_hf_cache():
    """Find the HuggingFace cache directory on C:."""
    home = Path.home()
    hf_cache = home / ".cache" / "huggingface"
    return hf_cache


def report_cache(cache_path):
    """Report what's in the HF cache."""
    if not cache_path.exists():
        print(f"  No HuggingFace cache found at {cache_path}")
        return False

    total_size = get_size_gb(cache_path)
    print(f"  HuggingFace cache found: {cache_path}")
    print(f"  Total size: {total_size} GB")

    # Check subdirectories
    hub_path = cache_path / "hub"
    datasets_path = cache_path / "datasets"

    if hub_path.exists():
        hub_size = get_size_gb(hub_path)
        print(f"    Models (hub/): {hub_size} GB")
        # List model folders
        for item in sorted(hub_path.iterdir()):
            if item.is_dir():
                item_size = get_size_gb(item)
                if item_size > 0.01:
                    print(f"      {item.name}: {item_size} GB")

    if datasets_path.exists():
        ds_size = get_size_gb(datasets_path)
        print(f"    Datasets: {ds_size} GB")

    return True


def move_cache(src, dst):
    """Move HF cache from src to dst."""
    src = Path(src)
    dst = Path(dst)

    if not src.exists():
        print(f"  Source not found: {src}")
        return False

    dst.mkdir(parents=True, exist_ok=True)

    print(f"\n  Moving: {src}")
    print(f"      →  {dst}")
    print(f"  This may take a while...")

    try:
        # Move contents (not the folder itself)
        for item in src.iterdir():
            dest_item = dst / item.name
            if dest_item.exists():
                if dest_item.is_dir():
                    # Merge directories
                    print(f"    Merging: {item.name}")
                    for sub in item.rglob("*"):
                        if sub.is_file():
                            rel = sub.relative_to(item)
                            target = dest_item / rel
                            target.parent.mkdir(parents=True, exist_ok=True)
                            if not target.exists():
                                shutil.move(str(sub), str(target))
                else:
                    print(f"    Skipping (exists): {item.name}")
            else:
                print(f"    Moving: {item.name}")
                shutil.move(str(item), str(dest_item))

        # Remove old cache directory
        if src.exists():
            shutil.rmtree(src, ignore_errors=True)
            print(f"\n  ✓ Old cache removed from C:")

        return True
    except Exception as e:
        print(f"\n  ✗ Error during move: {e}")
        print(f"  Some files may have been partially moved.")
        return False


def set_hf_env(new_cache_path):
    """Set environment variables for current process and print instructions for permanent setup."""
    new_cache = str(new_cache_path)

    # Set for current Python process
    os.environ["HF_HOME"] = new_cache
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(Path(new_cache) / "hub")
    os.environ["HF_DATASETS_CACHE"] = str(Path(new_cache) / "datasets")
    os.environ["TRANSFORMERS_CACHE"] = str(Path(new_cache) / "hub")

    print(f"\n  Environment variables set for this session:")
    print(f"    HF_HOME = {new_cache}")
    print(f"    HUGGINGFACE_HUB_CACHE = {Path(new_cache) / 'hub'}")
    print(f"    HF_DATASETS_CACHE = {Path(new_cache) / 'datasets'}")


def set_permanent_env(new_cache_path):
    """Set permanent Windows environment variables using setx."""
    new_cache = str(new_cache_path)
    print(f"\n  Setting permanent environment variables (Windows)...")

    cmds = [
        ["setx", "HF_HOME", new_cache],
        ["setx", "HUGGINGFACE_HUB_CACHE", str(Path(new_cache) / "hub")],
        ["setx", "HF_DATASETS_CACHE", str(Path(new_cache) / "datasets")],
        ["setx", "TRANSFORMERS_CACHE", str(Path(new_cache) / "hub")],
    ]

    for cmd in cmds:
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"    ✓ {cmd[1]} = {cmd[2]}")
        except Exception as e:
            print(f"    ✗ Failed to set {cmd[1]}: {e}")
            print(f"      → Manually run: setx {cmd[1]} \"{cmd[2]}\"")


def main():
    print("=" * 70)
    print("  FIX STORAGE — Move HuggingFace cache from C: to D:")
    print("=" * 70)

    # Step 1: Find and report
    print("\n[STEP 1] Scanning HuggingFace cache on C:...")
    hf_cache = find_hf_cache()
    found = report_cache(hf_cache)

    if not found:
        print("\n  Nothing to move. Cache may already be on D:.")
        print("  Proceeding to resume setup...\n")
    else:
        # Step 2: Move to D:
        new_cache = Path("D:\\hf_cache")
        print(f"\n[STEP 2] Moving cache to {new_cache}...")
        move_ok = move_cache(hf_cache, new_cache)

        if move_ok:
            print(f"\n  ✓ Cache moved to {new_cache}")
        else:
            print(f"\n  ⚠ Move had issues. Check D:\\hf_cache manually.")

        # Step 3: Set environment variables
        print(f"\n[STEP 3] Setting environment variables...")
        set_hf_env(new_cache)
        set_permanent_env(new_cache)

    # If cache wasn't found on C, still set env to D for future downloads
    if not found:
        new_cache = Path("D:\\hf_cache")
        new_cache.mkdir(parents=True, exist_ok=True)
        set_hf_env(new_cache)
        set_permanent_env(new_cache)

    # Step 4: Resume setup
    print(f"\n[STEP 4] Resuming setup_all.py...")
    print(f"  (It will skip what's already downloaded and continue from where it stopped)")
    print()

    try:
        subprocess.run(
            [sys.executable, "setup_all.py"],
            check=True,
            capture_output=False,
            env=os.environ,
        )
    except subprocess.CalledProcessError as e:
        print(f"\n  setup_all.py exited with code {e.returncode}")
        print(f"  Check the errors above and re-run: python fix_storage.py")
    except Exception as e:
        print(f"\n  Error: {e}")

    print("\n" + "=" * 70)
    print("  DONE")
    print("=" * 70)
    print()
    print("  All future HuggingFace downloads will go to D:\\hf_cache\\")
    print("  Your C: drive should have free space now.")
    print()
    print("  IMPORTANT: Close and reopen your terminal/Kiro for the")
    print("  permanent environment variables to take effect.")
    print()
    print("  After that, run:  python run_all.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
