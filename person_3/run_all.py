"""
Person 3 - Master Script
Runs the complete pipeline: download datasets -> train all models -> evaluate

Usage:
    python run_all.py              # Interactive mode (asks before each step)
    python run_all.py --auto       # Non-interactive (runs everything automatically)
"""
import sys
import subprocess
from pathlib import Path
from config import DATA_DIR, CHECKPOINTS_DIR


def run_command(script_name, description):
    """Run a Python script and handle errors"""
    print("\n" + "=" * 80)
    print(f"RUNNING: {description}")
    print("=" * 80)

    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=False,
        )
        print(f"\n✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} failed with error code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n✗ {description} failed: {e}")
        return False


def check_data_exists():
    train_file = DATA_DIR / "train.jsonl"
    return train_file.exists()


def check_model_exists(model_name):
    model_path = CHECKPOINTS_DIR / f"{model_name}_final"
    return model_path.exists()


def ask(prompt, auto_mode):
    """Ask user for confirmation, or auto-accept in auto mode."""
    if auto_mode:
        print(f"{prompt} [auto: yes]")
        return True
    response = input(f"{prompt} (yes/no): ")
    return response.lower() in ["yes", "y"]


def main():
    auto_mode = "--auto" in sys.argv

    print("=" * 80)
    print("PERSON 3 - HUMANIZATION MODULE")
    print("COMPLETE TRAINING PIPELINE")
    if auto_mode:
        print("(Running in AUTO mode — no prompts)")
    print("=" * 80)

    print("\nThis script will:")
    print("  1. Download and preprocess all datasets")
    print("  2. Train Flan-T5-XL model")
    print("  3. Train PEGASUS-large model")
    print("  4. Train Mistral-7B with QLoRA")
    print("  5. Setup DIPPER (optional)")
    print("  6. Run evaluation")

    if not ask("\nDo you want to continue?", auto_mode):
        print("Aborted.")
        return

    # Step 1: Download datasets
    if not check_data_exists():
        print("\n[STEP 1/6] Downloading datasets...")
        if not run_command("dataset_downloader.py", "Dataset Download"):
            print("\n✗ Dataset download failed. Cannot continue.")
            return
    else:
        print("\n[STEP 1/6] Datasets already exist, skipping download")

    # Step 2: Train Flan-T5
    if not check_model_exists("flan_t5_xl"):
        print("\n[STEP 2/6] Training Flan-T5-XL...")
        if not run_command("train_flan_t5.py", "Flan-T5-XL Training"):
            print("\n⚠ Flan-T5 training failed, but continuing...")
    else:
        print("\n[STEP 2/6] Flan-T5-XL already trained, skipping")

    # Step 3: Train PEGASUS
    if not check_model_exists("pegasus_large"):
        print("\n[STEP 3/6] Training PEGASUS-large...")
        if not run_command("train_pegasus.py", "PEGASUS-large Training"):
            print("\n⚠ PEGASUS training failed, but continuing...")
    else:
        print("\n[STEP 3/6] PEGASUS-large already trained, skipping")

    # Step 4: Train Mistral
    if not check_model_exists("mistral_7b_qlora"):
        print("\n[STEP 4/6] Training Mistral-7B with QLoRA...")
        print("  Note: This requires significant GPU memory (24GB+ recommended)")
        if ask("  Continue with Mistral training?", auto_mode):
            if not run_command("train_mistral.py", "Mistral-7B QLoRA Training"):
                print("\n⚠ Mistral training failed, but continuing...")
        else:
            print("  Skipping Mistral training")
    else:
        print("\n[STEP 4/6] Mistral-7B already trained, skipping")

    # Step 5: Setup DIPPER (optional — skip in auto mode by default)
    if not check_model_exists("dipper_xxl"):
        print("\n[STEP 5/6] Setting up DIPPER...")
        print("  Note: DIPPER is 11B parameters and optional")
        if not auto_mode and ask("  Download DIPPER?", False):
            run_command("setup_dipper.py", "DIPPER Setup")
        else:
            print("  Skipping DIPPER setup (optional, 11B params)")
    else:
        print("\n[STEP 5/6] DIPPER already setup, skipping")

    # Step 6: Test humanizer
    print("\n[STEP 6/6] Testing humanizer module...")
    run_command("humanizer.py", "Humanizer Test")

    print("\n" + "=" * 80)
    print("PERSON 3 PIPELINE COMPLETE")
    print("=" * 80)

    print("\nYour humanization module is ready!")
    print("\nTo use in Person 4's integration:")
    print("  from person_3.humanizer import humanize")
    print("  result = humanize('your text here')")


if __name__ == "__main__":
    main()
