"""
Person 3 - Quick Test Script
Tests if the module is ready for integration
"""
import sys
from pathlib import Path

def check_file(filepath, description):
    """Check if a file exists"""
    if Path(filepath).exists():
        print(f"  ✓ {description}")
        return True
    else:
        print(f"  ✗ {description} - NOT FOUND")
        return False

def check_import(module_name, description):
    """Check if a module can be imported"""
    try:
        __import__(module_name)
        print(f"  ✓ {description}")
        return True
    except ImportError as e:
        print(f"  ✗ {description} - IMPORT ERROR: {e}")
        return False

def main():
    print("=" * 80)
    print("PERSON 3 - QUICK TEST")
    print("=" * 80)
    
    all_good = True
    
    # Check Python packages
    print("\n[1/5] Checking Python packages...")
    packages = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("datasets", "Datasets"),
        ("peft", "PEFT (for LoRA)"),
        ("sklearn", "Scikit-learn")
    ]
    
    for pkg, desc in packages:
        if not check_import(pkg, desc):
            all_good = False
    
    # Check configuration files
    print("\n[2/5] Checking configuration files...")
    files = [
        ("config.py", "Configuration"),
        ("dataset_downloader.py", "Dataset downloader"),
        ("humanizer.py", "Humanizer module"),
        ("evaluator.py", "Evaluator module")
    ]
    
    for file, desc in files:
        if not check_file(file, desc):
            all_good = False
    
    # Check directories
    print("\n[3/5] Checking directories...")
    dirs = [
        ("data", "Data directory"),
        ("checkpoints", "Checkpoints directory"),
        ("logs", "Logs directory")
    ]
    
    for dir_path, desc in dirs:
        if not check_file(dir_path, desc):
            all_good = False
    
    # Check datasets
    print("\n[4/5] Checking datasets...")
    data_files = [
        ("data/train.jsonl", "Training data"),
        ("data/validation.jsonl", "Validation data"),
        ("data/test.jsonl", "Test data")
    ]
    
    datasets_exist = True
    for file, desc in data_files:
        if not check_file(file, desc):
            datasets_exist = False
    
    if not datasets_exist:
        print("\n  ⚠ Datasets not found. Run: python dataset_downloader.py")
    
    # Check trained models
    print("\n[5/5] Checking trained models...")
    models = [
        ("checkpoints/flan_t5_xl_final", "Flan-T5-XL"),
        ("checkpoints/pegasus_large_final", "PEGASUS-large"),
        ("checkpoints/mistral_7b_qlora_final", "Mistral-7B"),
        ("checkpoints/dipper_xxl", "DIPPER")
    ]
    
    models_exist = False
    for model_path, desc in models:
        if check_file(model_path, desc):
            models_exist = True
    
    if not models_exist:
        print("\n  ⚠ No trained models found. Run training scripts:")
        print("    - python train_flan_t5.py")
        print("    - python train_pegasus.py")
        print("    - python train_mistral.py")
        print("    - python setup_dipper.py")
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    if all_good and datasets_exist and models_exist:
        print("\n✓ ALL CHECKS PASSED")
        print("\nYour Person 3 module is ready for integration!")
        print("\nTo use in Person 4's pipeline:")
        print("  from person3.humanizer import humanize")
        print("  result = humanize('your text here')")
    elif all_good and datasets_exist:
        print("\n⚠ DEPENDENCIES OK, DATASETS OK, BUT NO TRAINED MODELS")
        print("\nNext steps:")
        print("  1. Run: python run_all.py")
        print("     OR")
        print("  2. Train models individually:")
        print("     - python train_flan_t5.py")
        print("     - python train_pegasus.py")
    elif all_good:
        print("\n⚠ DEPENDENCIES OK, BUT NO DATASETS")
        print("\nNext steps:")
        print("  1. Run: python dataset_downloader.py")
        print("  2. Then run: python run_all.py")
    else:
        print("\n✗ SOME CHECKS FAILED")
        print("\nPlease install missing dependencies:")
        print("  pip install -r requirements.txt")
    
    print()

if __name__ == "__main__":
    main()
