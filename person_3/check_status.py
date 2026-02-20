"""
Person 3 - Status Checker
Comprehensive check of module readiness
"""
import sys
from pathlib import Path
import json

def print_header(text):
    print("\n" + "=" * 80)
    print(text)
    print("=" * 80)

def print_section(text):
    print(f"\n{text}")
    print("-" * 80)

def check_dependencies():
    """Check if all required packages are installed"""
    print_section("[1/6] CHECKING DEPENDENCIES")
    
    required = {
        "torch": "PyTorch",
        "transformers": "Hugging Face Transformers",
        "datasets": "Hugging Face Datasets",
        "peft": "PEFT (LoRA)",
        "bitsandbytes": "BitsAndBytes (Quantization)",
        "sklearn": "Scikit-learn",
        "nltk": "NLTK",
        "tqdm": "Progress bars"
    }
    
    missing = []
    for package, name in required.items():
        try:
            __import__(package)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - NOT INSTALLED")
            missing.append(package)
    
    if missing:
        print(f"\n  Install missing packages:")
        print(f"    pip install {' '.join(missing)}")
        return False
    
    return True

def check_cuda():
    """Check CUDA availability"""
    print_section("[2/6] CHECKING GPU/CUDA")
    
    try:
        import torch
        
        cuda_available = torch.cuda.is_available()
        print(f"  CUDA Available: {cuda_available}")
        
        if cuda_available:
            print(f"  CUDA Version: {torch.version.cuda}")
            print(f"  GPU Count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                props = torch.cuda.get_device_properties(i)
                print(f"    Memory: {props.total_memory / 1024**3:.1f} GB")
        else:
            print("  ⚠ No CUDA GPU detected. Training will be slow on CPU.")
            print("    Consider using Google Colab or cloud GPU.")
        
        return True
    except Exception as e:
        print(f"  ✗ Error checking CUDA: {e}")
        return False

def check_datasets():
    """Check if datasets are downloaded"""
    print_section("[3/6] CHECKING DATASETS")
    
    data_dir = Path("data")
    
    if not data_dir.exists():
        print("  ✗ Data directory not found")
        print("    Run: python dataset_downloader.py")
        return False
    
    required_files = ["train.jsonl", "validation.jsonl", "test.jsonl", "metadata.json"]
    all_exist = True
    
    for filename in required_files:
        filepath = data_dir / filename
        if filepath.exists():
            if filename.endswith(".jsonl"):
                # Count lines
                with open(filepath) as f:
                    count = sum(1 for _ in f)
                print(f"  ✓ {filename} ({count:,} samples)")
            else:
                print(f"  ✓ {filename}")
        else:
            print(f"  ✗ {filename} - NOT FOUND")
            all_exist = False
    
    if not all_exist:
        print("\n  Download datasets:")
        print("    python dataset_downloader.py")
        return False
    
    # Show metadata
    metadata_file = data_dir / "metadata.json"
    if metadata_file.exists():
        try:
            with open(metadata_file) as f:
                metadata = json.load(f)
            print(f"\n  Dataset Summary:")
            print(f"    Total pairs: {metadata.get('total_pairs', 'N/A'):,}")
            print(f"    Sources: {', '.join(metadata.get('sources', []))}")
        except:
            pass
    
    return True

def check_models():
    """Check if models are trained"""
    print_section("[4/6] CHECKING TRAINED MODELS")
    
    checkpoints_dir = Path("checkpoints")
    
    if not checkpoints_dir.exists():
        print("  ✗ Checkpoints directory not found")
        return False
    
    models = {
        "flan_t5_xl_final": "Flan-T5-XL",
        "pegasus_large_final": "PEGASUS-large",
        "mistral_7b_qlora_final": "Mistral-7B (QLoRA)",
        "dipper_xxl": "DIPPER"
    }
    
    trained = []
    for model_dir, model_name in models.items():
        model_path = checkpoints_dir / model_dir
        if model_path.exists():
            # Check if it has model files
            has_model = any(model_path.glob("*.bin")) or any(model_path.glob("*.safetensors"))
            if has_model:
                print(f"  ✓ {model_name}")
                trained.append(model_name)
            else:
                print(f"  ⚠ {model_name} - Directory exists but no model files")
        else:
            print(f"  ✗ {model_name} - NOT TRAINED")
    
    if not trained:
        print("\n  No trained models found. Train models:")
        print("    python run_all.py")
        print("  Or train individually:")
        print("    python train_flan_t5.py")
        print("    python train_pegasus.py")
        print("    python train_mistral.py")
        print("    python setup_dipper.py")
        return False
    
    print(f"\n  {len(trained)}/{len(models)} models ready")
    return len(trained) > 0

def check_integration():
    """Check integration readiness"""
    print_section("[5/6] CHECKING INTEGRATION")
    
    # Check if humanizer can be imported
    try:
        from humanizer import humanize, Humanizer
        print("  ✓ Humanizer module can be imported")
    except ImportError as e:
        print(f"  ✗ Cannot import humanizer: {e}")
        return False
    
    # Check Person 1 integration
    person1_path = Path("../person1/ai_detector.py")
    if person1_path.exists():
        print("  ✓ Person 1's AI detector found (feedback loop enabled)")
    else:
        print("  ⚠ Person 1's AI detector not found (feedback loop disabled)")
        print(f"    Expected at: {person1_path.absolute()}")
    
    # Check if evaluator works
    try:
        from evaluator import HumanizationEvaluator
        print("  ✓ Evaluator module can be imported")
    except ImportError as e:
        print(f"  ✗ Cannot import evaluator: {e}")
        return False
    
    return True

def check_api():
    """Check if API is ready"""
    print_section("[6/6] CHECKING API READINESS")
    
    try:
        # Try to import and check function signature
        from humanizer import humanize
        import inspect
        
        sig = inspect.signature(humanize)
        print(f"  ✓ humanize() function signature: {sig}")
        
        # Check if at least one model is available
        from pathlib import Path
        checkpoints = Path("checkpoints")
        
        models_available = []
        for model in ["flan_t5_xl_final", "pegasus_large_final", "mistral_7b_qlora_final", "dipper_xxl"]:
            if (checkpoints / model).exists():
                models_available.append(model)
        
        if models_available:
            print(f"  ✓ API ready with models: {', '.join(models_available)}")
            print("\n  Usage for Person 4:")
            print("    from person3.humanizer import humanize")
            print("    result = humanize('your text here')")
            return True
        else:
            print("  ✗ No models available for API")
            return False
            
    except Exception as e:
        print(f"  ✗ API check failed: {e}")
        return False

def main():
    """Run all checks"""
    print_header("PERSON 3 - COMPREHENSIVE STATUS CHECK")
    
    results = {
        "Dependencies": check_dependencies(),
        "GPU/CUDA": check_cuda(),
        "Datasets": check_datasets(),
        "Models": check_models(),
        "Integration": check_integration(),
        "API": check_api()
    }
    
    print_header("SUMMARY")
    
    for check, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {check:20s}: {status}")
    
    all_passed = all(results.values())
    critical_passed = results["Dependencies"] and results["Models"]
    
    print("\n" + "=" * 80)
    
    if all_passed:
        print("STATUS: ✓ FULLY READY")
        print("\nYour Person 3 module is complete and ready for integration!")
        print("\nNext steps:")
        print("  1. Share with Person 4 for integration")
        print("  2. Test with: python example_usage.py")
        print("  3. Integration guide: INTEGRATION_GUIDE.md")
    elif critical_passed:
        print("STATUS: ⚠ PARTIALLY READY")
        print("\nCore functionality is ready, but some optional components are missing.")
        print("You can proceed with integration.")
    else:
        print("STATUS: ✗ NOT READY")
        print("\nCritical components are missing.")
        print("\nQuick start:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Run complete pipeline: python run_all.py")
        print("  3. Or follow README.md for step-by-step setup")
    
    print("=" * 80 + "\n")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
