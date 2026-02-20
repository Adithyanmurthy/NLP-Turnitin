"""
Person 3 - Setup DIPPER (Discourse Paraphraser)
DIPPER is already pretrained, we just download and configure it
"""
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from config import MODELS, CHECKPOINTS_DIR

def setup_dipper():
    """Download and setup DIPPER model"""
    print("=" * 80)
    print("SETTING UP DIPPER (DISCOURSE PARAPHRASER)")
    print("=" * 80)
    
    model_config = MODELS["dipper"]
    model_name = model_config["hf_path"]
    
    print(f"\n[1/3] Downloading DIPPER model: {model_name}")
    print("  Note: This is an 11B parameter model, download may take time...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        print(f"\n[2/3] Saving DIPPER to local checkpoints...")
        save_path = CHECKPOINTS_DIR / "dipper_xxl"
        save_path.mkdir(parents=True, exist_ok=True)
        
        model.save_pretrained(str(save_path))
        tokenizer.save_pretrained(str(save_path))
        
        print(f"\n[3/3] Testing DIPPER...")
        test_text = "This is a test sentence to verify DIPPER is working correctly."
        inputs = tokenizer(test_text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=100,
                num_beams=4,
                early_stopping=True
            )
        
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\n  Input: {test_text}")
        print(f"  Output: {result}")
        
        print(f"\n✓ DIPPER setup complete!")
        print(f"  Model saved to: {save_path}")
        print("=" * 80)
        print("DIPPER SETUP COMPLETE")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n✗ Error setting up DIPPER: {e}")
        print("  DIPPER is optional. Continuing without it...")
        print("  You can use Flan-T5, PEGASUS, and Mistral for humanization.")

if __name__ == "__main__":
    setup_dipper()
