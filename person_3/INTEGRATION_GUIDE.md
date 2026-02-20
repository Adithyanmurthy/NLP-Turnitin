# Person 3 - Integration Guide

## For Person 4 (Integration Lead)

This guide explains how to integrate Person 3's humanization module into the main pipeline.

## API Contract

Person 3 provides a single main function that Person 4 should use:

```python
from person3.humanizer import humanize

def humanize(text: str) -> dict:
    """
    Humanize AI-generated text
    
    Args:
        text (str): Input text to humanize
    
    Returns:
        dict: {
            "text": str,              # Humanized text
            "ai_score_before": float, # AI detection score before (0.0-1.0)
            "ai_score_after": float,  # AI detection score after (0.0-1.0)
            "iterations": int,        # Number of feedback iterations
            "diversity_used": int,    # Final diversity parameter (0-100)
            "reorder_used": int       # Final reorder parameter (0-100)
        }
    """
```

## Integration Steps

### Step 1: Import the Module

```python
# In Person 4's main pipeline
import sys
from pathlib import Path

# Add person3 to path
sys.path.insert(0, str(Path(__file__).parent / "person3"))

from humanizer import humanize
```

### Step 2: Use in Pipeline

```python
def process_text(text):
    """Example pipeline integration"""
    
    # Step 1: AI Detection (Person 1)
    from person1.ai_detector import detect
    ai_score = detect(text)
    
    # Step 2: Plagiarism Check (Person 2)
    from person2.plagiarism_detector import check
    plagiarism_report = check(text)
    
    # Step 3: Humanization (Person 3) - if needed
    humanized_result = None
    if ai_score > 0.5:  # If likely AI-generated
        humanized_result = humanize(text)
    
    return {
        "ai_score": ai_score,
        "plagiarism": plagiarism_report,
        "humanized": humanized_result
    }
```

### Step 3: Handle Results

```python
result = humanize(text)

# Access humanized text
humanized_text = result["text"]

# Check improvement
improvement = result["ai_score_before"] - result["ai_score_after"]
print(f"AI score reduced by {improvement:.2%}")

# Check if successful
if result["ai_score_after"] < 0.2:
    print("Successfully humanized!")
else:
    print("Humanization partially successful")
```

## Integration with Person 1 (AI Detector)

Person 3's module automatically integrates with Person 1's AI detector for the feedback loop.

### Requirements:
1. Person 1's `ai_detector.py` must be at: `../person1/ai_detector.py`
2. It must export a function: `detect(text: str) -> float`

### Configuration:
Edit `person3/config.py`:

```python
PERSON1_CONFIG = {
    "detector_path": "../person1/ai_detector.py",  # Adjust path if needed
    "checkpoint_path": "../person1/checkpoints",
    "available": False  # Auto-detected at runtime
}
```

### Feedback Loop Behavior:
- If Person 1's detector is available: Uses iterative feedback
- If not available: Single-pass humanization without feedback
- No errors thrown either way - graceful degradation

## Error Handling

```python
try:
    result = humanize(text)
    print(f"Success: {result['text']}")
except FileNotFoundError as e:
    print(f"Model not found: {e}")
    print("Person 3 needs to train models first")
except RuntimeError as e:
    print(f"Runtime error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Configuration Options

Person 4 can customize behavior by editing `person3/config.py`:

### Feedback Loop Settings:
```python
FEEDBACK_CONFIG = {
    "initial_diversity": 60,      # Starting diversity (0-100)
    "initial_reorder": 40,        # Starting reorder (0-100)
    "diversity_increment": 10,    # Increase per iteration
    "reorder_increment": 10,      # Increase per iteration
    "max_diversity": 100,         # Maximum diversity
    "max_reorder": 100,           # Maximum reorder
    "target_ai_score": 0.2,       # Target AI detection score
    "max_iterations": 5           # Maximum feedback iterations
}
```

### Model Selection:
```python
# In Person 4's code
from person3.humanizer import Humanizer

# Use specific model
humanizer = Humanizer(model_name="pegasus", use_feedback=True)
result = humanizer.humanize(text)

# Available models: "flan_t5", "pegasus", "mistral", "dipper"
```

## Performance Considerations

### Speed:
- **Flan-T5**: ~2-3 seconds per text (512 tokens)
- **PEGASUS**: ~2-3 seconds per text
- **Mistral-7B**: ~5-8 seconds per text (larger model)
- **DIPPER**: ~8-10 seconds per text (11B model)

### Memory:
- **Flan-T5**: ~6GB GPU memory
- **PEGASUS**: ~4GB GPU memory
- **Mistral-7B**: ~8GB GPU memory (with 4-bit quantization)
- **DIPPER**: ~22GB GPU memory (or use model parallelism)

### Recommendations:
- Use **Flan-T5** for production (best speed/quality balance)
- Use **PEGASUS** for lower memory requirements
- Use **Mistral-7B** for highest quality (if GPU available)
- Use **DIPPER** for research/testing only

## Testing Integration

```python
# test_integration.py
def test_person3_integration():
    """Test Person 3 integration"""
    from person3.humanizer import humanize
    
    test_text = "AI has revolutionized many industries."
    
    result = humanize(test_text)
    
    assert "text" in result
    assert "ai_score_before" in result
    assert "ai_score_after" in result
    assert isinstance(result["text"], str)
    assert 0.0 <= result["ai_score_before"] <= 1.0
    assert 0.0 <= result["ai_score_after"] <= 1.0
    
    print("✓ Person 3 integration test passed")

if __name__ == "__main__":
    test_person3_integration()
```

## CLI Integration Example

```python
# main.py (Person 4's CLI)
import argparse
from person3.humanizer import humanize

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--humanize", action="store_true")
    args = parser.parse_args()
    
    text = args.input
    
    if args.humanize:
        result = humanize(text)
        print(f"Humanized: {result['text']}")
        print(f"AI Score: {result['ai_score_before']:.2%} → {result['ai_score_after']:.2%}")
    else:
        # Just analyze
        pass

if __name__ == "__main__":
    main()
```

## Web API Integration Example

```python
# app.py (Person 4's FastAPI)
from fastapi import FastAPI
from pydantic import BaseModel
from person3.humanizer import humanize

app = FastAPI()

class HumanizeRequest(BaseModel):
    text: str

@app.post("/humanize")
async def humanize_endpoint(request: HumanizeRequest):
    result = humanize(request.text)
    return result

# Usage:
# POST http://localhost:8000/humanize
# Body: {"text": "Your AI-generated text here"}
```

## Troubleshooting

### Issue: "Model not found"
**Solution**: Person 3 needs to train models first
```bash
cd person3
python run_all.py
```

### Issue: "AI detector not found"
**Solution**: Either:
1. Person 1 completes their module, OR
2. Disable feedback loop in `config.py`:
```python
FEEDBACK_CONFIG = {
    "use_feedback": False
}
```

### Issue: Out of memory
**Solution**: Use smaller model or reduce batch size
```python
humanizer = Humanizer(model_name="pegasus")  # Smaller than Mistral
```

### Issue: Slow performance
**Solution**: 
1. Use GPU if available
2. Use Flan-T5 (fastest)
3. Disable feedback loop for speed

## Dependencies

Person 4 needs to ensure these are installed:
```bash
pip install torch transformers datasets peft bitsandbytes
```

Or use Person 3's requirements:
```bash
pip install -r person3/requirements.txt
```

## Deliverables Checklist

Person 3 provides:
- ✓ `humanizer.py` - Main API module
- ✓ `evaluator.py` - Evaluation utilities
- ✓ `config.py` - Configuration
- ✓ Trained model checkpoints in `checkpoints/`
- ✓ This integration guide

Person 4 needs:
- ✓ Import `humanize` function
- ✓ Call it with text input
- ✓ Handle returned dict
- ✓ Display results in CLI/web interface

## Contact

For integration issues:
1. Check this guide
2. Review `person3/README.md`
3. Test with `person3/example_usage.py`
4. Check `person3/quick_test.py` for setup verification
