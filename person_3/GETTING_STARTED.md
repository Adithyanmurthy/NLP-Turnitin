# Person 3 - Getting Started Guide

## Welcome!

You are **Person 3** - responsible for the **Humanization & Content Transformation Module**. This guide will help you get started quickly.

## What You Need to Do

Your job is to:
1. Download and preprocess paraphrase datasets
2. Train 4 humanization models (Flan-T5, PEGASUS, Mistral-7B, DIPPER)
3. Build a feedback loop with Person 1's AI detector
4. Provide a `humanize()` function for Person 4 to use

## Quick Start (Automatic)

### Option 1: One Command (Recommended)

```bash
cd person3
python run_all.py
```

This will automatically:
- Download all datasets
- Train all models
- Setup everything

**Time required**: 2-4 weeks (depending on GPU)

### Option 2: Step by Step

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download datasets
python dataset_downloader.py

# 3. Train models (one at a time)
python train_flan_t5.py      # ~1 week
python train_pegasus.py      # ~1 week  
python train_mistral.py      # ~1.5 weeks (optional, needs 24GB GPU)
python setup_dipper.py       # ~1 hour (optional, 11B model)

# 4. Test
python humanizer.py
```

## Check Your Status

At any time, run:
```bash
python check_status.py
```

This will tell you:
- ✓ What's ready
- ✗ What's missing
- ⚠ What's optional

## File Structure

```
person3/
├── GETTING_STARTED.md       ← You are here
├── README.md                ← Full documentation
├── INTEGRATION_GUIDE.md     ← For Person 4
├── requirements.txt         ← Install this first
├── config.py                ← Configuration
├── run_all.py              ← Run everything automatically
├── check_status.py         ← Check what's ready
├── quick_test.py           ← Quick validation
│
├── dataset_downloader.py   ← Downloads all datasets
├── data_loader.py          ← Loads data for training
│
├── train_flan_t5.py        ← Train Flan-T5-XL
├── train_pegasus.py        ← Train PEGASUS-large
├── train_mistral.py        ← Train Mistral-7B
├── setup_dipper.py         ← Setup DIPPER
│
├── humanizer.py            ← Main API (for Person 4)
├── evaluator.py            ← Evaluation metrics
├── example_usage.py        ← Usage examples
│
├── data/                   ← Downloaded datasets go here
├── checkpoints/            ← Trained models go here
└── logs/                   ← Training logs go here
```

## What Each File Does

### Core Files (Must Use)
- **run_all.py** - Runs everything automatically
- **humanizer.py** - Main API that Person 4 will use
- **config.py** - All settings and configurations

### Training Files (Run These)
- **dataset_downloader.py** - Downloads datasets from HuggingFace
- **train_flan_t5.py** - Trains Flan-T5-XL (start here)
- **train_pegasus.py** - Trains PEGASUS-large
- **train_mistral.py** - Trains Mistral-7B (optional, needs big GPU)
- **setup_dipper.py** - Downloads DIPPER (optional)

### Helper Files (Optional)
- **evaluator.py** - Measures humanization quality
- **example_usage.py** - Shows how to use the module
- **check_status.py** - Checks if everything is ready
- **quick_test.py** - Quick validation test

## Minimum Requirements

**To get started:**
- Python 3.10+
- GPU with 8GB VRAM (for Flan-T5 and PEGASUS)
- 32GB RAM
- 100GB free disk space
- Internet connection (for downloading)

**For full functionality:**
- GPU with 24GB VRAM (for Mistral-7B)
- 64GB RAM
- 200GB free disk space

## Timeline

**Week 1**: Setup and data preparation
- Install dependencies
- Download datasets
- Verify everything works

**Week 2**: Train Flan-T5 and PEGASUS
- Start with Flan-T5 (most important)
- Then train PEGASUS
- Test both models

**Week 3**: Train Mistral-7B (optional)
- Only if you have 24GB+ GPU
- Otherwise skip this

**Week 4**: Integration and testing
- Connect with Person 1's AI detector
- Test feedback loop
- Prepare for Person 4 integration

## Common Issues

### "No module named 'torch'"
**Solution**: Install dependencies
```bash
pip install -r requirements.txt
```

### "CUDA out of memory"
**Solution**: Reduce batch size in `config.py`
```python
MODELS = {
    "flan_t5": {
        "batch_size": 2,  # Reduce from 4 to 2
        ...
    }
}
```

### "Dataset download failed"
**Solution**: Check internet connection, try again
```bash
python dataset_downloader.py
```

### "No GPU detected"
**Solution**: Training will be slow on CPU. Consider:
- Google Colab (free GPU)
- AWS/GCP (paid GPU)
- Local GPU setup

## What to Deliver to Person 4

Person 4 needs:
1. ✓ The `humanizer.py` file
2. ✓ Trained model checkpoints in `checkpoints/`
3. ✓ The `INTEGRATION_GUIDE.md` file

They will import your module like this:
```python
from person3.humanizer import humanize
result = humanize("AI-generated text here")
```

## Testing Your Work

### Quick Test
```bash
python quick_test.py
```

### Full Test
```bash
python humanizer.py
```

### Example Usage
```bash
python example_usage.py
```

### Status Check
```bash
python check_status.py
```

## Getting Help

1. **Check status**: `python check_status.py`
2. **Read README**: `README.md` has full documentation
3. **Check config**: `config.py` has all settings
4. **Run examples**: `example_usage.py` shows how to use

## Priority Order

If you're short on time, do this in order:

1. **Must Do**:
   - Install dependencies
   - Download datasets
   - Train Flan-T5-XL (most important model)

2. **Should Do**:
   - Train PEGASUS-large
   - Test humanizer module

3. **Nice to Have**:
   - Train Mistral-7B (if GPU allows)
   - Setup DIPPER
   - Build feedback loop with Person 1

## Integration Checklist

Before sharing with Person 4:
- [ ] At least one model trained (Flan-T5 recommended)
- [ ] `humanizer.py` works without errors
- [ ] `check_status.py` shows "READY" or "PARTIALLY READY"
- [ ] Shared `INTEGRATION_GUIDE.md` with Person 4

## Next Steps

1. **Right now**: Run `python check_status.py`
2. **If not ready**: Run `python run_all.py`
3. **If ready**: Share with Person 4 and read `INTEGRATION_GUIDE.md`

## Questions?

- Check `README.md` for detailed documentation
- Check `INTEGRATION_GUIDE.md` for Person 4 integration
- Run `check_status.py` to see what's missing
- Look at `example_usage.py` for usage examples

Good luck! 🚀
