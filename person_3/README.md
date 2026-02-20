# Person 3 - Humanization & Content Transformation Module

## Overview
This module is responsible for transforming AI-generated text into naturally human-written text. It trains 4 models (DIPPER, Flan-T5-XL, PEGASUS-large, Mistral-7B) and implements a feedback loop with Person 1's AI detector.

## Quick Start

### 1. Install Dependencies
```bash
cd person3
pip install -r requirements.txt
```

### 2. Run Complete Pipeline (Automatic)
```bash
python run_all.py
```

This will automatically:
- Download all required datasets
- Train all models
- Setup DIPPER
- Test the humanizer

### 3. Manual Step-by-Step

#### Step 1: Download Datasets
```bash
python dataset_downloader.py
```

#### Step 2: Train Models
```bash
# Train Flan-T5-XL (recommended to start with)
python train_flan_t5.py

# Train PEGASUS-large
python train_pegasus.py

# Train Mistral-7B with QLoRA (requires 24GB+ GPU)
python train_mistral.py

# Setup DIPPER (optional, 11B model)
python setup_dipper.py
```

#### Step 3: Test Humanizer
```bash
python humanizer.py
```

## Integration with Other Persons

### Integration with Person 1 (AI Detector)
The humanizer automatically looks for Person 1's AI detector at:
```
../person1/ai_detector.py
```

If found, it will use the feedback loop to iteratively improve humanization until AI detection score drops below threshold.

### Integration with Person 4 (Main Pipeline)
Person 4 can import and use the humanizer like this:

```python
from person3.humanizer import humanize

# Humanize text
result = humanize("Your AI-generated text here")

# Result contains:
# {
#     "text": "humanized text",
#     "ai_score_before": 0.85,
#     "ai_score_after": 0.15,
#     "iterations": 2,
#     "diversity_used": 70,
#     "reorder_used": 50
# }
```

## Directory Structure
```
person3/
├── config.py                 # Configuration for all models and datasets
├── dataset_downloader.py     # Automatic dataset downloader
├── data_loader.py           # Data loading utilities
├── train_flan_t5.py         # Train Flan-T5-XL
├── train_pegasus.py         # Train PEGASUS-large
├── train_mistral.py         # Train Mistral-7B with QLoRA
├── setup_dipper.py          # Setup DIPPER model
├── humanizer.py             # Main humanization API (for Person 4)
├── evaluator.py             # Evaluation metrics
├── run_all.py               # Master script to run everything
├── requirements.txt         # Python dependencies
├── data/                    # Downloaded datasets
├── checkpoints/             # Trained model checkpoints
├── logs/                    # Training logs
└── README.md               # This file
```

## Datasets Used

1. **PAWS** - Adversarial paraphrase pairs
2. **QQP** - Quora Question Pairs (400K+ pairs)
3. **MRPC** - Microsoft Research Paraphrase Corpus
4. **BEA-2019 GEC** - Grammatical error correction (for human imperfections)
5. **STS Benchmark** - Semantic similarity (for meaning preservation)
6. **HC3** - Human vs ChatGPT pairs (AI→Human training)
7. **ParaNMT-50M** - Large-scale paraphrase dataset (optional)

All datasets are automatically downloaded from HuggingFace.

## Models

### 1. Flan-T5-XL (3B parameters)
- **Purpose**: Controlled seq2seq rewriting
- **Training time**: ~1 week on single GPU
- **Recommended**: Start with this model

### 2. PEGASUS-large (568M parameters)
- **Purpose**: Abstractive restructuring
- **Training time**: ~1 week on single GPU

### 3. Mistral-7B with QLoRA (7B parameters)
- **Purpose**: Full humanization with style variation
- **Training time**: ~1.5 weeks on 24GB+ GPU
- **Note**: Uses 4-bit quantization to fit in consumer GPUs

### 4. DIPPER (11B parameters)
- **Purpose**: Paragraph-level paraphrasing with control knobs
- **Note**: Uses pretrained weights, optional fine-tuning

## Feedback Loop

The humanizer implements an iterative feedback loop:

1. Initial humanization with diversity=60, reorder=40
2. Check AI detection score using Person 1's detector
3. If score > threshold (0.2), increase diversity and reorder
4. Re-humanize and check again
5. Repeat up to 5 iterations or until score < threshold

## Configuration

Edit `config.py` to customize:
- Model hyperparameters (learning rate, batch size, epochs)
- Training configuration (splits, gradient accumulation)
- Feedback loop parameters (diversity, reorder, target score)
- Dataset sampling sizes

## Hardware Requirements

**Minimum:**
- GPU: 8GB VRAM (for Flan-T5 and PEGASUS)
- RAM: 32GB
- Storage: 100GB

**Recommended:**
- GPU: 24GB VRAM (for Mistral-7B)
- RAM: 64GB
- Storage: 200GB

**For DIPPER:**
- GPU: 40GB+ VRAM or use model parallelism

## Troubleshooting

### Out of Memory Errors
- Reduce batch size in `config.py`
- Increase gradient accumulation steps
- Use smaller models (Flan-T5 or PEGASUS only)

### Dataset Download Fails
- Check internet connection
- Some datasets may require HuggingFace authentication
- Run `huggingface-cli login` if needed

### Model Training Fails
- Check GPU availability: `torch.cuda.is_available()`
- Verify CUDA installation
- Check disk space for checkpoints

### Integration Issues
- Ensure Person 1's `ai_detector.py` is in `../person1/`
- Check that the detector exports a `detect(text)` function
- Feedback loop will be disabled if detector not found

## Evaluation Metrics

The evaluator calculates:
- **Semantic Similarity**: Meaning preservation (0-1)
- **BLEU Score**: N-gram overlap
- **ROUGE Scores**: Recall-oriented metrics
- **Lexical Diversity**: Word variation (0-1)
- **AI Detection Score**: Before and after humanization

## API Reference

### Main Function: `humanize(text)`

```python
def humanize(text: str) -> dict:
    """
    Humanize AI-generated text
    
    Args:
        text: Input text to humanize
    
    Returns:
        {
            "text": str,              # Humanized text
            "ai_score_before": float, # AI score before (0-1)
            "ai_score_after": float,  # AI score after (0-1)
            "iterations": int,        # Feedback iterations
            "diversity_used": int,    # Final diversity (0-100)
            "reorder_used": int       # Final reorder (0-100)
        }
    """
```

### Advanced Usage

```python
from person3.humanizer import Humanizer

# Initialize with specific model
humanizer = Humanizer(model_name="pegasus", use_feedback=True)

# Humanize with custom parameters
result = humanizer.paraphrase(
    text="Your text here",
    diversity=80,
    reorder=60
)
```

## Timeline

**Week 1**: Data preparation
- Download and preprocess all datasets
- Create train/val/test splits

**Week 2**: Train Flan-T5 and PEGASUS
- Fine-tune both models on paraphrase data
- Evaluate on validation set

**Week 3**: Train Mistral-7B and setup DIPPER
- Fine-tune Mistral with QLoRA
- Download and configure DIPPER

**Week 4**: Feedback loop and integration
- Integrate with Person 1's AI detector
- Build feedback loop
- Test with Person 4's pipeline

## Contact & Support

For issues specific to Person 3's module:
1. Check this README
2. Review `config.py` for configuration options
3. Check training logs in `logs/` directory
4. Verify model checkpoints in `checkpoints/` directory

## License

This module is part of the Content Integrity & Authorship Intelligence Platform project.
