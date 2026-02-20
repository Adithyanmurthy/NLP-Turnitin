# Quick Start Guide

Get up and running with the Plagiarism Detection Engine in 5 minutes!

## Step 1: Install Dependencies (2 minutes)

```bash
# Navigate to the project directory
cd person2_plagiarism_detection

# Install required packages
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt')"
```

## Step 2: Run the Example (3 minutes)

The example script demonstrates all core functionality without requiring any setup:

```bash
python example.py
```

This will:
1. Build a small reference index from sample documents
2. Query the index for similar documents
3. Run plagiarism detection on three test cases:
   - Paraphrased text (high plagiarism)
   - Original text (no plagiarism)
   - Direct copy (very high plagiarism)

**Expected Output:**
```
==============================================================
PLAGIARISM DETECTION ENGINE - EXAMPLES
==============================================================

==============================================================
Example 1: Building Reference Index
==============================================================

Building index from 4 documents...
Index built successfully!

==============================================================
Example 2: Querying the Index
==============================================================

Query text: Machine learning is a branch of AI that focuses on developing algorithms...

Found 1 candidate matches:
  - doc1: Jaccard similarity = 0.654

==============================================================
Example 3: Plagiarism Detection
==============================================================

Initializing plagiarism detector...
Loaded Sentence-BERT model: sentence-transformers/all-mpnet-base-v2 on cuda
Loaded Cross-Encoder model: cross-encoder/nli-deberta-v3-large on cuda

------------------------------------------------------------
Test Case 1: Paraphrased Text (Expected: High Plagiarism)
------------------------------------------------------------

Results:
  Score: 87.3%
  Verdict: High plagiarism detected
  Matches: 1

------------------------------------------------------------
Test Case 2: Original Text (Expected: Low/No Plagiarism)
------------------------------------------------------------

Results:
  Score: 12.5%
  Verdict: No significant plagiarism detected
  Matches: 0

------------------------------------------------------------
Test Case 3: Direct Copy (Expected: Very High Plagiarism)
------------------------------------------------------------

Results:
  Score: 95.8%
  Verdict: High plagiarism detected
  Matches: 1
```

## Step 3: Try Your Own Text

Create a simple Python script:

```python
from src.plagiarism_detector import PlagiarismDetector

# Initialize detector with the example index
detector = PlagiarismDetector(
    index_path="example_index",
    use_sbert=True,
    use_cross_encoder=True
)

# Check your text
your_text = """
Put your text here to check for plagiarism.
It can be multiple sentences or paragraphs.
"""

report = detector.check(your_text)

print(f"Plagiarism Score: {report['score']:.2%}")
print(f"Verdict: {report['verdict']}")
```

## Next Steps

### For Production Use

1. **Build a Real Index**
   ```bash
   python scripts/build_index.py \
       --corpus_path /path/to/your/documents \
       --output_path reference_index
   ```

2. **Use the CLI Tool**
   ```bash
   python scripts/detect_plagiarism.py \
       --file document.txt \
       --index_path reference_index
   ```

3. **Fine-Tune Models** (Optional)
   ```bash
   python models/train_sentence_bert.py \
       --output_path checkpoints/sbert
   ```

### For Integration

If you're Person 4 integrating this module:

```python
from person2_plagiarism_detection.src.plagiarism_detector import PlagiarismDetector

# Initialize once
detector = PlagiarismDetector(
    index_path="reference_index",
    models_path="checkpoints"
)

# Use in your pipeline
def check_plagiarism(text: str) -> dict:
    return detector.check(text)
```

## Troubleshooting

### "No module named 'datasketch'"
```bash
pip install datasketch
```

### "CUDA out of memory"
```python
# Use CPU instead
detector = PlagiarismDetector(
    index_path="example_index",
    device="cpu"
)
```

### "No candidates found"
- The index might be too small
- Try lowering the LSH threshold when building the index
- Add more reference documents

## Documentation

- **Full Usage Guide**: See [USAGE.md](USAGE.md)
- **Project Summary**: See [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)
- **API Reference**: See docstrings in `src/` modules

## Support

For issues or questions:
1. Check the documentation files
2. Review the example code
3. Run the unit tests: `pytest tests/ -v`

Happy plagiarism detecting! 🔍
