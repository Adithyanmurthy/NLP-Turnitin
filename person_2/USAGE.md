# Usage Guide - Person 2: Plagiarism Detection Engine

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download NLTK Data

```python
python -c "import nltk; nltk.download('punkt')"
```

### 3. Build Reference Index

First, you need to build the MinHash/LSH index from your reference corpus:

```bash
python scripts/build_index.py \
    --corpus_path /path/to/pan_corpus \
    --output_path reference_index \
    --num_perm 128 \
    --threshold 0.5 \
    --shingle_size 3
```

**Parameters:**
- `corpus_path`: Directory containing reference documents (*.txt files)
- `output_path`: Where to save the index
- `num_perm`: Number of MinHash permutations (higher = more accurate, slower)
- `threshold`: Jaccard similarity threshold for LSH
- `shingle_size`: Size of word n-grams

### 4. Train Models (Optional)

If you want to fine-tune the models on your specific data:

#### Train Sentence-BERT

```bash
python models/train_sentence_bert.py \
    --output_path checkpoints/sbert \
    --batch_size 16 \
    --epochs 3
```

#### Train Cross-Encoder

```bash
python models/train_cross_encoder.py \
    --output_path checkpoints/cross_encoder \
    --batch_size 16 \
    --epochs 3
```

### 5. Detect Plagiarism

#### From Command Line

```bash
# Check text directly
python scripts/detect_plagiarism.py \
    --text "Your text to check for plagiarism..." \
    --index_path reference_index \
    --models_path checkpoints

# Check text from file
python scripts/detect_plagiarism.py \
    --file input.txt \
    --index_path reference_index \
    --output report.json
```

#### From Python Code

```python
from src.plagiarism_detector import PlagiarismDetector

# Initialize detector
detector = PlagiarismDetector(
    index_path="reference_index",
    models_path="checkpoints",  # Optional: use fine-tuned models
    use_sbert=True,
    use_cross_encoder=True
)

# Check text
text = """
Your text to check for plagiarism goes here.
It can be multiple paragraphs.
"""

report = detector.check(
    text=text,
    top_k_candidates=5,
    sentence_threshold=0.75,
    verification_threshold=0.7
)

# Display results
print(f"Plagiarism Score: {report['score']:.2%}")
print(f"Verdict: {report['verdict']}")
print(f"Matches Found: {report['num_matches']}")

# Access detailed matches
for match in report['matches']:
    print(f"\nSource: {match['source']}")
    print(f"Similarity: {match['similarity']:.2%}")
    for sent in match['sentences'][:3]:  # Top 3 sentences
        print(f"  - {sent['input_sentence'][:80]}...")
        print(f"    Matched: {sent['matched_sentence'][:80]}...")
        print(f"    Score: {sent['score']:.2%}")
```

## Advanced Usage

### Building Index from Dataset

If you have documents in a Python list:

```python
from src.reference_index import ReferenceIndexBuilder

documents = [
    {'id': 'doc1', 'text': 'Document 1 text...'},
    {'id': 'doc2', 'text': 'Document 2 text...'},
    # ... more documents
]

builder = ReferenceIndexBuilder(
    num_perm=128,
    threshold=0.5,
    shingle_size=3
)

builder.build_from_dataset(documents, output_path="reference_index")
```

### Using Individual Models

#### Sentence-BERT

```python
from src.similarity_models import SentenceBERTModel

model = SentenceBERTModel("sentence-transformers/all-mpnet-base-v2")

sentences1 = ["This is sentence one.", "This is sentence two."]
sentences2 = ["This is sentence one.", "Something different."]

similarity_matrix = model.compute_similarity(sentences1, sentences2)
print(similarity_matrix)
```

#### Cross-Encoder

```python
from src.similarity_models import CrossEncoderModel

model = CrossEncoderModel("cross-encoder/nli-deberta-v3-large")

pairs = [
    ("This is a test.", "This is a test."),
    ("This is a test.", "Something completely different.")
]

scores = model.predict(pairs)
print(scores)  # [high_score, low_score]
```

### Querying the Index Directly

```python
from src.reference_index import ReferenceIndexQuery

index = ReferenceIndexQuery("reference_index")

# Find similar documents
query_text = "Your query text..."
candidates = index.query(query_text, top_k=10)

for doc_id, jaccard_sim in candidates:
    print(f"{doc_id}: {jaccard_sim:.3f}")
    doc_text = index.get_document(doc_id)
    print(doc_text[:100])
```

## Evaluation

### Evaluate on PAN Corpus

```bash
python scripts/evaluate_on_pan.py \
    --test_path data/pan_test.json \
    --index_path reference_index \
    --models_path checkpoints \
    --threshold 0.5 \
    --output evaluation_results.json
```

**Test set format (JSON):**
```json
[
    {
        "text": "Text to check...",
        "label": 1,
        "source": "optional_source_id"
    },
    ...
]
```

## Configuration Options

### Detection Parameters

- `top_k_candidates`: Number of candidate documents to retrieve (default: 5)
- `sentence_threshold`: Similarity threshold for sentence matching (default: 0.75)
- `verification_threshold`: Cross-encoder verification threshold (default: 0.7)
- `min_sentence_length`: Minimum sentence length to consider (default: 10)

### Model Selection

```python
detector = PlagiarismDetector(
    index_path="reference_index",
    use_sbert=True,          # Use Sentence-BERT
    use_simcse=False,        # Use SimCSE instead
    use_cross_encoder=True,  # Enable cross-encoder verification
    use_longformer=False,    # Enable Longformer for long docs
    device="cuda"            # Use GPU
)
```

## Output Format

The `check()` method returns a dictionary:

```python
{
    "score": 0.85,  # Overall plagiarism score (0.0-1.0)
    "num_matches": 2,
    "verdict": "High plagiarism detected",
    "matches": [
        {
            "source": "doc123.txt",
            "similarity": 0.92,
            "jaccard_similarity": 0.65,
            "num_matched_sentences": 5,
            "sentences": [
                {
                    "input_sentence": "...",
                    "matched_sentence": "...",
                    "score": 0.95
                }
            ]
        }
    ]
}
```

## Performance Tips

1. **GPU Acceleration**: Use `device="cuda"` for faster inference
2. **Batch Processing**: Process multiple documents in batches
3. **Index Tuning**: Adjust `num_perm` and `threshold` for speed/accuracy tradeoff
4. **Model Selection**: Disable cross-encoder for faster (but less accurate) detection
5. **Caching**: The index is loaded once and reused for multiple queries

## Troubleshooting

### Out of Memory

- Reduce batch size in training scripts
- Use smaller models (e.g., base instead of large)
- Process documents in smaller chunks

### Slow Detection

- Reduce `top_k_candidates`
- Increase LSH `threshold` (fewer candidates)
- Disable cross-encoder verification
- Use CPU for small batches

### Low Accuracy

- Increase `num_perm` in index
- Lower LSH `threshold` (more candidates)
- Enable cross-encoder verification
- Fine-tune models on domain-specific data

## Integration with Other Modules

### Interface Contract

```python
def check(text: str) -> dict:
    """
    Check text for plagiarism.
    
    Args:
        text: Input text to check
    
    Returns:
        {
            "score": float,  # 0.0-1.0
            "matches": [
                {
                    "source": str,
                    "similarity": float,
                    "sentences": [...]
                }
            ]
        }
    """
```

This matches the interface expected by Person 4's integration module.
