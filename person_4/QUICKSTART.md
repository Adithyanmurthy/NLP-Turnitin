# Quick Start Guide

## Person 4 Implementation - Complete ✓

This guide helps you get started with the Content Integrity Platform.

## What's Been Implemented (Person 4)

### ✓ Core Integration
- `src/pipeline.py` - Main integration pipeline connecting all three modules
- `src/config.py` - Configuration system with dataclasses
- `src/utils.py` - Utility functions (logging, validation, formatting)

### ✓ Command Line Interface
- `main.py` - Full-featured CLI with argparse
- Support for multiple input methods (string, file, stdin)
- Flexible analysis options (--detect, --plagiarism, --humanize, --full)
- Output formatting (text/JSON)
- Caching support

### ✓ REST API
- `api/app.py` - FastAPI application with middleware
- `api/routes.py` - All API endpoints implemented
- `api/models.py` - Pydantic models for validation
- CORS support, rate limiting, error handling
- Health check endpoint

### ✓ Web Frontend
- `frontend/index.html` - Clean, modern UI
- `frontend/styles.css` - Dark theme styling
- `frontend/script.js` - Interactive JavaScript
- Real-time character counter
- Progress indicators
- Results visualization

### ✓ Testing
- `tests/test_pipeline.py` - Integration tests
- `tests/test_api.py` - API endpoint tests
- Edge case testing
- Error handling tests

### ✓ Documentation
- `README.md` - Complete documentation
- `QUICKSTART.md` - This file
- `requirements.txt` - All dependencies
- `setup.py` - Package setup

## Installation (5 minutes)

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify installation
python -c "import fastapi, torch, transformers; print('✓ All dependencies installed')"
```

## Testing the Implementation

### 1. Test CLI (Without Models)

The CLI works with stub implementations:

```bash
# Test basic analysis
python main.py --input "This is a test text for analysis. Machine learning is fascinating." --detect

# Test with file
echo "Sample text for testing the platform." > test.txt
python main.py --file test.txt --full

# Test JSON output
python main.py --input "Test text" --detect --format json
```

Expected output (with stubs):
```
═══════════════════════════════════════════════════════
  CONTENT ANALYSIS REPORT
═══════════════════════════════════════════════════════

  AI Detection Score:        76.0%
  Classification:            AI-generated

  Processing Time:           0.05s
═══════════════════════════════════════════════════════
```

### 2. Test API (Without Models)

Start the server:
```bash
python api/app.py
```

In another terminal:
```bash
# Health check
curl http://localhost:8000/health

# AI detection
curl -X POST "http://localhost:8000/detect-ai?text=Test+text+for+analysis"

# Full analysis
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Test text", "check_ai": true, "check_plagiarism": true}'
```

### 3. Test Web UI

1. Start server: `python api/app.py`
2. Open browser: `http://localhost:8000`
3. Paste text and click "Analyze"

### 4. Run Tests

```bash
# Run all tests
pytest -v

# Run specific test file
pytest tests/test_pipeline.py -v

# Run with coverage
pytest --cov=src --cov=api
```

## Integration with Person 1, 2, 3

### For Person 1 (AI Detection)

Replace the stub in `src/modules/ai_detector.py`:

```python
class AIDetector:
    def __init__(self, config: AIDetectorConfig):
        # Load your trained models
        self.deberta = load_model(config.deberta_path)
        self.roberta = load_model(config.roberta_path)
        self.longformer = load_model(config.longformer_path)
        self.xlm_roberta = load_model(config.xlm_roberta_path)
        self.meta_classifier = load_model(config.meta_classifier_path)
    
    def detect(self, text: str) -> float:
        # Your implementation
        scores = {
            'deberta': self.deberta.predict(text),
            'roberta': self.roberta.predict(text),
            'longformer': self.longformer.predict(text),
            'xlm_roberta': self.xlm_roberta.predict(text)
        }
        self.last_model_scores = scores
        return self.meta_classifier.predict(scores)
```

### For Person 2 (Plagiarism Detection)

Replace the stub in `src/modules/plagiarism_detector.py`:

```python
class PlagiarismDetector:
    def __init__(self, config: PlagiarismDetectorConfig):
        # Load your models and index
        self.lsh_index = load_lsh_index(config.index_dir)
        self.sentence_bert = load_model(config.sentence_bert_path)
        self.cross_encoder = load_model(config.cross_encoder_path)
    
    def check(self, text: str) -> Dict[str, Any]:
        # Your implementation
        candidates = self.lsh_index.query(text)
        matches = []
        for candidate in candidates:
            similarity = self.compute_similarity(text, candidate)
            if similarity > self.config.similarity_threshold:
                matches.append({
                    'source': candidate.id,
                    'similarity': similarity,
                    'matched_sentences': self.find_matches(text, candidate)
                })
        return {
            'score': max([m['similarity'] for m in matches]) if matches else 0.0,
            'matches': matches,
            'total_matches': len(matches),
            'highest_similarity': max([m['similarity'] for m in matches]) if matches else 0.0
        }
```

### For Person 3 (Humanization)

Replace the stub in `src/modules/humanizer.py`:

```python
class Humanizer:
    def __init__(self, config: HumanizerConfig):
        # Load your models
        self.dipper = load_model(config.dipper_path)
        self.flan_t5 = load_model(config.flan_t5_path)
        self.ai_detector = None  # Will be set by pipeline
    
    def humanize(self, text: str) -> Dict[str, Any]:
        # Your implementation with feedback loop
        ai_score_before = self.ai_detector.detect(text) if self.ai_detector else 0.5
        
        diversity = self.config.initial_diversity
        reorder = self.config.initial_reorder
        iterations = 0
        
        while iterations < self.config.max_iterations:
            humanized = self.dipper.paraphrase(text, diversity, reorder)
            ai_score_after = self.ai_detector.detect(humanized)
            
            if ai_score_after < self.config.target_ai_score:
                break
            
            diversity += self.config.diversity_step
            reorder += self.config.reorder_step
            iterations += 1
        
        return {
            'text': humanized,
            'ai_score_before': ai_score_before,
            'ai_score_after': ai_score_after,
            'iterations': iterations,
            'success': ai_score_after < self.config.target_ai_score
        }
```

## Directory Structure for Models

Person 1, 2, 3 should save their trained models here:

```
models/
├── ai_detection/
│   ├── deberta-v3-large/
│   ├── roberta-large/
│   ├── longformer-base/
│   ├── xlm-roberta-large/
│   └── meta_classifier.pkl
├── plagiarism/
│   ├── sentence-bert/
│   ├── simcse/
│   ├── cross-encoder/
│   └── longformer-similarity/
└── humanization/
    ├── dipper/
    ├── flan-t5-xl/
    ├── pegasus-large/
    └── mistral-7b-lora/
```

## Configuration

Update `src/config.py` with actual model paths:

```python
@dataclass
class AIDetectorConfig:
    model_dir: Path = MODELS_DIR / "ai_detection"
    deberta_path: str = str(model_dir / "deberta-v3-large")
    roberta_path: str = str(model_dir / "roberta-large")
    # ... etc
```

## Next Steps

1. **Person 1**: Implement `ai_detector.py` with trained models
2. **Person 2**: Implement `plagiarism_detector.py` with trained models
3. **Person 3**: Implement `humanizer.py` with trained models
4. **Integration**: Test complete pipeline with all modules
5. **Deployment**: Deploy to production server

## Troubleshooting

### Import Errors
```bash
# Add project root to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Port Already in Use
```bash
# Use different port
uvicorn api.app:app --port 8001
```

### GPU Not Detected
```bash
# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

## Support

- Check `README.md` for detailed documentation
- Run tests to verify functionality
- Check API docs at `http://localhost:8000/docs`

## Summary

Person 4's work is **100% complete** and ready for integration with Person 1, 2, and 3's modules. The system includes:

- ✓ Complete pipeline integration
- ✓ Full-featured CLI
- ✓ REST API with all endpoints
- ✓ Modern web interface
- ✓ Comprehensive test suite
- ✓ Complete documentation

All interfaces are defined and tested with stubs. Once Person 1, 2, and 3 implement their modules following the defined contracts, the entire system will work seamlessly.
