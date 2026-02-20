# Content Integrity & Authorship Intelligence Platform

A comprehensive platform for analyzing text content with three integrated modules:
- **AI Detection**: Detect AI-generated text with high accuracy
- **Plagiarism Detection**: Find copied or paraphrased content
- **Humanization**: Transform AI text to human-like writing

## Project Structure

```
nlp-content-integrity/
├── src/
│   ├── modules/          # Person 1, 2, 3's modules
│   ├── pipeline.py       # Person 4: Integration layer
│   ├── config.py         # Person 4: Configuration
│   └── utils.py          # Person 4: Utilities
├── api/
│   ├── app.py           # Person 4: FastAPI backend
│   ├── routes.py        # Person 4: API routes
│   └── models.py        # Person 4: API models
├── frontend/
│   ├── index.html       # Person 4: Web UI
│   ├── styles.css       # Person 4: Styling
│   └── script.js        # Person 4: Frontend logic
├── tests/               # Person 4: Test suite
├── main.py             # Person 4: CLI entry point
└── requirements.txt
```

## Installation

### Prerequisites
- Python 3.10+
- GPU with 8GB+ VRAM (recommended)
- 64GB RAM (recommended)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd nlp-content-integrity
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download required models (Person 1, 2, 3 will provide):
```bash
# Person 1's AI detection models
# Person 2's plagiarism detection models
# Person 3's humanization models
```

## Usage

### Command Line Interface (CLI)

#### Basic Analysis
```bash
# Analyze text directly
python main.py --input "Your text here" --full

# Analyze from file
python main.py --file document.txt --detect --plagiarism

# Read from stdin
echo "Text to analyze" | python main.py --stdin --full
```

#### Specific Analyses
```bash
# AI detection only
python main.py --input "text" --detect

# Plagiarism check only
python main.py --input "text" --plagiarism

# Humanize AI-generated text
python main.py --input "AI text" --humanize
```

#### Output Options
```bash
# Save output to file
python main.py --input "text" --full --output report.json

# JSON format output
python main.py --input "text" --full --format json

# Quiet mode (results only)
python main.py --input "text" --full --quiet
```

### Web Application

1. Start the API server:
```bash
python api/app.py
# Or using uvicorn:
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

2. Open browser to `http://localhost:8000`

3. Use the web interface:
   - Paste text in the input area
   - Select analysis options (AI Detection, Plagiarism, Humanization)
   - Click "Analyze"
   - View results

### API Endpoints

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Complete Analysis
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your text here",
    "check_ai": true,
    "check_plagiarism": true,
    "humanize": false
  }'
```

#### AI Detection Only
```bash
curl -X POST "http://localhost:8000/detect-ai?text=Your+text+here"
```

#### Plagiarism Check Only
```bash
curl -X POST "http://localhost:8000/check-plagiarism?text=Your+text+here"
```

#### Humanization Only
```bash
curl -X POST "http://localhost:8000/humanize?text=Your+AI+text+here"
```

## Testing

Run the test suite:
```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_pipeline.py

# Run with coverage
pytest --cov=src --cov=api
```

## Configuration

Create a `config.yaml` file to customize settings:

```yaml
ai_detector:
  threshold: 0.5
  max_length: 512
  device: cuda

plagiarism_detector:
  similarity_threshold: 0.85
  max_candidates: 10

humanizer:
  target_ai_score: 0.1
  max_iterations: 5

enable_caching: true
log_level: INFO
max_text_length: 50000
```

Load custom config:
```bash
python main.py --config config.yaml --input "text"
```

## Module Interfaces

### Person 1: AI Detector
```python
from src.modules.ai_detector import AIDetector

detector = AIDetector(config)
score = detector.detect(text)  # Returns float 0.0-1.0
```

### Person 2: Plagiarism Detector
```python
from src.modules.plagiarism_detector import PlagiarismDetector

detector = PlagiarismDetector(config)
result = detector.check(text)  # Returns dict with matches
```

### Person 3: Humanizer
```python
from src.modules.humanizer import Humanizer

humanizer = Humanizer(config)
result = humanizer.humanize(text)  # Returns dict with humanized text
```

## Development

### Person 1 (AI Detection)
- Implement `src/modules/ai_detector.py`
- Train DeBERTa, RoBERTa, Longformer, XLM-RoBERTa
- Save models to `models/ai_detection/`
- Implement `detect(text) -> float` method

### Person 2 (Plagiarism Detection)
- Implement `src/modules/plagiarism_detector.py`
- Build MinHash/LSH index
- Train Sentence-BERT, SimCSE, Cross-Encoder
- Save models to `models/plagiarism/`
- Implement `check(text) -> dict` method

### Person 3 (Humanization)
- Implement `src/modules/humanizer.py`
- Train DIPPER, Flan-T5, PEGASUS, Mistral-7B
- Save models to `models/humanization/`
- Implement `humanize(text) -> dict` method
- Integrate with Person 1's detector for feedback loop

### Person 4 (Integration) - COMPLETED ✓
- ✓ Pipeline integration (`src/pipeline.py`)
- ✓ CLI tool (`main.py`)
- ✓ FastAPI backend (`api/`)
- ✓ Web frontend (`frontend/`)
- ✓ Test suite (`tests/`)
- ✓ Configuration system (`src/config.py`)
- ✓ Utilities (`src/utils.py`)

## API Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Performance

Expected processing times (on GPU):
- AI Detection: 0.5-2s per document
- Plagiarism Check: 1-5s per document (depends on corpus size)
- Humanization: 2-10s per document (depends on iterations)

## Troubleshooting

### GPU Out of Memory
- Reduce batch size in config
- Use smaller models
- Process text in chunks

### Slow Performance
- Enable caching
- Use GPU instead of CPU
- Reduce max_text_length

### Module Not Found
- Ensure Person 1, 2, 3 have implemented their modules
- Check that models are downloaded
- Verify Python path includes src/

## License

[Your License Here]

## Contributors

- Person 1: Data Pipeline & AI Detection
- Person 2: Plagiarism Detection Engine
- Person 3: Humanization & Content Transformation
- Person 4: Integration, CLI & Web Application

## Citation

If you use this platform in your research, please cite:

```bibtex
@software{content_integrity_platform,
  title={Content Integrity & Authorship Intelligence Platform},
  author={[Your Team]},
  year={2024},
  url={[Your URL]}
}
```
