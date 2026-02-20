# Training & Execution Guide

## Quick Start — One Command

After downloading datasets, run everything with:

```bash
python run_project.py
```

That's it. This single command trains all models across all 4 persons and verifies the integrated pipeline.

---

## What Happens When You Run It

### Phase 1 — Person 1: Data Pipeline & AI Detection

Runs first because everyone else depends on Person 1's preprocessed data.

| Step | What it does | Output |
|------|-------------|--------|
| Download | Fetches 10 datasets from HuggingFace, prints instructions for 8 manual ones | `person_1/data/raw/` |
| Preprocess | Converts all datasets to unified JSONL formats, deduplicates, cleans | `person_1/data/processed/` |
| Split | Creates 80/10/10 train/val/test stratified splits | `person_1/data/splits/{dataset}/train.jsonl` |
| Train DeBERTa-v3-large | Primary AI classifier on RAID + HC3 + M4 + FAIDSet | `person_1/checkpoints/deberta_ai_detector/` |
| Train RoBERTa-large | Ensemble member on RAID + HC3 + GPT-2 Output | `person_1/checkpoints/roberta_ai_detector/` |
| Train Longformer-base | Long document detection on RAID (texts >2000 chars) | `person_1/checkpoints/longformer_ai_detector/` |
| Train XLM-RoBERTa-large | Multilingual detection on M4 + FAIDSet | `person_1/checkpoints/xlm_roberta_ai_detector/` |
| Train Meta-Classifier | Logistic regression combining all 4 models | `person_1/checkpoints/meta_classifier.joblib` |
| Evaluate | Full ensemble evaluation on test sets | `person_1/checkpoints/evaluation_report.json` |

### Phase 2 — Person 2 & 3 (Run in Parallel)

Person 2 and Person 3 run simultaneously since they don't depend on each other.

**Person 2 — Plagiarism Detection:**

| Step | What it does | Output |
|------|-------------|--------|
| Build Index | MinHash/LSH reference index from corpus documents | `person_2/reference_index/` |
| Train Sentence-BERT | Fine-tune all-mpnet-base-v2 on STS + PAWS + QQP | `person_2/checkpoints/sbert/` |
| Train Cross-Encoder | Fine-tune DeBERTa-v3 on PAWS + MRPC + STS | `person_2/checkpoints/cross_encoder/` |

**Person 3 — Humanization:**

| Step | What it does | Output |
|------|-------------|--------|
| Load Data | Uses Person 1's preprocessed paraphrase splits (auto-fallback) | `person_3/data/` |
| Train Flan-T5-XL | Seq2seq rewriting model (3B params) | `person_3/checkpoints/flan_t5_xl_final/` |
| Train PEGASUS-large | Abstractive restructuring (568M params) | `person_3/checkpoints/pegasus_large_final/` |
| Train Mistral-7B | QLoRA fine-tuning for full humanization (7B params) | `person_3/checkpoints/mistral_7b_qlora_final/` |
| Setup DIPPER | Optional 11B paraphraser (downloads pretrained weights) | `person_3/checkpoints/dipper_xxl/` |

### Phase 4 — Integration Verification

Loads all three modules through Person 4's pipeline and runs a smoke test.

---

## After Training — Using the Platform

### CLI Usage

```bash
cd person_4

# Full analysis (AI detection + plagiarism + humanization + deplagiarization)
python main.py --input "paste your text here" --full

# AI detection only
python main.py --input "your text" --detect

# Plagiarism check only
python main.py --input "your text" --plagiarism

# Humanize AI text (target: ≤5% AI score, multi-model fallback)
python main.py --input "your text" --humanize

# Deplagiarize text (rewrite plagiarized sections to ≤5%)
python main.py --input "your text" --deplagiarize

# From a file (supports TXT, PDF, DOCX, HTML, MD)
python main.py --file document.txt --full
python main.py --file essay.pdf --detect --plagiarism
python main.py --file report.docx --humanize

# Save output as JSON
python main.py --input "your text" --full --output report.json --format json
```

### Web UI

```bash
cd person_4
python run_server.py
# Open http://localhost:8000 in your browser
# API docs at http://localhost:8000/docs
```

The web UI supports:
- Text input or file upload (drag & drop or click)
- AI Detection, Plagiarism Check, Humanization, Deplagiarization
- File formats: TXT, PDF, DOCX, HTML, MD

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/analyze` | POST | Full text analysis (AI + plagiarism + humanize + deplagiarize) |
| `/detect-ai` | POST | AI detection only |
| `/check-plagiarism` | POST | Plagiarism check only |
| `/humanize` | POST | Humanize text (target ≤5% AI score) |
| `/deplagiarize` | POST | Deplagiarize text (target ≤5% plagiarism) |
| `/upload` | POST | Upload file and extract text |
| `/upload-and-analyze` | POST | Upload file and run analysis in one step |
| `/health` | GET | System health check |

### Python API

```python
import sys
sys.path.insert(0, 'person_4')

from src.pipeline import ContentIntegrityPipeline
from src.config import load_config

pipeline = ContentIntegrityPipeline(load_config())

# Full analysis
report = pipeline.analyze("your text here", check_ai=True, check_plagiarism=True, humanize=True, deplagiarize=True)

print(f"AI Score: {report['ai_detection']['score']:.2%}")
print(f"Plagiarism: {report['plagiarism']['score']:.2%}")
print(f"Humanized: {report['humanization']['text'][:100]}...")
print(f"Deplagiarized: {report['deplagiarization']['text'][:100]}...")

# File input (auto-detects format)
report = pipeline.analyze("path/to/document.pdf", check_ai=True)
```

---

## Advanced Options

```bash
# Run sequentially (if GPU memory is tight)
python run_project.py --sequential

# Skip Person 1 (data already preprocessed)
python run_project.py --skip-p1

# Run only specific persons
python run_project.py --only p1
python run_project.py --only p2 p3
python run_project.py --only p4    # just verify integration

# Skip final verification
python run_project.py --skip-verify
```

---

## Data Flow Diagram

```
person_1/data/raw/          ← Downloaded datasets (18 total)
       │
       ▼
person_1/data/processed/    ← Cleaned, unified JSONL
       │
       ▼
person_1/data/splits/       ← 80/10/10 train/val/test splits
       │
       ├──→ person_1/checkpoints/     ← 4 AI detection models + meta-classifier
       │
       ├──→ person_2/reference_index/ ← MinHash/LSH index
       │    person_2/checkpoints/     ← Sentence-BERT + Cross-Encoder
       │
       └──→ person_3/data/            ← Paraphrase data (auto-copied from P1 splits)
            person_3/checkpoints/     ← Flan-T5, PEGASUS, Mistral-7B
                     │
                     ▼
            person_4/src/pipeline.py  ← Loads all 3 modules, serves CLI + Web API
```

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 16 GB | 24 GB (RTX 4090 / A100) |
| RAM | 32 GB | 64 GB |
| Storage | 150 GB | 300 GB |
| Training time (all models) | ~48 hours | ~24 hours (A100) |

Person 2 & 3 run in parallel by default, saving ~30% wall-clock time compared to sequential.

---

## Troubleshooting

**"Dataset not found" errors in Person 1:**
Some datasets require manual download. The download script prints instructions for each one. Place them in `person_1/data/raw/{dataset_name}/`.

**Person 3 says "No training data found":**
Either run Person 1 first (it preprocesses paraphrase data that Person 3 auto-uses), or run `cd person_3 && python dataset_downloader.py` to download Person 3's own data.

**GPU out of memory:**
- Run `python run_project.py --sequential` to avoid parallel GPU usage
- Reduce batch sizes in the respective `config.py` files
- Use `CUDA_VISIBLE_DEVICES=0` to pin to a single GPU

**Person 4 shows "module not available":**
This means the corresponding person's training didn't complete. The pipeline works in degraded mode — available modules still function. Re-run the failed person's training individually: `cd person_X && python run_all.py`
