# Training From Scratch

Custom transformer models trained entirely from zero — no pretrained weights,
no transfer learning. Everything learned from our datasets only.

## Models

| # | Model | Architecture | Params | Task |
|---|-------|-------------|--------|------|
| 1 | AI Detector | 12-layer Transformer Encoder | ~85M | Binary classification (human vs AI) |
| 2 | Plagiarism Detector | Siamese Transformer Encoder | ~85M | Sentence-pair similarity (0.0–1.0) |
| 3 | Humanizer | 6+6 Encoder-Decoder Transformer | ~60M | Seq2seq rewriting (AI → human) |

## Training Pipeline

Each model has two phases:
1. **Pre-training** — unsupervised (MLM or denoising) on all available text
2. **Fine-tuning** — supervised on task-specific labeled data

## Quick Start

```bash
# 1. Check status (datasets, GPU, dependencies)
python run_all_scratch.py --check

# 2. Train tokenizer first
python run_all_scratch.py --only tok

# 3. Train everything sequentially (single GPU)
python run_all_scratch.py

# 4. Or train individual models with multi-GPU
torchrun --nproc_per_node=8 train_ai_detector_scratch.py
torchrun --nproc_per_node=8 train_plagiarism_detector_scratch.py
torchrun --nproc_per_node=8 train_humanizer_scratch.py

# 5. Evaluate
python run_all_scratch.py --only eval
```

## Multi-GPU Support

Uses PyTorch DistributedDataParallel (DDP). Works with 1 to N GPUs:

```bash
# 1 GPU (automatic)
python train_ai_detector_scratch.py

# 4 GPUs
torchrun --nproc_per_node=4 train_ai_detector_scratch.py

# 8 GPUs
torchrun --nproc_per_node=8 train_ai_detector_scratch.py
```

Set `SCRATCH_NUM_WORKERS=4` on Linux clusters for faster data loading
(defaults to 0 for Windows compatibility).

## Hardware Estimates (8x A100 40GB)

| Model | Pre-training | Fine-tuning | Total |
|-------|-------------|-------------|-------|
| AI Detector | ~8-12 hrs | ~2-3 hrs | ~10-15 hrs |
| Plagiarism Detector | ~6-10 hrs | ~2-3 hrs | ~8-13 hrs |
| Humanizer | ~10-15 hrs | ~3-4 hrs | ~13-19 hrs |
| **All three** | | | **~31-47 hrs** |

## Dependencies

Only 3 packages needed (no HuggingFace transformers):
- `torch>=2.1.0`
- `tokenizers>=0.15.0`
- `numpy>=1.24.0`
