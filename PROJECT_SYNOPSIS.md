# School of Computer Science and Engineering
## Project Synopsis

---

### Project Team Members

| # | Student Name | USN | Sign |
|---|---|---|---|
| 1. | __________ | __________ | |
| 2. | __________ | __________ | |
| 3. | __________ | __________ | |
| 4. | __________ | __________ | |

**Team Id:** _______

**Guide Name:** Dr./Prof. __________

---

### Title of the Project

**Multilingual Content Integrity and Authorship Intelligence Platform: An Ensemble Transformer Approach for AI-Generated Text Detection, Semantic Plagiarism Identification, and Adaptive Content Humanization**

---

### Objective / Purpose

The objectives of the proposed work are as follows:

1. To develop a high-accuracy multilingual AI-generated text detection system using an ensemble of fine-tuned transformer models (DeBERTa-v3, RoBERTa, Longformer, XLM-RoBERTa) capable of identifying machine-generated content across 100+ languages with a target accuracy exceeding 95%.

2. To build a semantic plagiarism detection engine that goes beyond traditional n-gram matching by employing sentence-level embeddings (Sentence-BERT, SimCSE) and cross-encoder verification (DeBERTa) to detect paraphrase-level and meaning-level plagiarism.

3. To design an adaptive content humanization module using sequence-to-sequence models (Flan-T5-XL, PEGASUS, DIPPER, Mistral-7B with QLoRA) that transforms AI-generated text into naturally human-written content, achieving near-zero AI detection scores while preserving semantic meaning.

4. To integrate all three modules into a unified, self-contained platform that operates entirely offline without dependency on third-party APIs, providing transparent and reproducible content analysis through a command-line interface and web application.

---

### Abstract (Approximately 10 lines)

The rapid proliferation of large language models has created an urgent need for reliable systems that can detect AI-generated content, identify plagiarism, and ensure content authenticity across multiple languages. Existing commercial tools suffer from high false-positive rates, limited multilingual support, and inability to detect paraphrase-level plagiarism. This project proposes a unified multilingual content intelligence platform built on an ensemble of fine-tuned transformer architectures. The AI detection module employs four specialized classifiers — DeBERTa-v3-large, RoBERTa-large, Longformer-base for long documents, and XLM-RoBERTa-large for multilingual text — combined through a meta-classifier to maximize detection accuracy across diverse text formats and languages. The plagiarism detection engine uses MinHash/LSH for fast candidate retrieval followed by Sentence-BERT and SimCSE embeddings for semantic similarity scoring, with a DeBERTa cross-encoder for verification. The humanization module fine-tunes DIPPER, Flan-T5-XL, PEGASUS, and Mistral-7B to rewrite flagged content into naturally human-like text, employing an iterative feedback loop with the detection module to ensure transformed output registers minimal AI signatures. The system is trained on 18 large-scale datasets including RAID (10M+ documents), M4 (multilingual), HC3, and PAN corpora, and operates entirely locally without external API dependencies.

---

### Hardware & Software Requirements

**Hardware:**
- GPU: NVIDIA RTX 4090 (24 GB VRAM) or NVIDIA A100 (40/80 GB VRAM) for model training
- RAM: 64 GB minimum
- Storage: 500 GB+ NVMe SSD for datasets, model weights, and checkpoints
- CPU: 8+ core processor (Intel i7/i9 or AMD Ryzen 7/9)
- Cloud alternative: RunPod / Lambda Labs / Google Colab Pro+ with A100 GPU instances

**Software:**
- Python 3.10+
- PyTorch 2.0+ with CUDA 12.1+ support
- Hugging Face Transformers 4.36+
- Hugging Face Datasets library
- Sentence-Transformers library
- Datasketch (MinHash/LSH)
- scikit-learn for evaluation metrics and meta-classifier
- NLTK / spaCy for text preprocessing
- FastAPI for web application backend
- Weights & Biases / TensorBoard for experiment tracking

---

### Innovation / Novelty Involved

1. **Unified Three-Module Architecture:** Unlike existing tools that treat AI detection, plagiarism checking, and content transformation as separate problems, this platform integrates all three into a single pipeline sharing a common semantic understanding layer, resulting in higher accuracy and consistency.

2. **Multilingual Ensemble Detection:** The system employs XLM-RoBERTa-large trained on the M4 multilingual dataset, enabling AI detection across 100+ languages — a capability absent in most commercial detectors that focus primarily on English.

3. **Feedback-Loop Humanization:** The content transformation module uses an iterative refinement loop where the output is continuously evaluated by the detection module, adjusting diversity and reordering parameters until the AI detection score falls below the target threshold — ensuring genuinely human-like output.

4. **Semantic Plagiarism Detection:** By combining MinHash/LSH fast screening with Sentence-BERT/SimCSE embeddings and DeBERTa cross-encoder verification, the system detects meaning-level plagiarism that traditional string-matching tools completely miss.

5. **Fully Self-Contained and Transparent:** All models are trained locally on open datasets with no dependency on proprietary APIs, making the system fully reproducible, auditable, and capable of offline operation.

---

### Algorithms / Technology Planned

1. **Transformer-based Sequence Classification** — Fine-tuning DeBERTa-v3-large, RoBERTa-large, Longformer-base, and XLM-RoBERTa-large with binary classification heads for AI-generated text detection.

2. **Ensemble Meta-Classification** — Logistic regression meta-classifier that combines probability outputs from all four detection models into a single optimized prediction score.

3. **MinHash / Locality-Sensitive Hashing (LSH)** — Approximate nearest neighbor search for fast first-pass duplicate and plagiarism candidate retrieval from large reference corpora.

4. **Contrastive Learning for Sentence Embeddings** — Fine-tuning Sentence-BERT (all-mpnet-base-v2) and SimCSE using contrastive loss on semantic similarity datasets for high-quality sentence-level similarity scoring.

5. **Cross-Encoder Pairwise Verification** — DeBERTa-v3 cross-encoder for precise pairwise similarity scoring between candidate plagiarism pairs.

6. **Sequence-to-Sequence Text Generation** — Fine-tuning Flan-T5-XL and PEGASUS-large for controlled paraphrase generation and abstractive text restructuring.

7. **Parameter-Efficient Fine-Tuning (QLoRA)** — Quantized Low-Rank Adaptation for fine-tuning Mistral-7B on consumer hardware with 4-bit quantization.

8. **Discourse-Level Paraphrasing (DIPPER)** — Paragraph-level paraphrasing with controllable lexical diversity and content reordering knobs.

9. **Iterative Feedback Refinement** — Closed-loop system where humanized output is re-evaluated by the AI detector, with parameters adjusted iteratively until detection scores meet the target threshold.

---

### Methodology (in bullets)

- Collect and preprocess 18 large-scale NLP datasets spanning AI detection (RAID, HC3, M4, GPT-2 Output, FAIDSet, PAN Author ID), plagiarism detection (PAN Plagiarism Corpora, Clough & Stevenson, Webis, WikiSplit, STS Benchmark, PAWS), and humanization (ParaNMT-50M, PAWS, QQP, MRPC, BEA-2019 GEC, STS Benchmark).

- Convert all datasets into unified formats: classification (text + label), paraphrase (input + output), and similarity (text_a + text_b + score). Apply text cleaning, deduplication, and stratified 80/10/10 train/validation/test splits.

- Fine-tune four transformer classifiers for AI detection: DeBERTa-v3-large as primary detector, RoBERTa-large as ensemble member, Longformer-base for long documents (up to 4096 tokens), and XLM-RoBERTa-large for multilingual content.

- Train a logistic regression meta-classifier on the combined prediction outputs of all four models to produce an optimized ensemble AI detection score.

- Implement MinHash/LSH-based reference indexing for fast plagiarism candidate retrieval, followed by Sentence-BERT and SimCSE sentence-level embedding comparison.

- Fine-tune a DeBERTa-v3 cross-encoder for precise pairwise plagiarism verification on flagged sentence pairs.

- Fine-tune Flan-T5-XL, PEGASUS-large, and Mistral-7B (via QLoRA) on paraphrase datasets for content humanization. Configure DIPPER for paragraph-level paraphrasing with controllable diversity.

- Implement an iterative feedback loop: pass humanized output through the AI detector, and if the score exceeds the threshold, increase diversity/reordering parameters and regenerate until the target is met.

- Integrate all three modules into a unified pipeline with a CLI interface (argparse-based) supporting --detect, --plagiarism, --humanize, and --full flags.

- Build a web application using FastAPI (backend) and HTML/CSS/JS (frontend) for user-friendly text analysis and transformation.

- Evaluate the complete system using accuracy, F1-score, precision, recall, and AUROC metrics on held-out test sets, with benchmarking against existing commercial tools.

---

### Expected Outcomes

The expected outcomes are as follows:

1. An AI detection ensemble achieving 95%+ accuracy on standard benchmarks (RAID, HC3, M4) with robust performance across multiple languages, text lengths, and AI generators, while maintaining a false positive rate below 5%.

2. A semantic plagiarism detection system capable of identifying not only copy-paste plagiarism but also paraphrase-level and meaning-level plagiarism, evaluated on PAN benchmark corpora with competitive precision and recall scores.

3. A content humanization module that transforms AI-generated text to achieve near-zero (< 5%) AI detection scores on both the internal detector and external commercial tools, while preserving over 85% semantic similarity with the original content.

4. A fully integrated, self-contained platform with CLI and web interfaces that processes text through all three modules in under 30 seconds per document, operating entirely offline without external API dependencies.

---

### Literature Survey (Minimum 10 Papers)

| # | Paper Title | Authors | Year | Key Contribution |
|---|---|---|---|---|
| 1 | RAID: A Shared Benchmark for Robust Evaluation of Machine-Generated Text Detectors | Dugan, L., Hwang, A., Trhlik, F., et al. | 2024 | Introduced the largest AI text detection benchmark with 6M+ generations across 11 models, 8 domains, and 11 adversarial attacks. Published at ACL 2024. |
| 2 | M4: Multi-generator, Multi-domain, Multi-lingual Black-Box Machine-Generated Text Detection | Wang, Y., et al. (MBZUAI) | 2024 | Presented a multilingual, multi-generator dataset and detection framework covering multiple LLMs and languages for robust cross-lingual AI text detection. |
| 3 | HC3: Human ChatGPT Comparison Corpus | Guo, B., et al. (Hello-SimpleAI) | 2023 | Created paired human vs. ChatGPT answer datasets across multiple domains, enabling direct comparison of human and AI writing patterns. |
| 4 | On the Reliability of AI-Text Detectors | Sadasivan, V.S., et al. | 2023 | Demonstrated fundamental limitations of current AI text detectors, showing that paraphrasing attacks can significantly reduce detection accuracy. |
| 5 | Detecting Machine-Generated Text: A Critical Survey | Tang, R., et al. | 2023 | Provided a comprehensive survey of machine-generated text detection methods, categorizing approaches into statistical, neural, and watermarking-based techniques. |
| 6 | Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks | Reimers, N. & Gurevych, I. | 2019 | Proposed Siamese and triplet network structures for BERT to derive semantically meaningful sentence embeddings suitable for similarity comparison tasks. |
| 7 | SimCSE: Simple Contrastive Learning of Sentence Embeddings | Gao, T., Yao, X., & Chen, D. | 2021 | Introduced a contrastive learning framework that significantly improves sentence embedding quality using both unsupervised and supervised approaches. |
| 8 | Paraphrasing Evades Detectors of AI-Generated Text, but Retrieval is an Effective Defense (DIPPER) | Krishna, K., et al. | 2023 | Demonstrated that discourse-level paraphrasing can evade AI detectors and proposed retrieval-based detection as a countermeasure. Introduced the DIPPER paraphraser. |
| 9 | DeBERTa: Decoding-enhanced BERT with Disentangled Attention | He, P., Liu, X., Gao, J., & Chen, W. | 2021 | Proposed disentangled attention mechanism and enhanced mask decoder that achieved state-of-the-art results on multiple NLU benchmarks. |
| 10 | Fine-tuned LLMs for Multilingual Machine-Generated Text Detection (SemEval-2024 Task 8) | Hlavnova, E. & Pikuliak, M. | 2024 | Explored fine-tuning strategies for multilingual AI text detection, showing that LoRA-adapted RoBERTa with majority voting is effective in multilingual contexts. |
| 11 | Fine-grained AI-generated Text Detection using Multi-task Auxiliary and Multi-level Contrastive Learning (FAIDSet) | Multiple authors | 2025 | Introduced a fine-grained detection framework classifying text as fully human, fully AI, or mixed, with a large-scale multilingual dataset. |
| 12 | A Survey on Plagiarism Detection | Foltýnek, T., Meuschke, N., & Gipp, B. | 2019 | Comprehensive survey of plagiarism detection methods covering string matching, citation analysis, and semantic approaches. |
| 13 | Authorship Style Transfer with Policy Optimization | Multiple authors | 2024 | Proposed policy optimization techniques for authorship style transfer, relevant to understanding how writing style can be transformed while preserving content. |

---


Signature of the Guide                                          Project Coordinator
