# Content Integrity & Authorship Intelligence Platform

## Project Blueprint — Complete Reference Document

---

## 1. The Real-World Problem

The internet is flooded with AI-generated content. Students submit AI-written essays. Professionals publish AI-drafted articles. Researchers pad papers with machine-generated paragraphs. Content mills produce thousands of AI blog posts daily. Meanwhile, plagiarism — copying from existing human sources — remains as widespread as ever.

The tools that exist today to deal with this are broken in three specific ways:

**AI Detection is unreliable.** Tools like GPTZero, Originality.ai, and Turnitin's AI detector produce inconsistent results. They flag genuine human writing as AI-generated (false positives), miss sophisticated AI output (false negatives), and penalize non-native English speakers disproportionately. They operate as black boxes — no one can see how they reach their verdict. According to the RAID benchmark (ACL 2024), the best commercial detector achieved only 85% accuracy on base AI text, and accuracy drops sharply when adversarial techniques are applied.

**Plagiarism detection hasn't evolved.** Traditional plagiarism checkers rely on string matching and n-gram comparison against proprietary databases. They catch copy-paste plagiarism but struggle with paraphrase plagiarism — where someone rewrites a source just enough to avoid detection. Semantic plagiarism (same ideas, completely different words) is largely invisible to current tools.

**There is no unified system.** AI detection and plagiarism detection are treated as separate problems by separate tools. A user must run content through multiple services, pay multiple subscriptions, and reconcile conflicting results. No single platform answers all three questions simultaneously:
- Was this written by AI?
- Was this copied from an existing source?
- Can this content be transformed into genuinely original, human-quality writing?

This fragmentation creates confusion, mistrust, and a lack of accountability in how written content is evaluated.

---

## 2. The Vision — What This Project Solves

This project builds a single, self-contained content intelligence platform with three integrated modules:

| Module | What It Does |
|--------|-------------|
| **AI Detection** | Analyzes any text input and determines the percentage of AI-generated content. The goal is to detect AI involvement with the highest possible accuracy across all text formats, lengths, languages, and AI models. |
| **Plagiarism Detection** | Compares input text against reference sources to identify copied, paraphrased, or semantically duplicated content. Goes beyond surface-level string matching to catch meaning-level plagiarism. |
| **Content Humanization** | Takes AI-generated or flagged content and transforms it into naturally human-written text. The output should read as authentically human — with natural variation, imperfect structure, genuine reasoning flow, and stylistic personality. The transformed content should register as 0% AI on detection systems. |

**The key insight:** These three modules share a common semantic understanding layer. The same technology that understands text deeply enough to detect AI can also understand it deeply enough to detect plagiarism and to rewrite it meaningfully. By building all three on a shared foundation, the system is more accurate, more efficient, and more consistent than running three separate tools.

**What makes this different from existing tools:**
- It is transparent — not a black box
- It is self-built and self-trained — no dependency on third-party APIs
- It combines detection and transformation in one pipeline
- It works across text formats, lengths, and languages
- It is designed to run locally, giving the user full control

---

## 3. Project Constraints — What You Must Do to Build This

### Hardware Requirements
- A machine with a dedicated GPU (minimum 8GB VRAM for fine-tuning smaller models, 24GB+ recommended for larger models like DIPPER or Mistral-7B)
- At least 64GB RAM for dataset processing
- 500GB+ free storage for datasets, model weights, and checkpoints
- If local hardware is insufficient, use cloud GPU instances (Google Colab Pro, AWS, RunPod, or Lambda Labs)

### Software Requirements
- Python 3.10+
- PyTorch 2.0+ with CUDA support
- Hugging Face Transformers library
- Hugging Face Datasets library
- Sentence-Transformers library
- Datasketch library (for MinHash/LSH in plagiarism module)
- scikit-learn (for evaluation metrics and meta-classifier)
- NLTK or spaCy (for text preprocessing)
- Weights & Biases or TensorBoard (for tracking training runs)

### Knowledge Prerequisites
- Solid understanding of Python
- Familiarity with transformer architecture (attention, tokenization, fine-tuning)
- Understanding of classification vs. sequence-to-sequence tasks
- Basic knowledge of training loops, loss functions, learning rate scheduling
- Understanding of evaluation metrics: accuracy, precision, recall, F1, AUROC

### Development Constraints
- All models must be trained locally or on your own cloud instances — no API calls to OpenAI, Google, or Anthropic for inference
- All datasets must be downloaded and stored locally
- The system must work offline after training is complete
- Build and test in terminal first before any UI/app development
- Each module should be independently testable before integration

### Ethical Constraints (to be enforced after project completion)
- Rate limiting to prevent bulk abuse
- Logging and audit trails for transformation requests
- Content length limits per request
- Terms of service prohibiting academic fraud
- Optional: require users to demonstrate comprehension before transformation (quiz, summary, or explanation step)

---

## 4. Datasets — All Three Divisions

### Division 1: AI Detection Datasets

| # | Dataset | Description | Size | Source |
|---|---------|-------------|------|--------|
| 1 | **RAID** | The largest benchmark for AI text detection. Covers 11 LLMs, 11 genres, 12 adversarial attacks, 4 decoding strategies. Contains 10M+ documents with both human and machine-generated text. | 10M+ documents | [github.com/liamdugan/raid](https://github.com/liamdugan/raid) |
| 2 | **HC3** (Human-ChatGPT Comparison Corpus) | Paired dataset of human answers vs. ChatGPT answers on identical prompts. Covers multiple domains including finance, medicine, open QA, and Wikipedia. | ~40K QA pairs | [huggingface.co/datasets/Hello-SimpleAI/HC3](https://huggingface.co/datasets/Hello-SimpleAI/HC3) |
| 3 | **M4** (Multi-generator, Multi-domain, Multi-lingual) | Covers text from multiple LLMs (ChatGPT, LLaMA, etc.), multiple domains, and multiple languages. Critical for building a detector that generalizes. | Multi-million samples | [github.com/mbzuai-nlp/M4](https://github.com/mbzuai-nlp/M4) |
| 4 | **OpenAI GPT-2 Output Dataset** | 250K human-written samples + 250K generated samples per GPT-2 model size. Includes top-k sampling variants. Older but useful as baseline training data. | 1.5M+ samples | [github.com/openai/gpt-2-output-dataset](https://github.com/openai/gpt-2-output-dataset) |
| 5 | **FAIDSet** (2025) | Multilingual, multi-domain, multi-generator dataset supporting fine-grained detection — classifies text as fully human, fully AI, or mixed. | Large-scale | [arxiv.org/abs/2505.14271](https://arxiv.org/abs/2505.14271) |
| 6 | **PAN Author Identification Corpora** | Stylometry and authorship verification datasets spanning multiple years of shared tasks. Essential for understanding human writing patterns and behavior. | Varies by year | [pan.webis.de/data.html](https://pan.webis.de/data.html) |

### Division 2: Plagiarism Detection Datasets

| # | Dataset | Description | Size | Source |
|---|---------|-------------|------|--------|
| 1 | **PAN Plagiarism Detection Corpora** (2009–2015) | The gold standard for plagiarism research. Includes copy-paste plagiarism, paraphrase plagiarism, cross-lingual plagiarism, and source retrieval tasks with full annotations. | Thousands of annotated document pairs | [pan.webis.de/clef.html](https://pan.webis.de/clef.html) |
| 2 | **Clough & Stevenson Plagiarism Corpus** | Academic plagiarism cases annotated at four levels: near-copy, light revision, heavy revision, and non-plagiarism. Small but high-quality. | ~100 documents | Search: "Clough Stevenson 2011 plagiarism corpus" |
| 3 | **Webis Crowd Paraphrase Corpus 2011** | Crowdsourced paraphrases with plagiarism annotations. Directly targets the paraphrase-plagiarism detection problem. | ~4K pairs | [webis.de/data](https://webis.de/data.html) |
| 4 | **WikiSplit** | 1 million sentence splits from Wikipedia. Useful for detecting when someone restructures a source by splitting or merging sentences to disguise copying. | 1M sentence pairs | [github.com/google-research-datasets/wiki-split](https://github.com/google-research-datasets/wiki-split) |
| 5 | **STS Benchmark** *(shared with Humanization)* | Semantic textual similarity scores on a continuous 0–5 scale. Used to define the threshold: "how similar is too similar?" | ~8.6K pairs | [ixa2.si.ehu.eus/stswiki](https://ixa2.si.ehu.eus/stswiki/index.php/STSbenchmark) |
| 6 | **PAWS** *(shared with Humanization)* | Adversarial paraphrase pairs that look different on the surface but mean the same thing (and vice versa). Trains the system to catch disguised copying. | ~108K pairs | [github.com/google-research-datasets/paws](https://github.com/google-research-datasets/paws) |

### Division 3: Humanization (Content Transformation) Datasets

| # | Dataset | Description | Size | Source |
|---|---------|-------------|------|--------|
| 1 | **ParaNMT-50M** | 50 million paraphrase pairs generated via back-translation. Massive scale for pretraining a paraphrase/rewriting model. | 50M pairs | [github.com/jwieting/para-nmt-50m](https://github.com/jwieting/para-nmt-50m) |
| 2 | **PAWS** *(shared with Plagiarism)* | Adversarial paraphrase pairs. Teaches the rewriter to make non-trivial transformations, not just synonym swaps. | ~108K pairs | [github.com/google-research-datasets/paws](https://github.com/google-research-datasets/paws) |
| 3 | **QQP** (Quora Question Pairs) | 400K+ sentence pairs labeled for semantic equivalence. Trains the model to understand meaning-preserving transformations. | 400K+ pairs | [kaggle.com/c/quora-question-pairs](https://www.kaggle.com/c/quora-question-pairs) |
| 4 | **STS Benchmark** *(shared with Plagiarism)* | Continuous similarity scores. Used to evaluate whether the rewriter preserves the original meaning after transformation. | ~8.6K pairs | [ixa2.si.ehu.eus/stswiki](https://ixa2.si.ehu.eus/stswiki/index.php/STSbenchmark) |
| 5 | **MRPC** (Microsoft Research Paraphrase Corpus) | High-quality paraphrase pairs from news articles. Good for fine-tuning on formal/professional text rewriting. | ~5.8K pairs | [microsoft.com/download](https://www.microsoft.com/en-us/download/details.aspx?id=52398) |
| 6 | **BEA-2019 GEC Dataset** | Grammatical error correction data. Used to teach the rewriter to introduce natural human imperfections — because real humans make small grammatical variations that AI does not. | Large-scale | [cl.cam.ac.uk/research/nl/bea2019st](https://www.cl.cam.ac.uk/research/nl/bea2019st/) |

---

## 5. Models — All Three Divisions

### Division 1: AI Detection Models

| # | Model | Role | Parameters | Source |
|---|-------|------|-----------|--------|
| 1 | **DeBERTa-v3-large** | Primary classifier backbone. Best-in-class for text classification tasks. Fine-tune as the main AI detection head. | 304M | [huggingface.co/microsoft/deberta-v3-large](https://huggingface.co/microsoft/deberta-v3-large) |
| 2 | **RoBERTa-large** | Secondary classifier. OpenAI's own GPT-2 detector was built on this. Use as an ensemble member alongside DeBERTa. | 355M | [huggingface.co/roberta-large](https://huggingface.co/roberta-large) |
| 3 | **Longformer-base** | Long document classifier. Handles up to 4,096 tokens. Essential for essays, articles, and research papers that exceed BERT's 512-token limit. | 149M | [huggingface.co/allenai/longformer-base-4096](https://huggingface.co/allenai/longformer-base-4096) |
| 4 | **XLM-RoBERTa-large** | Multilingual detection. Supports 100+ languages. Required if the system must detect AI text in non-English content. | 560M | [huggingface.co/xlm-roberta-large](https://huggingface.co/xlm-roberta-large) |

### Division 2: Plagiarism Detection Models

| # | Model | Role | Parameters | Source |
|---|-------|------|-----------|--------|
| 1 | **Sentence-BERT (all-mpnet-base-v2)** | Sentence embedding engine. Converts text into vectors. High cosine similarity between two vectors = potential plagiarism. Primary similarity engine. | 109M | [huggingface.co/sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) |
| 2 | **DeBERTa-v3 Cross-Encoder** *(shared with AI Detection)* | Takes two text passages as input and directly scores their similarity. More accurate than embedding comparison. Used as the verification step after initial screening. | 304M | [huggingface.co/cross-encoder/nli-deberta-v3-large](https://huggingface.co/cross-encoder/nli-deberta-v3-large) |
| 3 | **SimCSE** | Contrastive learning framework for sentence embeddings. Produces better similarity representations than vanilla Sentence-BERT. Fine-tune on your paraphrase data. | ~110M | [github.com/princeton-nlp/SimCSE](https://github.com/princeton-nlp/SimCSE) |
| 4 | **Longformer** *(shared with AI Detection)* | Document-level comparison for long texts. Compares full essays or papers, not just individual sentences. | 149M | [huggingface.co/allenai/longformer-base-4096](https://huggingface.co/allenai/longformer-base-4096) |
| 5 | **MinHash / LSH** (algorithmic, not neural) | Fast approximate duplicate detection. First-pass filter that quickly identifies candidate matches before running expensive neural comparisons. | N/A | [github.com/ekzhu/datasketch](https://github.com/ekzhu/datasketch) |

### Division 3: Humanization (Content Transformation) Models

| # | Model | Role | Parameters | Source |
|---|-------|------|-----------|--------|
| 1 | **DIPPER** (Discourse Paraphraser) | Purpose-built paragraph-level paraphraser with control knobs for lexical diversity and content reordering. The most directly relevant model for humanization. | 11B | [huggingface.co/kalpeshk2011/dipper-paraphraser-xxl](https://huggingface.co/kalpeshk2011/dipper-paraphraser-xxl) |
| 2 | **Flan-T5-XL** | Encoder-decoder model excellent for controlled text generation. Fine-tune on paraphrase datasets for high-quality rewriting. | 3B | [huggingface.co/google/flan-t5-xl](https://huggingface.co/google/flan-t5-xl) |
| 3 | **PEGASUS-large** | Designed for abstractive summarization — essentially rewriting with compression. Learns to restructure ideas, not just swap words. | 568M | [huggingface.co/google/pegasus-large](https://huggingface.co/google/pegasus-large) |
| 4 | **Mistral-7B** | Open-weight generative model. Fine-tune with LoRA/QLoRA on paraphrase data for a powerful local rewriter. Best option if you want a single large model for humanization. | 7B | [huggingface.co/mistralai/Mistral-7B-v0.3](https://huggingface.co/mistralai/Mistral-7B-v0.3) |

---

## 6. Dataset-to-Model Mapping — What Trains What

This table shows exactly which datasets feed into which models.

| Model | Trained On (Datasets) | Task |
|-------|----------------------|------|
| DeBERTa-v3-large | RAID, HC3, M4, FAIDSet | Binary/fine-grained AI detection classification |
| RoBERTa-large | RAID, HC3, OpenAI GPT-2 Output Dataset | AI detection (ensemble member) |
| Longformer-base | RAID (long documents), PAN Plagiarism Corpora | Long-document AI detection + plagiarism comparison |
| XLM-RoBERTa-large | M4 (multilingual split), FAIDSet | Multilingual AI detection |
| Sentence-BERT | STS Benchmark, PAWS, QQP | Sentence embedding for plagiarism similarity scoring |
| SimCSE | STS Benchmark, PAWS, ParaNMT-50M (subset) | Improved sentence embeddings for plagiarism |
| DeBERTa-v3 Cross-Encoder | STS Benchmark, PAWS, MRPC, PAN Plagiarism Corpora | Pairwise plagiarism verification |
| MinHash / LSH | PAN Plagiarism Corpora, WikiSplit | Fast first-pass duplicate screening (no training — algorithmic) |
| DIPPER | ParaNMT-50M, PAWS | Paragraph-level paraphrasing for humanization |
| Flan-T5-XL | ParaNMT-50M, QQP, MRPC, BEA-2019 GEC | Controlled rewriting and humanization |
| PEGASUS-large | ParaNMT-50M (subset), MRPC | Abstractive rewriting and restructuring |
| Mistral-7B (LoRA) | ParaNMT-50M, PAWS, QQP, BEA-2019 GEC | Full humanization with style variation |

---

## 7. How to Use These Models with These Datasets

### AI Detection Module — How the pieces connect

The AI detection models are all **classification models**. They take a piece of text as input and output a label: "human" or "AI" (or a probability score between 0 and 1).

**Step 1: Prepare the data.** Each dataset (RAID, HC3, M4, etc.) contains text samples labeled as human-written or AI-generated. Combine them into a unified format: `{"text": "...", "label": 0 or 1}` where 0 = human and 1 = AI. Split into training (80%), validation (10%), and test (10%) sets.

**Step 2: Tokenize.** Each model has its own tokenizer. Run the text through the model's tokenizer to convert words into token IDs. For DeBERTa and RoBERTa, the maximum input length is 512 tokens — truncate or chunk longer texts. For Longformer, the limit is 4,096 tokens.

**Step 3: Fine-tune.** Load the pretrained model, add a classification head (a single linear layer on top), and train it on your labeled data. The model learns to distinguish AI patterns from human patterns in the text.

**Step 4: Ensemble.** After training each model separately, combine their predictions. For any input text, run it through all trained detectors and use a meta-classifier (a simple logistic regression or weighted vote) to produce the final AI probability score.

### Plagiarism Detection Module — How the pieces connect

Plagiarism detection is a **similarity problem**, not a classification problem. You are comparing an input document against a collection of reference documents.

**Step 1: Build a reference index.** Take your reference corpus (PAN documents, or any collection of source texts). Run each document through MinHash to create a fingerprint. Store these fingerprints in an LSH index for fast lookup.

**Step 2: First-pass screening.** When a new document comes in, compute its MinHash fingerprint and query the LSH index. This returns candidate documents that are approximately similar — fast but rough.

**Step 3: Sentence-level comparison.** For each candidate match, break both documents into sentences. Encode each sentence using Sentence-BERT or SimCSE. Compute cosine similarity between all sentence pairs. Flag pairs above a threshold (e.g., 0.85) as potential plagiarism.

**Step 4: Verification.** For flagged sentence pairs, run them through the DeBERTa cross-encoder for precise similarity scoring. This catches paraphrase plagiarism that embedding comparison might miss.

**Step 5: Report.** Output a plagiarism report showing which sentences match, what sources they match against, and the similarity scores.

### Humanization Module — How the pieces connect

Humanization is a **sequence-to-sequence generation problem**. The model takes AI-generated text as input and produces rewritten human-like text as output.

**Step 1: Prepare parallel data.** From your paraphrase datasets (ParaNMT, PAWS, QQP), create input-output pairs where the input is one version and the output is the paraphrased version. Additionally, generate AI text using any available LLM, then pair it with human-written equivalents from HC3.

**Step 2: Fine-tune the rewriter.** Load DIPPER, Flan-T5, or Mistral-7B. Fine-tune on the parallel data so the model learns to transform text while preserving meaning. For Mistral-7B, use LoRA (Low-Rank Adaptation) to make fine-tuning feasible on consumer hardware.

**Step 3: Add quality signals.** Use the BEA-2019 GEC data to teach the model that human text contains natural imperfections. Use STS Benchmark to evaluate that the rewritten text still means the same thing as the original.

**Step 4: Feedback loop.** After the rewriter produces output, run that output through your own AI detection module (Module 1). If it still scores as AI-generated, feed it back through the rewriter with adjusted parameters (higher lexical diversity, more reordering). Repeat until the detection score drops below your target threshold.

---

## 8. How to Train — Step-by-Step Process

Assuming you have downloaded all datasets and have your code ready:

### Phase 1: Data Preparation (1–2 weeks)

1. Download all datasets listed in Section 4 to a local directory structure
2. Write preprocessing scripts to convert each dataset into a unified JSON Lines format
3. For classification datasets: `{"text": "...", "label": 0/1}`
4. For paraphrase datasets: `{"input": "...", "output": "..."}`
5. For similarity datasets: `{"text_a": "...", "text_b": "...", "score": 0.0–1.0}`
6. Clean the data: remove duplicates, handle encoding issues, filter out extremely short or long samples
7. Create train/validation/test splits (80/10/10) with stratification

### Phase 2: Train AI Detection Models (2–3 weeks)

1. Start with DeBERTa-v3-large — this is your primary detector
2. Load the pretrained model from Hugging Face
3. Add a binary classification head
4. Training configuration:
   - Learning rate: 2e-5 (start here, adjust based on validation loss)
   - Batch size: 16 (or 8 with gradient accumulation if GPU memory is limited)
   - Epochs: 3–5 (transformers overfit quickly — monitor validation loss)
   - Optimizer: AdamW with weight decay 0.01
   - Scheduler: Linear warmup for 10% of steps, then linear decay
5. Train on the combined RAID + HC3 + M4 dataset
6. Evaluate on the held-out test set — measure accuracy, F1, precision, recall, and AUROC
7. Repeat for RoBERTa-large and Longformer with the same process
8. Train the meta-classifier (logistic regression) on the combined predictions of all three models

**Algorithm — Ensemble Meta-Classifier:**
```
For each input text T:
    score_1 = DeBERTa(T)        → probability of AI
    score_2 = RoBERTa(T)        → probability of AI
    score_3 = Longformer(T)     → probability of AI (for long texts)
    
    final_score = MetaClassifier(score_1, score_2, score_3)
    
    if final_score > threshold:
        label = "AI-generated"
    else:
        label = "Human-written"
```

### Phase 3: Train Plagiarism Detection Models (1–2 weeks)

1. Fine-tune Sentence-BERT on STS Benchmark + PAWS using contrastive loss
2. Fine-tune SimCSE on the same data using the SimCSE training procedure (unsupervised + supervised)
3. Fine-tune the DeBERTa cross-encoder on PAWS + MRPC + PAN pairs as a pairwise classifier
4. Build the MinHash/LSH index from PAN Plagiarism Corpora — this requires no training, just implementation
5. Evaluate on PAN test sets using the standard PAN evaluation metrics (precision, recall at character level)

### Phase 4: Train Humanization Models (2–4 weeks)

1. Start with Flan-T5-XL — it is the most straightforward to fine-tune for seq2seq tasks
2. Prepare input-output pairs from ParaNMT-50M (sample a manageable subset — 1M to 5M pairs)
3. Fine-tune with:
   - Learning rate: 1e-4
   - Batch size: 8 (with gradient accumulation)
   - Epochs: 2–3
   - Max input length: 512 tokens
   - Max output length: 512 tokens
4. For Mistral-7B, use QLoRA:
   - LoRA rank: 16–64
   - LoRA alpha: 32–128
   - Target modules: all attention layers
   - 4-bit quantization to fit in consumer GPU memory
5. For DIPPER, the model is already pretrained for paraphrasing — fine-tune lightly on your specific data or use it as-is with its control knobs (lexical diversity: 0–100, order diversity: 0–100)
6. After training, run the output through your AI detection module and measure the detection rate. Iterate.

**Algorithm — Humanization with Feedback Loop:**
```
For each AI-generated input text T:
    rewritten = Humanizer(T, diversity=60, reorder=40)
    ai_score = AIDetector(rewritten)
    
    while ai_score > target_threshold:
        diversity += 10
        reorder += 10
        rewritten = Humanizer(T, diversity, reorder)
        ai_score = AIDetector(rewritten)
        
        if diversity > 100:
            break  → maximum transformation reached
    
    output = rewritten
```

### Phase 5: Integration (1 week)

1. Connect all three modules into a single pipeline
2. Input text flows through: AI Detection → Plagiarism Check → (if needed) Humanization
3. Output: a comprehensive report with AI score, plagiarism matches, and optionally the humanized version

---

## 9. How to Execute the Project

### Stage 1: Terminal Execution (First milestone)

Once all models are trained and saved, the project runs as a command-line tool.

**How it works in the terminal:**

The user runs a Python script from the terminal (VSCode terminal, macOS Terminal, or any terminal). They provide input text either as a string argument, a file path, or piped from stdin.

The system processes the input through all three modules and prints results to the terminal:

```
$ python main.py --input "paste your text here"

═══════════════════════════════════════════════
  CONTENT ANALYSIS REPORT
═══════════════════════════════════════════════

  AI Detection Score:        87.3% AI-generated
  Plagiarism Score:          12.1% matched content
  Sources Found:             2 matches

  Matched Sources:
    1. [Source A] — 8.4% similarity (sentences 3, 7, 12)
    2. [Source B] — 3.7% similarity (sentence 19)

═══════════════════════════════════════════════

$ python main.py --input "paste your text here" --humanize

  Humanized Output:
  [transformed text appears here]

  Post-Humanization AI Score: 2.1%

═══════════════════════════════════════════════
```

**What to validate in terminal stage:**
- AI detection accuracy on your test set (target: 95%+ on clean text)
- Plagiarism detection precision and recall on PAN test data
- Humanization quality: run output through your own detector AND through external detectors (GPTZero, Originality.ai) to verify
- Processing speed: measure time per document
- Memory usage: ensure it runs within your hardware limits

**Terminal stage is complete when:**
- All three modules produce consistent, reliable results
- The pipeline handles edge cases (very short text, very long text, mixed content, non-English text)
- Processing time is acceptable (under 30 seconds for a typical essay)

### Stage 2: Application Development (After terminal works perfectly)

Once the terminal version is stable and validated, move to building a user-facing application.

**Recommended progression:**

1. **Web Application (recommended first)**
   - Build a simple web interface using Flask or FastAPI (backend) + React or plain HTML/CSS/JS (frontend)
   - User pastes text into a text box, clicks "Analyze," and sees the report
   - Add a "Humanize" button that triggers the transformation module
   - This is the fastest path to a usable product

2. **Desktop Application (optional)**
   - Package the web app as a desktop app using Electron or Tauri
   - Runs locally, no internet required after installation
   - Good for users who want privacy and offline access

3. **API Service (for integration)**
   - Wrap the pipeline in a REST API using FastAPI
   - Other applications can send text and receive analysis results
   - This is the path toward a SaaS product

4. **Mobile Application (later stage)**
   - Build a mobile frontend that connects to your API backend
   - React Native or Flutter for cross-platform
   - Only pursue this after the web version is stable

---

## 10. Complete Project Summary

| Aspect | Detail |
|--------|--------|
| **Project Name** | Content Integrity & Authorship Intelligence Platform |
| **Core Problem** | AI-generated content is undetectable, plagiarism tools are outdated, and no unified system exists to handle both detection and transformation |
| **Solution** | A three-module system: AI Detection + Plagiarism Detection + Content Humanization, built on a shared semantic backbone |
| **Total Datasets** | 18 datasets across three divisions (6 for AI detection, 6 for plagiarism, 6 for humanization — with some shared) |
| **Total Models** | 13 models across three divisions (4 for AI detection, 5 for plagiarism, 4 for humanization — with some shared) |
| **Training Time Estimate** | 6–10 weeks total (data prep + training all modules + integration) |
| **First Deliverable** | Terminal-based CLI tool that analyzes and transforms text |
| **Second Deliverable** | Web application with a user interface |
| **Hardware Needed** | GPU with 24GB+ VRAM (or cloud GPU), 64GB RAM, 500GB storage |
| **Key Differentiator** | Self-built, self-trained, transparent, offline-capable, combines detection and transformation in one system |

---

## 11. Research Papers — Complete Reference List

### AI Detection Papers
1. "RAID: A Shared Benchmark for Robust Evaluation of Machine-Generated Text Detectors" — Dugan et al., ACL 2024 → [arxiv.org/abs/2405.07940](https://arxiv.org/abs/2405.07940)
2. "On the Reliability of AI-Text Detectors" — 2023 → [arxiv.org/abs/2304.02819](https://arxiv.org/abs/2304.02819)
3. "Detecting Machine-Generated Text: A Critical Survey" — 2023 → [arxiv.org/abs/2303.07205](https://arxiv.org/abs/2303.07205)
4. "M4: Multi-generator, Multi-domain, Multi-lingual Black-Box MGT Detection" — Wang et al., 2024 → [arxiv.org/abs/2305.14902](https://arxiv.org/abs/2305.14902)
5. "Fine-grained AI-generated Text Detection using Multi-task Auxiliary and Multi-level Contrastive Learning" — 2025 → [arxiv.org/abs/2505.14271](https://arxiv.org/abs/2505.14271)

### Plagiarism Detection Papers
6. "A Survey on Plagiarism Detection" — Foltýnek et al., 2019 → [arxiv.org/abs/1703.05546](https://arxiv.org/abs/1703.05546)
7. "Semantic Plagiarism Detection Using Transformer Models" — 2024 → [ceur-ws.org/Vol-4038/paper_324.pdf](https://ceur-ws.org/Vol-4038/paper_324.pdf)
8. "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks" — Reimers & Gurevych, 2019 → [arxiv.org/abs/1908.10084](https://arxiv.org/abs/1908.10084)
9. "SimCSE: Simple Contrastive Learning of Sentence Embeddings" — Gao et al., 2021 → [arxiv.org/abs/2104.08821](https://arxiv.org/abs/2104.08821)

### Humanization & Paraphrasing Papers
10. "Paraphrasing evades detectors of AI-generated text, but retrieval is an effective defense" (DIPPER paper) — Krishna et al., 2023 → [arxiv.org/abs/2303.13408](https://arxiv.org/abs/2303.13408)
11. "Authorship Style Transfer with Policy Optimization" — 2024 → [arxiv.org/abs/2403.08043](https://arxiv.org/abs/2403.08043)
12. "Paraphrase Generation: A Survey" — 2022 → [arxiv.org/abs/2206.05233](https://arxiv.org/abs/2206.05233)
13. "Distilling Text Style Transfer with Self-Explanation from LLMs" — 2024 → [arxiv.org/abs/2403.08043](https://arxiv.org/abs/2403.08043)

---

*This document serves as the complete project blueprint. All datasets, models, training procedures, execution steps, and references are contained within this single document. No external dependencies or API calls are required — everything is built, trained, and run locally.*
