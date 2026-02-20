"""
Person 1 — Train Meta-Classifier (Ensemble Combiner)
Logistic regression that combines predictions from all 4 detection models
into a single final AI probability score.
"""

import json
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import joblib

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader as TorchDataLoader

from config import CHECKPOINT_DIR, META_CLASSIFIER_PATH, MODELS, SEED
from data_loader import DataLoader
from train_utils import set_seed, get_device


# Model checkpoint names
MODEL_CHECKPOINTS = [
    ("deberta_ai_detector", MODELS["deberta"]["max_length"]),
    ("roberta_ai_detector", MODELS["roberta"]["max_length"]),
    ("longformer_ai_detector", MODELS["longformer"]["max_length"]),
    ("xlm_roberta_ai_detector", MODELS["xlm_roberta"]["max_length"]),
]


def get_model_predictions(
    checkpoint_name: str,
    max_length: int,
    texts: list[str],
    batch_size: int = 32,
) -> np.ndarray:
    """Run inference with a single model and return probability scores."""
    device = get_device()
    ckpt_path = CHECKPOINT_DIR / checkpoint_name

    if not ckpt_path.exists():
        print(f"  [WARN] Checkpoint not found: {ckpt_path}. Returning zeros.")
        return np.zeros(len(texts))

    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt_path)
    model.to(device)
    model.eval()

    all_probs = []
    for i in tqdm(range(0, len(texts), batch_size), desc=f"  {checkpoint_name}"):
        batch_texts = texts[i : i + batch_size]
        encoding = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoding = {k: v.to(device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = model(**encoding)
            probs = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()
        all_probs.extend(probs)

    # Free GPU memory
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return np.array(all_probs)


def main():
    print("=" * 60)
    print("  Training Meta-Classifier (Ensemble Combiner)")
    print("=" * 60)

    set_seed()
    loader = DataLoader()

    # Load validation data (use val set to train meta-classifier, test set for final eval)
    print("\nLoading data for meta-classifier training...")
    val_records = loader.load_combined(
        ["raid", "hc3", "m4"], split="val", max_per_dataset=10_000
    )
    test_records = loader.load_combined(
        ["raid", "hc3", "m4"], split="test", max_per_dataset=10_000
    )

    val_texts = [r["text"] for r in val_records]
    val_labels = np.array([r["label"] for r in val_records])
    test_texts = [r["text"] for r in test_records]
    test_labels = np.array([r["label"] for r in test_records])

    print(f"Meta-train samples (from val): {len(val_texts):,}")
    print(f"Meta-test samples:             {len(test_texts):,}")

    # Collect predictions from each model
    print("\nCollecting predictions from all models...")
    val_features = []
    test_features = []

    for ckpt_name, max_len in MODEL_CHECKPOINTS:
        print(f"\n  Model: {ckpt_name}")
        val_preds = get_model_predictions(ckpt_name, max_len, val_texts)
        test_preds = get_model_predictions(ckpt_name, max_len, test_texts)
        val_features.append(val_preds)
        test_features.append(test_preds)

    # Stack into feature matrices: (n_samples, n_models)
    X_train = np.column_stack(val_features)
    X_test = np.column_stack(test_features)

    # Train logistic regression meta-classifier
    print("\nTraining logistic regression meta-classifier...")
    meta_clf = LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=SEED,
        solver="lbfgs",
    )
    meta_clf.fit(X_train, val_labels)

    # Evaluate on test set
    test_probs = meta_clf.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(test_labels, test_preds),
        "f1": f1_score(test_labels, test_preds),
        "auroc": roc_auc_score(test_labels, test_probs),
    }

    print(f"\n  Meta-Classifier Test Results:")
    for k, v in metrics.items():
        print(f"    {k}: {v:.4f}")

    # Print model weights (feature importances)
    print(f"\n  Model weights (coefficients):")
    for (ckpt_name, _), coef in zip(MODEL_CHECKPOINTS, meta_clf.coef_[0]):
        print(f"    {ckpt_name}: {coef:.4f}")

    # Save
    joblib.dump(meta_clf, META_CLASSIFIER_PATH)
    print(f"\n  Saved meta-classifier to {META_CLASSIFIER_PATH}")

    # Save metrics
    metrics_path = CHECKPOINT_DIR / "meta_classifier_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved metrics to {metrics_path}")

    print(f"\n{'=' * 60}")
    print(f"  Meta-Classifier Training Complete")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
