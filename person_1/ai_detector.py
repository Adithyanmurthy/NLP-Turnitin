"""
Person 1 — AI Detector Module
The main deliverable. Provides: detect(text) → float (0.0 = human, 1.0 = AI)

Usage:
    from person_1.ai_detector import AIDetector

    detector = AIDetector()
    score = detector.detect("Some text to check")
    print(f"AI probability: {score:.2%}")

    # Batch detection
    scores = detector.detect_batch(["text1", "text2", "text3"])

    # Get detailed per-model scores
    details = detector.detect_detailed("Some text")
    # → {"deberta": 0.87, "roberta": 0.91, "longformer": 0.82, "xlm_roberta": 0.79, "ensemble": 0.86}
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import joblib
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import CHECKPOINT_DIR, META_CLASSIFIER_PATH, MODELS, AI_THRESHOLD
from train_utils import get_device


class AIDetector:
    """
    AI Detection Ensemble.
    Loads all 4 fine-tuned models + meta-classifier and produces
    a single AI probability score for any input text.
    """

    MODEL_CONFIGS = {
        "deberta": {
            "checkpoint": "deberta_ai_detector",
            "max_length": MODELS["deberta"]["max_length"],
        },
        "roberta": {
            "checkpoint": "roberta_ai_detector",
            "max_length": MODELS["roberta"]["max_length"],
        },
        "longformer": {
            "checkpoint": "longformer_ai_detector",
            "max_length": MODELS["longformer"]["max_length"],
        },
        "xlm_roberta": {
            "checkpoint": "xlm_roberta_ai_detector",
            "max_length": MODELS["xlm_roberta"]["max_length"],
        },
    }

    def __init__(
        self,
        checkpoint_dir: Optional[Path] = None,
        device: Optional[torch.device] = None,
        load_all: bool = True,
    ):
        """
        Initialize the AI Detector.

        Args:
            checkpoint_dir: Path to model checkpoints. Defaults to config.
            device: Torch device. Auto-detected if None.
            load_all: If True, load all models on init. If False, load lazily.
        """
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else CHECKPOINT_DIR
        self.device = device or get_device()
        self.models = {}
        self.tokenizers = {}
        self.meta_classifier = None

        if load_all:
            self._load_all_models()

    def _load_all_models(self):
        """Load all model checkpoints and the meta-classifier."""
        print("Loading AI Detection Ensemble...")

        for name, cfg in self.MODEL_CONFIGS.items():
            self._load_model(name)

        # Load meta-classifier
        meta_path = self.checkpoint_dir / "meta_classifier.joblib"
        if meta_path.exists():
            self.meta_classifier = joblib.load(meta_path)
            print(f"  Loaded meta-classifier from {meta_path}")
        else:
            print(f"  [WARN] Meta-classifier not found at {meta_path}. Using average.")

        print("  Ensemble ready.")

    def _load_model(self, name: str):
        """Load a single model checkpoint."""
        cfg = self.MODEL_CONFIGS[name]
        ckpt_path = self.checkpoint_dir / cfg["checkpoint"]

        if not ckpt_path.exists():
            print(f"  [WARN] {name} checkpoint not found at {ckpt_path}. Skipping.")
            return

        self.tokenizers[name] = AutoTokenizer.from_pretrained(ckpt_path)
        self.models[name] = AutoModelForSequenceClassification.from_pretrained(ckpt_path)
        self.models[name].to(self.device)
        self.models[name].eval()
        print(f"  Loaded {name} from {ckpt_path}")

    def _predict_single_model(
        self, name: str, text: str
    ) -> float:
        """Get AI probability from a single model."""
        if name not in self.models:
            return 0.5  # Neutral if model not loaded

        cfg = self.MODEL_CONFIGS[name]
        tokenizer = self.tokenizers[name]
        model = self.models[name]

        encoding = tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=cfg["max_length"],
            return_tensors="pt",
        )
        encoding = {k: v.to(self.device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = model(**encoding)
            prob = torch.softmax(outputs.logits, dim=-1)[0, 1].item()

        return prob

    def _predict_single_model_batch(
        self, name: str, texts: list[str], batch_size: int = 32
    ) -> np.ndarray:
        """Get AI probabilities from a single model for a batch of texts."""
        if name not in self.models:
            return np.full(len(texts), 0.5)

        cfg = self.MODEL_CONFIGS[name]
        tokenizer = self.tokenizers[name]
        model = self.models[name]

        all_probs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encoding = tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=cfg["max_length"],
                return_tensors="pt",
            )
            encoding = {k: v.to(self.device) for k, v in encoding.items()}

            with torch.no_grad():
                outputs = model(**encoding)
                probs = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()
            all_probs.extend(probs)

        return np.array(all_probs)

    def detect(self, text: str) -> float:
        """
        Detect AI-generated content.

        Args:
            text: Input text to analyze.

        Returns:
            Float between 0.0 (human) and 1.0 (AI).
        """
        scores = []
        for name in self.MODEL_CONFIGS:
            score = self._predict_single_model(name, text)
            scores.append(score)

        scores = np.array(scores).reshape(1, -1)

        if self.meta_classifier is not None:
            return float(self.meta_classifier.predict_proba(scores)[0, 1])
        else:
            return float(np.mean(scores))

    def detect_batch(self, texts: list[str], batch_size: int = 32) -> list[float]:
        """
        Detect AI content for multiple texts.

        Args:
            texts: List of texts to analyze.
            batch_size: Batch size for inference.

        Returns:
            List of floats, each between 0.0 and 1.0.
        """
        all_model_preds = []
        for name in self.MODEL_CONFIGS:
            preds = self._predict_single_model_batch(name, texts, batch_size)
            all_model_preds.append(preds)

        # Stack: (n_samples, n_models)
        features = np.column_stack(all_model_preds)

        if self.meta_classifier is not None:
            return self.meta_classifier.predict_proba(features)[:, 1].tolist()
        else:
            return np.mean(features, axis=1).tolist()

    def detect_detailed(self, text: str) -> dict:
        """
        Get per-model scores and ensemble score.

        Returns:
            Dict with model names as keys and scores as values,
            plus an "ensemble" key with the final combined score.
        """
        result = {}
        scores = []

        for name in self.MODEL_CONFIGS:
            score = self._predict_single_model(name, text)
            result[name] = round(score, 4)
            scores.append(score)

        scores = np.array(scores).reshape(1, -1)
        if self.meta_classifier is not None:
            ensemble_score = float(self.meta_classifier.predict_proba(scores)[0, 1])
        else:
            ensemble_score = float(np.mean(scores))

        result["ensemble"] = round(ensemble_score, 4)
        return result

    def is_ai_generated(self, text: str, threshold: float = AI_THRESHOLD) -> bool:
        """Binary decision: is this text AI-generated?"""
        return self.detect(text) >= threshold


# ─── Convenience function (matches interface contract) ───

_detector_instance = None


def detect(text: str) -> float:
    """
    Convenience function matching the interface contract.
    detect(text: str) → float (0.0 = human, 1.0 = AI)

    Lazily initializes the detector on first call.
    """
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = AIDetector()
    return _detector_instance.detect(text)
