"""
AI Detection Module — Wrapper around Person 1's implementation
Bridges Person 1's AIDetector into Person 4's pipeline.

Person 1's contract: detect(text: str) → float (0.0 = human, 1.0 = AI)
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add person_1 to sys.path so we can import from it
_PERSON1_DIR = Path(__file__).resolve().parent.parent.parent.parent / "person_1"
if str(_PERSON1_DIR) not in sys.path:
    sys.path.insert(0, str(_PERSON1_DIR))


class AIDetector:
    """
    AI Detection Module — wraps Person 1's AIDetector.
    
    Accepts Person 4's AIDetectorConfig but translates it into
    the parameters Person 1's AIDetector actually expects.
    """

    def __init__(self, config=None):
        """
        Initialize AI Detector by loading Person 1's implementation.

        Args:
            config: Person 4's AIDetectorConfig dataclass (optional).
                    If None, Person 1's defaults are used.
        """
        self.config = config
        self.last_model_scores = {}
        self._detector = None

        try:
            from ai_detector import AIDetector as P1AIDetector

            # Person 1's AIDetector expects:
            #   checkpoint_dir (Path), device (torch.device), load_all (bool)
            kwargs = {}
            if config is not None:
                kwargs["checkpoint_dir"] = config.model_dir
            
            self._detector = P1AIDetector(**kwargs)
            print("[AIDetector] Loaded Person 1's AI detection ensemble")
        except Exception as e:
            print(f"[AIDetector] Could not load Person 1's module: {e}")
            print("[AIDetector] Running in stub/mock mode")

    def detect(self, text: str) -> float:
        """
        Detect if text is AI-generated.

        Args:
            text: Input text to analyze.

        Returns:
            Float between 0.0 (human) and 1.0 (AI).
        """
        if self._detector is not None:
            score = self._detector.detect(text)
            # Capture per-model scores for the pipeline report
            if hasattr(self._detector, 'detect_detailed'):
                try:
                    details = self._detector.detect_detailed(text)
                    self.last_model_scores = {
                        k: v for k, v in details.items() if k != "ensemble"
                    }
                except Exception:
                    self.last_model_scores = {}
            return score

        # Stub fallback when Person 1's module isn't available yet
        print(f"[AIDetector STUB] detect() called — text length: {len(text)}")
        self.last_model_scores = {
            "deberta": 0.75, "roberta": 0.82,
            "longformer": 0.68, "xlm_roberta": 0.79,
        }
        return 0.76

    def get_model_scores(self) -> Dict[str, float]:
        """Get individual model scores from last detection."""
        return self.last_model_scores
