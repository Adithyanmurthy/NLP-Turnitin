"""
Humanization Module — Wrapper around Person 3's implementation
Bridges Person 3's Humanizer into Person 4's pipeline.

Person 3's contract: humanize(text: str) → dict with {text, ai_score_before, ai_score_after, ...}
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add person_3 to sys.path so we can import from it
_PERSON3_DIR = Path(__file__).resolve().parent.parent.parent.parent / "person_3"
if str(_PERSON3_DIR) not in sys.path:
    sys.path.insert(0, str(_PERSON3_DIR))


class Humanizer:
    """
    Humanization Module — wraps Person 3's Humanizer.

    Accepts Person 4's HumanizerConfig but translates it into
    the parameters Person 3's Humanizer actually expects.
    """

    def __init__(self, config=None):
        """
        Initialize Humanizer by loading Person 3's implementation.

        Args:
            config: Person 4's HumanizerConfig dataclass (optional).
        """
        self.config = config
        self._humanizer = None

        try:
            from humanizer import Humanizer as P3Humanizer

            # Person 3's Humanizer expects:
            #   model_name (str), use_feedback (bool)
            model_name = "flan_t5"  # default model
            use_feedback = True

            self._humanizer = P3Humanizer(
                model_name=model_name,
                use_feedback=use_feedback,
            )
            print("[Humanizer] Loaded Person 3's humanization module")
        except Exception as e:
            print(f"[Humanizer] Could not load Person 3's module: {e}")
            print("[Humanizer] Running in stub/mock mode")

    def humanize(self, text: str) -> Dict[str, Any]:
        """
        Transform AI-generated text to human-like text.

        Args:
            text: Input text to humanize.

        Returns:
            Dict with keys: text, ai_score_before, ai_score_after, iterations, success
        """
        if self._humanizer is not None:
            result = self._humanizer.humanize(text)
            # Normalize to the shape Person 4's pipeline expects
            target = self.config.target_ai_score if self.config else 0.05
            return {
                "text": result.get("text", text),
                "ai_score_before": result.get("ai_score_before", 0.0),
                "ai_score_after": result.get("ai_score_after", 0.0),
                "iterations": result.get("iterations", 0),
                "success": result.get("ai_score_after", 1.0) <= target,
                "model_used": result.get("model_used", "unknown"),
            }

        # Stub fallback
        print(f"[Humanizer STUB] humanize() called — text length: {len(text)}")
        target = self.config.target_ai_score if self.config else 0.05
        return {
            "text": text,
            "ai_score_before": 0.5,
            "ai_score_after": 0.5,
            "iterations": 0,
            "success": False,
            "model_used": "stub",
        }

    def set_ai_detector(self, ai_detector):
        """Pass Person 1's AI detector to Person 3 for the feedback loop."""
        if self._humanizer is not None and hasattr(self._humanizer, "ai_detector"):
            self._humanizer.ai_detector = ai_detector.detect if hasattr(ai_detector, "detect") else ai_detector
            print("[Humanizer] AI detector set for feedback loop")
