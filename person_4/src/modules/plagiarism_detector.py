"""
Plagiarism Detection Module — Wrapper around Person 2's implementation
Bridges Person 2's PlagiarismDetector into Person 4's pipeline.

Person 2's contract: check(text: str) → dict with {score, matches, ...}
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add person_2 to sys.path so we can import from it
_PERSON2_DIR = Path(__file__).resolve().parent.parent.parent.parent / "person_2"
if str(_PERSON2_DIR) not in sys.path:
    sys.path.insert(0, str(_PERSON2_DIR))


class PlagiarismDetector:
    """
    Plagiarism Detection Module — wraps Person 2's PlagiarismDetector.

    Accepts Person 4's PlagiarismDetectorConfig but translates it into
    the parameters Person 2's PlagiarismDetector actually expects.
    """

    def __init__(self, config=None):
        """
        Initialize Plagiarism Detector by loading Person 2's implementation.

        Args:
            config: Person 4's PlagiarismDetectorConfig dataclass (optional).
        """
        self.config = config
        self._detector = None

        try:
            from src.plagiarism_detector import PlagiarismDetector as P2Detector

            # Person 2's PlagiarismDetector expects:
            #   index_path (str), models_path (str|None), device (str|None), ...
            index_path = str(config.index_dir) if config else str(
                _PERSON2_DIR / "reference_index"
            )
            models_path = str(config.model_dir) if config else str(
                _PERSON2_DIR / "models"
            )
            device = config.device if config else None

            self._detector = P2Detector(
                index_path=index_path,
                models_path=models_path,
                device=device,
            )
            print("[PlagiarismDetector] Loaded Person 2's plagiarism engine")
        except Exception as e:
            print(f"[PlagiarismDetector] Could not load Person 2's module: {e}")
            print("[PlagiarismDetector] Running in stub/mock mode")

    def check(self, text: str) -> Dict[str, Any]:
        """
        Check text for plagiarism.

        Args:
            text: Input text to check.

        Returns:
            Dict with keys: score, matches, total_matches, highest_similarity, verdict
        """
        if self._detector is not None:
            result = self._detector.check(text)
            # Normalize to the shape Person 4's pipeline expects
            return {
                "score": result.get("score", 0.0),
                "matches": result.get("matches", []),
                "total_matches": result.get("num_matches", len(result.get("matches", []))),
                "highest_similarity": max(
                    (m.get("similarity", 0.0) for m in result.get("matches", [])),
                    default=0.0,
                ),
                "verdict": result.get("verdict", ""),
            }

        # Stub fallback
        print(f"[PlagiarismDetector STUB] check() called — text length: {len(text)}")
        return {
            "score": 0.0,
            "matches": [],
            "total_matches": 0,
            "highest_similarity": 0.0,
            "verdict": "No significant plagiarism detected",
        }

    def build_index(self, documents: List[str], document_ids: List[str]):
        """Proxy to Person 2's index builder (if available)."""
        if self._detector is not None and hasattr(self._detector, "build_index"):
            self._detector.build_index(documents, document_ids)
        else:
            print("[PlagiarismDetector STUB] build_index() — not available")
