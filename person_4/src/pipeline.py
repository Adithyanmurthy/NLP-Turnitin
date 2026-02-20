"""
Main integration pipeline for Content Integrity Platform
Person 4: Integrates all three modules (P1, P2, P3) + file parsing + deplagiarization

Features:
  - File parsing: accepts TXT, PDF, DOCX, HTML (not just raw text)
  - AI detection with ensemble scoring
  - Plagiarism detection with source matching
  - Humanization with multi-model fallback (target ≤5% AI score)
  - Deplagiarization: detects plagiarized sections → rewrites → re-checks → repeat
  - Full analysis: all of the above in one call
"""

import time
from typing import Dict, Any, Optional
from pathlib import Path
import json

from src.config import PipelineConfig, CONFIG
from src.file_parser import parse_input
from src.utils import (
    get_logger,
    timing_decorator,
    validate_text_input,
    compute_text_hash,
    save_json,
    load_json
)

# Import modules from Person 1, 2, 3
try:
    from src.modules.ai_detector import AIDetector
except ImportError:
    AIDetector = None

try:
    from src.modules.plagiarism_detector import PlagiarismDetector
except ImportError:
    PlagiarismDetector = None

try:
    from src.modules.humanizer import Humanizer
except ImportError:
    Humanizer = None

try:
    from src.deplagiarizer import Deplagiarizer
except ImportError:
    Deplagiarizer = None


logger = get_logger(__name__)


class ContentIntegrityPipeline:
    """
    Main pipeline that integrates AI Detection, Plagiarism Detection,
    Humanization, and Deplagiarization.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or CONFIG
        self.cache_dir = self.config.cache_dir if self.config.enable_caching else None

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Initializing Content Integrity Pipeline...")

        # Person 1: AI Detection
        if AIDetector is not None:
            logger.info("Loading AI Detection module...")
            self.ai_detector = AIDetector(self.config.ai_detector)
        else:
            logger.warning("AI Detection module not available (Person 1's work)")
            self.ai_detector = None

        # Person 2: Plagiarism Detection
        if PlagiarismDetector is not None:
            logger.info("Loading Plagiarism Detection module...")
            self.plagiarism_detector = PlagiarismDetector(self.config.plagiarism_detector)
        else:
            logger.warning("Plagiarism Detection module not available (Person 2's work)")
            self.plagiarism_detector = None

        # Person 3: Humanization
        if Humanizer is not None:
            logger.info("Loading Humanization module...")
            self.humanizer = Humanizer(self.config.humanizer)
        else:
            logger.warning("Humanization module not available (Person 3's work)")
            self.humanizer = None

        # Connect AI detector to humanizer for feedback loop
        if self.humanizer is not None and self.ai_detector is not None:
            if hasattr(self.humanizer, 'set_ai_detector'):
                self.humanizer.set_ai_detector(self.ai_detector)
                logger.info("Connected AI detector to humanizer feedback loop")

        # Deplagiarizer: bridges Person 2 (detect) + Person 3 (rewrite)
        if Deplagiarizer is not None:
            self.deplagiarizer = Deplagiarizer(
                plagiarism_detector=self.plagiarism_detector,
                humanizer=self.humanizer,
            )
            logger.info("Deplagiarizer initialized")
        else:
            self.deplagiarizer = None

        logger.info("Pipeline initialization complete")

    # ─── Cache helpers ────────────────────────────────────

    def _get_cache_path(self, text_hash: str, operation: str) -> Path:
        return self.cache_dir / f"{operation}_{text_hash}.json"

    def _load_from_cache(self, text: str, operation: str) -> Optional[Dict[Any, Any]]:
        if not self.cache_dir:
            return None
        text_hash = compute_text_hash(text)
        cache_path = self._get_cache_path(text_hash, operation)
        if cache_path.exists():
            logger.info(f"Loading {operation} result from cache")
            return load_json(cache_path)
        return None

    def _save_to_cache(self, text: str, operation: str, result: Dict[Any, Any]):
        if not self.cache_dir:
            return
        text_hash = compute_text_hash(text)
        cache_path = self._get_cache_path(text_hash, operation)
        save_json(result, cache_path)

    def _validate(self, text: str):
        is_valid, error_msg = validate_text_input(
            text, self.config.min_text_length, self.config.max_text_length
        )
        if not is_valid:
            raise ValueError(f"Invalid input: {error_msg}")

    # ─── Core operations ──────────────────────────────────

    @timing_decorator
    def detect_ai(self, text: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Run AI detection on text (or file path).
        Accepts raw text OR a file path (TXT, PDF, DOCX).
        """
        text = parse_input(text)
        self._validate(text)

        if use_cache:
            cached = self._load_from_cache(text, "ai_detection")
            if cached:
                return cached

        if self.ai_detector is None:
            logger.warning("AI Detector not available, returning mock result")
            result = {
                'score': 0.5, 'label': 'unknown', 'confidence': 0.0,
                'model_scores': {}, 'error': 'AI Detector module not loaded'
            }
        else:
            score = self.ai_detector.detect(text)
            threshold = getattr(self.config.ai_detector, 'threshold', 0.5)
            result = {
                'score': float(score),
                'label': 'ai' if score > threshold else 'human',
                'confidence': abs(score - 0.5) * 2,
                'model_scores': getattr(self.ai_detector, 'last_model_scores', {})
            }

        if use_cache:
            self._save_to_cache(text, "ai_detection", result)
        return result

    @timing_decorator
    def check_plagiarism(self, text: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Check text for plagiarism.
        Accepts raw text OR a file path (TXT, PDF, DOCX).
        """
        text = parse_input(text)
        self._validate(text)

        if use_cache:
            cached = self._load_from_cache(text, "plagiarism")
            if cached:
                return cached

        if self.plagiarism_detector is None:
            logger.warning("Plagiarism Detector not available, returning mock result")
            result = {
                'score': 0.0, 'matches': [], 'total_matches': 0,
                'highest_similarity': 0.0,
                'error': 'Plagiarism Detector module not loaded'
            }
        else:
            result = self.plagiarism_detector.check(text)

        if use_cache:
            self._save_to_cache(text, "plagiarism", result)
        return result

    @timing_decorator
    def humanize(self, text: str, use_cache: bool = False) -> Dict[str, Any]:
        """
        Transform AI-generated text to human-like text (target ≤5% AI score).
        Uses multi-model fallback: Flan-T5 → PEGASUS → Mistral.
        Accepts raw text OR a file path.
        """
        text = parse_input(text)
        self._validate(text)

        if use_cache:
            cached = self._load_from_cache(text, "humanization")
            if cached:
                return cached

        if self.humanizer is None:
            logger.warning("Humanizer not available, returning mock result")
            result = {
                'text': text, 'ai_score_before': 0.5, 'ai_score_after': 0.5,
                'iterations': 0, 'success': False,
                'error': 'Humanizer module not loaded'
            }
        else:
            result = self.humanizer.humanize(text)

        if use_cache:
            self._save_to_cache(text, "humanization", result)
        return result

    @timing_decorator
    def deplagiarize(self, text: str, use_cache: bool = False) -> Dict[str, Any]:
        """
        Detect plagiarized sections and rewrite them until plagiarism ≤5%.
        Accepts raw text OR a file path.

        Pipeline:
          1. Plagiarism check → find flagged sentences
          2. Rewrite flagged sentences via humanizer
          3. Re-check → repeat until clean
        """
        text = parse_input(text)
        self._validate(text)

        if self.deplagiarizer is None:
            logger.warning("Deplagiarizer not available")
            return {
                'text': text, 'plagiarism_score_before': 0.0,
                'plagiarism_score_after': 0.0, 'sentences_rewritten': 0,
                'iterations': 0, 'success': False,
                'error': 'Deplagiarizer not available (needs Person 2 + Person 3)'
            }

        return self.deplagiarizer.deplagiarize(text)

    # ─── Full analysis ────────────────────────────────────

    @timing_decorator
    def analyze(
        self,
        text: str,
        check_ai: bool = True,
        check_plagiarism: bool = True,
        humanize: bool = False,
        deplagiarize: bool = False,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Run complete analysis on text.
        Accepts raw text OR a file path (TXT, PDF, DOCX).

        Args:
            text: Input text or file path
            check_ai: Run AI detection
            check_plagiarism: Run plagiarism check
            humanize: Humanize the text (reduce AI score to ≤5%)
            deplagiarize: Deplagiarize the text (reduce plagiarism to ≤5%)
            use_cache: Use cached results
        """
        # Parse file if needed
        parsed_text = parse_input(text)
        start_time = time.time()

        logger.info("Starting content analysis...")
        logger.info(f"Text length: {len(parsed_text)} characters")
        logger.info(f"Operations: AI={check_ai}, Plagiarism={check_plagiarism}, "
                     f"Humanize={humanize}, Deplagiarize={deplagiarize}")

        report = {
            'input_length': len(parsed_text),
            'timestamp': time.time(),
        }

        # AI Detection
        if check_ai:
            try:
                logger.info("Running AI detection...")
                report['ai_detection'] = self.detect_ai(parsed_text, use_cache)
            except Exception as e:
                logger.error(f"AI detection failed: {e}")
                report['ai_detection'] = {'error': str(e)}

        # Plagiarism Check
        if check_plagiarism:
            try:
                logger.info("Running plagiarism check...")
                report['plagiarism'] = self.check_plagiarism(parsed_text, use_cache)
            except Exception as e:
                logger.error(f"Plagiarism check failed: {e}")
                report['plagiarism'] = {'error': str(e)}

        # Humanization (reduce AI detection to ≤5%)
        if humanize:
            try:
                logger.info("Running humanization (target: ≤5% AI score)...")
                report['humanization'] = self.humanize(parsed_text, use_cache)
            except Exception as e:
                logger.error(f"Humanization failed: {e}")
                report['humanization'] = {'error': str(e)}

        # Deplagiarization (reduce plagiarism to ≤5%)
        if deplagiarize:
            try:
                logger.info("Running deplagiarization (target: ≤5% plagiarism)...")
                report['deplagiarization'] = self.deplagiarize(parsed_text, use_cache)
            except Exception as e:
                logger.error(f"Deplagiarization failed: {e}")
                report['deplagiarization'] = {'error': str(e)}

        report['processing_time'] = time.time() - start_time
        logger.info(f"Analysis complete in {report['processing_time']:.2f}s")
        return report

    def health_check(self) -> Dict[str, Any]:
        """Check health status of all modules."""
        return {
            'ai_detector': self.ai_detector is not None,
            'plagiarism_detector': self.plagiarism_detector is not None,
            'humanizer': self.humanizer is not None,
            'deplagiarizer': self.deplagiarizer is not None,
            'cache_enabled': self.cache_dir is not None,
            'supported_formats': ['.txt', '.pdf', '.docx', '.html', '.md'],
        }
