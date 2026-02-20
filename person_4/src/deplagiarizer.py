"""
Deplagiarizer Module — Detects plagiarized sections and rewrites them to 0%.

Pipeline:
  1. Run plagiarism check (Person 2) to find flagged sentences
  2. Send only flagged sentences through humanizer (Person 3) for rewriting
  3. Re-check plagiarism on rewritten sections
  4. Repeat until plagiarism score drops below threshold

This bridges Person 2 (detection) and Person 3 (rewriting) into a
plagiarism elimination loop, similar to the AI humanization feedback loop.
"""

import re
from typing import Dict, Any, List, Optional


# Default config
DEPLAG_CONFIG = {
    "similarity_threshold": 0.3,   # Sentences above this are considered plagiarized
    "target_score": 0.05,          # Target overall plagiarism score (≤5%)
    "max_iterations": 5,           # Max rewrite cycles
    "min_sentence_length": 20,     # Skip very short sentences
}


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def join_sentences(sentences: List[str]) -> str:
    """Join sentences back into text."""
    return " ".join(sentences)


class Deplagiarizer:
    """
    Plagiarism elimination module.
    Uses Person 2's detector to find plagiarized sections,
    then Person 3's humanizer to rewrite them until clean.
    """

    def __init__(self, plagiarism_detector=None, humanizer=None, config=None):
        """
        Args:
            plagiarism_detector: Person 2's PlagiarismDetector instance
            humanizer: Person 3's Humanizer instance
            config: Override default config
        """
        self.detector = plagiarism_detector
        self.humanizer = humanizer
        self.config = config or DEPLAG_CONFIG

    def _check_plagiarism(self, text: str) -> Dict[str, Any]:
        """Run plagiarism check on text."""
        if self.detector is None:
            return {"score": 0.0, "matches": [], "total_matches": 0}
        return self.detector.check(text)

    def _humanize_text(self, text: str) -> str:
        """Rewrite text using the humanizer."""
        if self.humanizer is None:
            return text
        result = self.humanizer.humanize(text)
        if isinstance(result, dict):
            return result.get("text", text)
        return str(result)

    def _find_flagged_sentences(self, text: str, plag_result: Dict) -> List[int]:
        """
        Identify which sentence indices are flagged as plagiarized.
        Uses the matches from the plagiarism detector to find overlapping sentences.
        """
        sentences = split_into_sentences(text)
        flagged = set()

        matches = plag_result.get("matches", [])
        if not matches:
            # If overall score is high but no specific matches,
            # flag all sentences proportionally
            score = plag_result.get("score", 0.0)
            if score > self.config["similarity_threshold"]:
                # Flag all sentences
                return list(range(len(sentences)))
            return []

        # Check each match against sentences
        for match in matches:
            similarity = match.get("similarity", 0.0)
            if similarity < self.config["similarity_threshold"]:
                continue

            matched_text = match.get("text", match.get("sentence", "")).lower()
            if not matched_text:
                # If no specific text in match, flag based on score
                flagged.update(range(len(sentences)))
                continue

            # Find which sentences overlap with the matched text
            for i, sent in enumerate(sentences):
                if len(sent) < self.config["min_sentence_length"]:
                    continue
                # Simple overlap check
                sent_words = set(sent.lower().split())
                match_words = set(matched_text.split())
                if len(sent_words & match_words) > len(sent_words) * 0.3:
                    flagged.add(i)

        return sorted(flagged)

    def deplagiarize(self, text: str) -> Dict[str, Any]:
        """
        Main deplagiarization function.

        Detects plagiarized sections, rewrites them, re-checks,
        and repeats until the plagiarism score is below threshold.

        Args:
            text: Input text to deplagiarize.

        Returns:
            dict with:
                - text: Deplagiarized text
                - plagiarism_score_before: Score before processing
                - plagiarism_score_after: Score after processing
                - sentences_rewritten: Number of sentences that were rewritten
                - iterations: Number of rewrite cycles
                - success: Whether target was achieved
        """
        # Initial plagiarism check
        initial_result = self._check_plagiarism(text)
        score_before = initial_result.get("score", 0.0)

        # Already clean
        if score_before <= self.config["target_score"]:
            return {
                "text": text,
                "plagiarism_score_before": float(score_before),
                "plagiarism_score_after": float(score_before),
                "sentences_rewritten": 0,
                "iterations": 0,
                "success": True,
            }

        sentences = split_into_sentences(text)
        total_rewritten = 0
        current_text = text

        for iteration in range(self.config["max_iterations"]):
            # Find flagged sentences
            plag_result = self._check_plagiarism(current_text)
            current_score = plag_result.get("score", 0.0)

            print(f"[DEPLAGIARIZER] Iteration {iteration + 1}: "
                  f"score={current_score:.3f} (target≤{self.config['target_score']})")

            if current_score <= self.config["target_score"]:
                print(f"[DEPLAGIARIZER] Target reached!")
                return {
                    "text": current_text,
                    "plagiarism_score_before": float(score_before),
                    "plagiarism_score_after": float(current_score),
                    "sentences_rewritten": total_rewritten,
                    "iterations": iteration + 1,
                    "success": True,
                }

            # Find which sentences to rewrite
            current_sentences = split_into_sentences(current_text)
            flagged_indices = self._find_flagged_sentences(current_text, plag_result)

            if not flagged_indices:
                # No specific sentences flagged but score is high — rewrite everything
                print(f"[DEPLAGIARIZER] No specific matches, rewriting full text")
                current_text = self._humanize_text(current_text)
                total_rewritten += len(current_sentences)
                continue

            print(f"[DEPLAGIARIZER] Rewriting {len(flagged_indices)} flagged sentences")

            # Rewrite only flagged sentences
            for idx in flagged_indices:
                if idx < len(current_sentences):
                    original = current_sentences[idx]
                    if len(original) >= self.config["min_sentence_length"]:
                        rewritten = self._humanize_text(original)
                        current_sentences[idx] = rewritten
                        total_rewritten += 1

            current_text = join_sentences(current_sentences)

        # Final check
        final_result = self._check_plagiarism(current_text)
        final_score = final_result.get("score", 0.0)

        return {
            "text": current_text,
            "plagiarism_score_before": float(score_before),
            "plagiarism_score_after": float(final_score),
            "sentences_rewritten": total_rewritten,
            "iterations": self.config["max_iterations"],
            "success": final_score <= self.config["target_score"],
        }
