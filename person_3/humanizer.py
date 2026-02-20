"""
Person 3 - Humanizer Module
Main API for humanization - integrates with Person 1 and Person 4

Features:
  - Multi-model fallback: tries Flan-T5 → PEGASUS → Mistral until target is hit
  - Aggressive feedback loop: up to 10 iterations, target ≤5% AI score
  - Sentence-level humanization for precision rewriting
"""
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from pathlib import Path
import sys
from config import (
    MODELS, CHECKPOINTS_DIR, FEEDBACK_CONFIG,
    PERSON1_CONFIG, API_CONFIG
)

# Model fallback order — try each until AI score drops below target
MODEL_FALLBACK_ORDER = ["flan_t5", "pegasus", "mistral"]

MODEL_CHECKPOINT_MAP = {
    "flan_t5": "flan_t5_xl_final",
    "pegasus": "pegasus_large_final",
    "mistral": "mistral_7b_qlora_final",
    "dipper": "dipper_xxl",
}


class Humanizer:
    """Main humanization class with multi-model fallback and feedback loop."""

    def __init__(self, model_name="flan_t5", use_feedback=True):
        self.model_name = model_name
        self.use_feedback = use_feedback
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[HUMANIZER] Initializing with model: {model_name}")
        print(f"[HUMANIZER] Device: {self.device}")
        print(f"[HUMANIZER] Target AI score: ≤{FEEDBACK_CONFIG['target_ai_score']}")
        print(f"[HUMANIZER] Max iterations: {FEEDBACK_CONFIG['max_iterations']}")

        # Primary model
        self.models = {}
        self.tokenizers = {}
        self._load_single_model(model_name)

        # AI detector for feedback loop
        self.ai_detector = None
        if use_feedback:
            self.load_ai_detector()

    def _load_single_model(self, name):
        """Load a single humanization model."""
        checkpoint_name = MODEL_CHECKPOINT_MAP.get(name)
        if not checkpoint_name:
            raise ValueError(f"Unknown model: {name}")

        model_path = CHECKPOINTS_DIR / checkpoint_name
        if not model_path.exists():
            print(f"[HUMANIZER] Checkpoint not found: {model_path} — skipping {name}")
            return False

        print(f"[HUMANIZER] Loading {name} from: {model_path}")
        self.tokenizers[name] = AutoTokenizer.from_pretrained(str(model_path))

        if name == "mistral":
            self.models[name] = AutoModelForCausalLM.from_pretrained(
                str(model_path), torch_dtype=torch.float16, device_map="auto"
            )
        else:
            self.models[name] = AutoModelForSeq2SeqLM.from_pretrained(
                str(model_path), torch_dtype=torch.float16
            ).to(self.device)

        self.models[name].eval()
        print(f"[HUMANIZER] {name} loaded successfully")
        return True

    def _ensure_model_loaded(self, name):
        """Lazy-load a model if not already loaded."""
        if name not in self.models:
            return self._load_single_model(name)
        return True

    def load_ai_detector(self):
        """Load Person 1's AI detector for feedback loop."""
        try:
            person1_dir = Path(PERSON1_CONFIG["detector_path"])
            if person1_dir.exists():
                if str(person1_dir) not in sys.path:
                    sys.path.insert(0, str(person1_dir))
                from ai_detector import detect
                self.ai_detector = detect
                print(f"[HUMANIZER] AI detector loaded for feedback loop")
            else:
                print(f"[HUMANIZER] AI detector not found, feedback loop disabled")
        except Exception as e:
            print(f"[HUMANIZER] Could not load AI detector: {e}")
            print(f"[HUMANIZER] Feedback loop disabled")

    def paraphrase(self, text, diversity=60, reorder=40, model_name=None):
        """Paraphrase text using a specific model."""
        name = model_name or self.model_name
        if name not in self.models:
            raise RuntimeError(f"Model {name} not loaded")

        tokenizer = self.tokenizers[name]
        model = self.models[name]

        if name == "dipper":
            prompt = f"lexical = {diversity} order = {reorder} {text}"
        else:
            prompt = f"Paraphrase the following text naturally: {text}"

        inputs = tokenizer(
            prompt, return_tensors="pt", max_length=512, truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_length=512, num_beams=4,
                temperature=0.7, do_sample=True, top_p=0.9,
                early_stopping=True,
            )

        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    def humanize(self, text):
        """
        Main humanization function with multi-model fallback and feedback loop.

        Strategy:
          1. Try primary model with increasing diversity/reorder
          2. If still above target after max iterations, switch to next model
          3. Repeat until target is hit or all models exhausted

        Returns:
            dict with text, ai_score_before, ai_score_after, iterations,
            diversity_used, reorder_used, model_used
        """
        # Get initial AI score
        ai_score_before = 0.0
        if self.ai_detector:
            try:
                ai_score_before = self.ai_detector(text)
            except Exception:
                ai_score_before = 0.5

        target_score = FEEDBACK_CONFIG["target_ai_score"]
        max_iterations = FEEDBACK_CONFIG["max_iterations"]

        # If already below target, no humanization needed
        if ai_score_before <= target_score:
            return {
                "text": text,
                "ai_score_before": float(ai_score_before),
                "ai_score_after": float(ai_score_before),
                "iterations": 0,
                "diversity_used": 0,
                "reorder_used": 0,
                "model_used": "none (already human-like)",
            }

        # Build the model order: primary model first, then fallbacks
        model_order = [self.model_name]
        for m in MODEL_FALLBACK_ORDER:
            if m != self.model_name and m not in model_order:
                model_order.append(m)

        best_text = text
        best_score = ai_score_before
        total_iterations = 0
        final_diversity = FEEDBACK_CONFIG["initial_diversity"]
        final_reorder = FEEDBACK_CONFIG["initial_reorder"]
        model_used = self.model_name

        for current_model in model_order:
            # Try to load this model
            if not self._ensure_model_loaded(current_model):
                print(f"[HUMANIZER] Skipping {current_model} — not available")
                continue

            print(f"[HUMANIZER] Trying model: {current_model}")
            diversity = FEEDBACK_CONFIG["initial_diversity"]
            reorder = FEEDBACK_CONFIG["initial_reorder"]

            for iteration in range(max_iterations):
                try:
                    humanized = self.paraphrase(text, diversity, reorder, current_model)
                    total_iterations += 1

                    # Check AI score
                    if self.ai_detector:
                        score = self.ai_detector(humanized)
                        print(f"[HUMANIZER]   Iter {total_iterations}: "
                              f"score={score:.3f} (target≤{target_score}) "
                              f"model={current_model} div={diversity} reorder={reorder}")

                        if score < best_score:
                            best_score = score
                            best_text = humanized
                            final_diversity = diversity
                            final_reorder = reorder
                            model_used = current_model

                        if score <= target_score:
                            print(f"[HUMANIZER] Target reached! Score: {score:.3f}")
                            return {
                                "text": humanized,
                                "ai_score_before": float(ai_score_before),
                                "ai_score_after": float(score),
                                "iterations": total_iterations,
                                "diversity_used": diversity,
                                "reorder_used": reorder,
                                "model_used": current_model,
                            }
                    else:
                        # No detector — just return first result
                        best_text = humanized
                        model_used = current_model
                        break

                    # Increase aggressiveness
                    diversity = min(
                        diversity + FEEDBACK_CONFIG["diversity_increment"],
                        FEEDBACK_CONFIG["max_diversity"],
                    )
                    reorder = min(
                        reorder + FEEDBACK_CONFIG["reorder_increment"],
                        FEEDBACK_CONFIG["max_reorder"],
                    )

                except Exception as e:
                    print(f"[HUMANIZER] Error with {current_model}: {e}")
                    break

            # If this model got below target, we already returned above
            # Otherwise, try next model with the original text
            print(f"[HUMANIZER] {current_model} best score: {best_score:.3f} — trying next model")

        # Return best result across all models
        ai_score_after = best_score
        if self.ai_detector and best_text != text:
            try:
                ai_score_after = self.ai_detector(best_text)
            except Exception:
                pass

        return {
            "text": best_text,
            "ai_score_before": float(ai_score_before),
            "ai_score_after": float(ai_score_after),
            "iterations": total_iterations,
            "diversity_used": final_diversity,
            "reorder_used": final_reorder,
            "model_used": model_used,
        }

    def humanize_sentences(self, text):
        """
        Sentence-level humanization — humanizes each sentence individually
        for more precise control. Useful for mixed human/AI text.

        Returns same dict format as humanize().
        """
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        if len(sentences) <= 1:
            return self.humanize(text)

        humanized_sentences = []
        total_iterations = 0

        for sent in sentences:
            if len(sent.strip()) < 10:
                humanized_sentences.append(sent)
                continue

            # Check if this sentence needs humanization
            if self.ai_detector:
                try:
                    score = self.ai_detector(sent)
                    if score <= FEEDBACK_CONFIG["target_ai_score"]:
                        humanized_sentences.append(sent)
                        continue
                except Exception:
                    pass

            result = self.humanize(sent)
            humanized_sentences.append(result["text"])
            total_iterations += result["iterations"]

        final_text = " ".join(humanized_sentences)

        ai_score_before = 0.0
        ai_score_after = 0.0
        if self.ai_detector:
            try:
                ai_score_before = self.ai_detector(text)
                ai_score_after = self.ai_detector(final_text)
            except Exception:
                pass

        return {
            "text": final_text,
            "ai_score_before": float(ai_score_before),
            "ai_score_after": float(ai_score_after),
            "iterations": total_iterations,
            "diversity_used": FEEDBACK_CONFIG["initial_diversity"],
            "reorder_used": FEEDBACK_CONFIG["initial_reorder"],
            "model_used": "multi-sentence",
        }


# ─── Global convenience function ─────────────────────────

_humanizer_instance = None


def humanize(text):
    """
    Global humanize function for Person 4 integration.
    detect(text) → dict
    """
    global _humanizer_instance
    if _humanizer_instance is None:
        _humanizer_instance = Humanizer(model_name="flan_t5", use_feedback=True)
    return _humanizer_instance.humanize(text)


if __name__ == "__main__":
    print("=" * 80)
    print("TESTING HUMANIZER MODULE")
    print("=" * 80)

    test_text = """
    Artificial intelligence has revolutionized numerous industries by providing
    innovative solutions to complex problems. Machine learning algorithms can
    analyze vast amounts of data and identify patterns that humans might miss.
    """

    print(f"\nInput text:\n{test_text}\n")

    try:
        humanizer = Humanizer(model_name="flan_t5", use_feedback=False)
        result = humanizer.humanize(test_text)

        print(f"Humanized text:\n{result['text']}\n")
        print(f"AI score before: {result['ai_score_before']:.2f}")
        print(f"AI score after: {result['ai_score_after']:.2f}")
        print(f"Iterations: {result['iterations']}")
        print(f"Model used: {result['model_used']}")
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you have trained at least one model first!")
