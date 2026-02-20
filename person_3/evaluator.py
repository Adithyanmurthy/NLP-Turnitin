"""
Person 3 - Evaluation Module
Evaluates humanization quality and meaning preservation
"""
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
import nltk

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    pass

class HumanizationEvaluator:
    """Evaluates humanization quality"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load sentence transformer for semantic similarity
        print("[EVALUATOR] Loading sentence transformer...")
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        self.model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2").to(self.device)
        self.model.eval()
        
        # Initialize ROUGE
        self.rouge = Rouge()
        
        print("[EVALUATOR] Evaluator initialized")
    
    def get_embedding(self, text):
        """Get sentence embedding"""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        return embeddings.cpu().numpy()
    
    def semantic_similarity(self, text1, text2):
        """Calculate semantic similarity between two texts"""
        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)
        
        similarity = cosine_similarity(emb1, emb2)[0][0]
        return float(similarity)
    
    def calculate_bleu(self, reference, hypothesis):
        """Calculate BLEU score"""
        reference_tokens = reference.split()
        hypothesis_tokens = hypothesis.split()
        
        smoothing = SmoothingFunction().method1
        score = sentence_bleu(
            [reference_tokens],
            hypothesis_tokens,
            smoothing_function=smoothing
        )
        return float(score)
    
    def calculate_rouge(self, reference, hypothesis):
        """Calculate ROUGE scores"""
        try:
            scores = self.rouge.get_scores(hypothesis, reference)[0]
            return {
                "rouge-1": scores["rouge-1"]["f"],
                "rouge-2": scores["rouge-2"]["f"],
                "rouge-l": scores["rouge-l"]["f"]
            }
        except:
            return {"rouge-1": 0.0, "rouge-2": 0.0, "rouge-l": 0.0}
    
    def evaluate(self, original, humanized):
        """
        Comprehensive evaluation of humanization
        
        Args:
            original: Original text
            humanized: Humanized text
        
        Returns:
            dict with evaluation metrics
        """
        metrics = {}
        
        # Semantic similarity (meaning preservation)
        metrics["semantic_similarity"] = self.semantic_similarity(original, humanized)
        
        # BLEU score
        metrics["bleu"] = self.calculate_bleu(original, humanized)
        
        # ROUGE scores
        rouge_scores = self.calculate_rouge(original, humanized)
        metrics.update(rouge_scores)
        
        # Lexical diversity
        original_words = set(original.lower().split())
        humanized_words = set(humanized.lower().split())
        
        if len(original_words) > 0:
            word_overlap = len(original_words & humanized_words) / len(original_words)
            metrics["word_overlap"] = word_overlap
            metrics["lexical_diversity"] = 1.0 - word_overlap
        else:
            metrics["word_overlap"] = 0.0
            metrics["lexical_diversity"] = 0.0
        
        return metrics
    
    def evaluate_batch(self, original_texts, humanized_texts):
        """Evaluate a batch of texts"""
        results = []
        
        for orig, human in zip(original_texts, humanized_texts):
            metrics = self.evaluate(orig, human)
            results.append(metrics)
        
        # Calculate averages
        avg_metrics = {}
        for key in results[0].keys():
            avg_metrics[f"avg_{key}"] = np.mean([r[key] for r in results])
        
        return results, avg_metrics

# Test function
if __name__ == "__main__":
    print("=" * 80)
    print("TESTING EVALUATOR MODULE")
    print("=" * 80)
    
    evaluator = HumanizationEvaluator()
    
    original = "Artificial intelligence has revolutionized numerous industries."
    humanized = "AI has transformed many sectors and changed how businesses operate."
    
    print(f"\nOriginal: {original}")
    print(f"Humanized: {humanized}\n")
    
    metrics = evaluator.evaluate(original, humanized)
    
    print("Evaluation Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
