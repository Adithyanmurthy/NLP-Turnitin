"""
Main Plagiarism Detection Pipeline
Integrates MinHash/LSH, Sentence-BERT, SimCSE, Cross-Encoder, and Longformer
"""

import os
import numpy as np
from typing import List, Dict, Optional, Union
from .reference_index import ReferenceIndexQuery
from .similarity_models import (
    SentenceBERTModel,
    SimCSEModel,
    CrossEncoderModel,
    LongformerSimilarity
)
from .utils import (
    preprocess_text,
    split_sentences,
    align_sentences,
    calculate_document_similarity,
    format_plagiarism_report
)


class PlagiarismDetector:
    """
    Complete plagiarism detection pipeline.
    
    Pipeline stages:
    1. MinHash/LSH screening - Fast candidate retrieval
    2. Sentence-level embedding similarity - Initial scoring
    3. Cross-encoder verification - Precise similarity scoring
    4. Document-level comparison (for long texts) - Overall similarity
    """
    
    def __init__(
        self,
        index_path: str,
        models_path: Optional[str] = None,
        use_sbert: bool = True,
        use_simcse: bool = False,
        use_cross_encoder: bool = True,
        use_longformer: bool = False,
        device: Optional[str] = None
    ):
        """
        Initialize plagiarism detector.
        
        Args:
            index_path: Path to MinHash/LSH index
            models_path: Path to fine-tuned model checkpoints (optional)
            use_sbert: Use Sentence-BERT for embeddings
            use_simcse: Use SimCSE for embeddings
            use_cross_encoder: Use Cross-Encoder for verification
            use_longformer: Use Longformer for long documents
            device: Device to use for models
        """
        self.index_path = index_path
        self.models_path = models_path
        self.device = device
        
        # Load reference index
        print("Loading reference index...")
        self.index = ReferenceIndexQuery(index_path)
        
        # Initialize models
        self.sbert = None
        self.simcse = None
        self.cross_encoder = None
        self.longformer = None
        
        if use_sbert:
            print("Loading Sentence-BERT...")
            sbert_path = self._get_model_path("sbert") or "sentence-transformers/all-mpnet-base-v2"
            self.sbert = SentenceBERTModel(sbert_path, device=device)
        
        if use_simcse:
            print("Loading SimCSE...")
            simcse_path = self._get_model_path("simcse") or "princeton-nlp/sup-simcse-roberta-large"
            self.simcse = SimCSEModel(simcse_path, device=device)
        
        if use_cross_encoder:
            print("Loading Cross-Encoder...")
            ce_path = self._get_model_path("cross_encoder") or "cross-encoder/nli-deberta-v3-large"
            self.cross_encoder = CrossEncoderModel(ce_path, device=device)
        
        if use_longformer:
            print("Loading Longformer...")
            lf_path = self._get_model_path("longformer") or "allenai/longformer-base-4096"
            self.longformer = LongformerSimilarity(lf_path, device=device)
        
        print("Plagiarism detector initialized successfully!")
    
    def _get_model_path(self, model_name: str) -> Optional[str]:
        """Get path to fine-tuned model checkpoint if available."""
        if not self.models_path:
            return None
        
        model_path = os.path.join(self.models_path, model_name)
        if os.path.exists(model_path):
            return model_path
        return None
    
    def check(
        self,
        text: str,
        top_k_candidates: int = 5,
        sentence_threshold: float = 0.75,
        verification_threshold: float = 0.7,
        min_sentence_length: int = 10
    ) -> Dict:
        """
        Check text for plagiarism.
        
        Args:
            text: Input text to check
            top_k_candidates: Number of candidate documents to retrieve from index
            sentence_threshold: Similarity threshold for sentence matching
            verification_threshold: Threshold for cross-encoder verification
            min_sentence_length: Minimum sentence length to consider
        
        Returns:
            Plagiarism report dictionary with format:
            {
                "score": float,  # Overall plagiarism score (0.0-1.0)
                "num_matches": int,
                "matches": [
                    {
                        "source": str,
                        "similarity": float,
                        "sentences": [
                            {
                                "input_sentence": str,
                                "matched_sentence": str,
                                "score": float
                            }
                        ]
                    }
                ],
                "verdict": str
            }
        """
        if not text or len(text.strip()) < min_sentence_length:
            return format_plagiarism_report([], 0.0)
        
        # Stage 1: MinHash/LSH screening
        print("Stage 1: Screening candidates with MinHash/LSH...")
        candidates = self.index.query(text, top_k=top_k_candidates)
        
        if not candidates:
            print("No candidate matches found in index")
            return format_plagiarism_report([], 0.0)
        
        print(f"Found {len(candidates)} candidate documents")
        
        # Split input text into sentences
        input_sentences = split_sentences(text)
        if not input_sentences:
            return format_plagiarism_report([], 0.0)
        
        print(f"Input text split into {len(input_sentences)} sentences")
        
        # Stage 2 & 3: Sentence-level comparison and verification
        all_matches = []
        
        for doc_id, jaccard_sim in candidates:
            print(f"\nAnalyzing candidate: {doc_id} (Jaccard: {jaccard_sim:.3f})")
            
            # Get reference document
            ref_doc = self.index.get_document(doc_id)
            if not ref_doc:
                continue
            
            # Split reference into sentences
            ref_sentences = split_sentences(ref_doc)
            if not ref_sentences:
                continue
            
            # Stage 2: Compute sentence embeddings and similarity
            similarity_matrix = self._compute_sentence_similarity(
                input_sentences,
                ref_sentences
            )
            
            # Find high-similarity sentence pairs
            alignments = align_sentences(
                input_sentences,
                ref_sentences,
                similarity_matrix,
                threshold=sentence_threshold
            )
            
            if not alignments:
                continue
            
            print(f"Found {len(alignments)} potential sentence matches")
            
            # Stage 3: Cross-encoder verification (if enabled)
            if self.cross_encoder:
                verified_matches = self._verify_with_cross_encoder(
                    input_sentences,
                    ref_sentences,
                    alignments,
                    threshold=verification_threshold
                )
            else:
                verified_matches = alignments
            
            if not verified_matches:
                continue
            
            print(f"Verified {len(verified_matches)} matches with cross-encoder")
            
            # Calculate document-level similarity
            doc_similarity = calculate_document_similarity(
                verified_matches,
                len(input_sentences)
            )
            
            # Format match result
            match_result = {
                "source": doc_id,
                "similarity": float(doc_similarity),
                "jaccard_similarity": float(jaccard_sim),
                "num_matched_sentences": len(verified_matches),
                "sentences": []
            }
            
            # Add sentence-level details (top 10 matches)
            for i, j, score in verified_matches[:10]:
                match_result["sentences"].append({
                    "input_sentence": input_sentences[i],
                    "matched_sentence": ref_sentences[j],
                    "score": float(score)
                })
            
            all_matches.append(match_result)
        
        # Sort matches by similarity
        all_matches.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Calculate overall plagiarism score
        if all_matches:
            # Weighted average of top matches
            top_scores = [m["similarity"] for m in all_matches[:3]]
            overall_score = sum(top_scores) / len(top_scores)
        else:
            overall_score = 0.0
        
        # Format and return report
        report = format_plagiarism_report(all_matches, overall_score)
        
        print(f"\n{'='*60}")
        print(f"Plagiarism Detection Complete")
        print(f"Overall Score: {overall_score:.2%}")
        print(f"Verdict: {report['verdict']}")
        print(f"Matches Found: {len(all_matches)}")
        print(f"{'='*60}")
        
        return report
    
    def _compute_sentence_similarity(
        self,
        input_sentences: List[str],
        ref_sentences: List[str]
    ) -> np.ndarray:
        """
        Compute sentence-level similarity matrix.
        
        Args:
            input_sentences: Input sentences
            ref_sentences: Reference sentences
        
        Returns:
            Similarity matrix (input x reference)
        """
        # Use Sentence-BERT if available, otherwise SimCSE
        if self.sbert:
            similarity_matrix = self.sbert.compute_similarity(
                input_sentences,
                ref_sentences
            )
        elif self.simcse:
            similarity_matrix = self.simcse.compute_similarity(
                input_sentences,
                ref_sentences
            )
        else:
            raise ValueError("No embedding model available (need Sentence-BERT or SimCSE)")
        
        return similarity_matrix
    
    def _verify_with_cross_encoder(
        self,
        input_sentences: List[str],
        ref_sentences: List[str],
        alignments: List[tuple],
        threshold: float = 0.7
    ) -> List[tuple]:
        """
        Verify sentence alignments with cross-encoder.
        
        Args:
            input_sentences: Input sentences
            ref_sentences: Reference sentences
            alignments: List of (input_idx, ref_idx, score) tuples
            threshold: Verification threshold
        
        Returns:
            Verified alignments
        """
        if not alignments:
            return []
        
        # Create sentence pairs for verification
        pairs = []
        for i, j, _ in alignments:
            pairs.append((input_sentences[i], ref_sentences[j]))
        
        # Get cross-encoder scores
        ce_scores = self.cross_encoder.predict(pairs)
        
        # Filter by threshold
        verified = []
        for (i, j, _), ce_score in zip(alignments, ce_scores):
            if ce_score >= threshold:
                verified.append((i, j, float(ce_score)))
        
        return verified
    
    def check_document_similarity(
        self,
        text1: str,
        text2: str
    ) -> float:
        """
        Compute document-level similarity between two texts.
        Useful for comparing long documents.
        
        Args:
            text1: First document
            text2: Second document
        
        Returns:
            Similarity score (0.0-1.0)
        """
        if self.longformer:
            similarity_matrix = self.longformer.compute_similarity(text1, text2)
            return float(similarity_matrix[0, 0])
        else:
            # Fall back to sentence-level comparison
            sentences1 = split_sentences(text1)
            sentences2 = split_sentences(text2)
            
            if not sentences1 or not sentences2:
                return 0.0
            
            similarity_matrix = self._compute_sentence_similarity(sentences1, sentences2)
            
            # Return average of maximum similarities
            max_sims = similarity_matrix.max(axis=1)
            return float(max_sims.mean())
    
    def get_statistics(self) -> Dict:
        """
        Get detector statistics.
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            "index_stats": self.index.get_statistics(),
            "models_loaded": {
                "sentence_bert": self.sbert is not None,
                "simcse": self.simcse is not None,
                "cross_encoder": self.cross_encoder is not None,
                "longformer": self.longformer is not None
            }
        }
        return stats
