"""
Unit tests for utility functions
"""

import os
import sys
import pytest
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import (
    preprocess_text,
    split_sentences,
    compute_cosine_similarity,
    create_shingles,
    compute_overlap_ratio,
    align_sentences,
    calculate_document_similarity,
    get_plagiarism_verdict
)


class TestUtils:
    
    def test_preprocess_text(self):
        """Test text preprocessing."""
        text = "  This  is   a   TEST.  "
        processed = preprocess_text(text, lowercase=True, remove_extra_spaces=True)
        assert processed == "this is a test."
        
        processed_no_lower = preprocess_text(text, lowercase=False)
        assert processed_no_lower == "This is a TEST."
    
    def test_split_sentences(self):
        """Test sentence splitting."""
        text = "This is the first sentence. This is the second sentence. And a third one."
        sentences = split_sentences(text)
        assert len(sentences) == 3
        assert "first sentence" in sentences[0]
        assert "second sentence" in sentences[1]
    
    def test_split_sentences_short(self):
        """Test that very short sentences are filtered out."""
        text = "Hi. This is a longer sentence that should be kept."
        sentences = split_sentences(text)
        assert len(sentences) == 1  # "Hi." should be filtered out
    
    def test_compute_cosine_similarity(self):
        """Test cosine similarity computation."""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([1.0, 0.0, 0.0])
        similarity = compute_cosine_similarity(vec1, vec2)
        assert abs(similarity - 1.0) < 1e-6
        
        vec3 = np.array([0.0, 1.0, 0.0])
        similarity = compute_cosine_similarity(vec1, vec3)
        assert abs(similarity - 0.0) < 1e-6
    
    def test_create_shingles(self):
        """Test shingle creation."""
        text = "the quick brown fox"
        shingles = create_shingles(text, k=2)
        assert len(shingles) == 3
        assert "the quick" in shingles
        assert "quick brown" in shingles
        assert "brown fox" in shingles
    
    def test_create_shingles_short_text(self):
        """Test shingle creation with text shorter than k."""
        text = "hello"
        shingles = create_shingles(text, k=3)
        assert len(shingles) == 1
        assert shingles[0] == "hello"
    
    def test_compute_overlap_ratio(self):
        """Test word overlap ratio."""
        text1 = "the quick brown fox"
        text2 = "the quick brown dog"
        ratio = compute_overlap_ratio(text1, text2)
        assert ratio > 0.5  # 3 out of 5 unique words overlap
        
        text3 = "completely different words"
        ratio = compute_overlap_ratio(text1, text3)
        assert ratio < 0.2
    
    def test_align_sentences(self):
        """Test sentence alignment."""
        input_sentences = ["This is sentence one.", "This is sentence two."]
        ref_sentences = ["This is sentence one.", "Something different."]
        
        # Create mock similarity matrix
        similarity_matrix = np.array([
            [0.95, 0.3],
            [0.4, 0.2]
        ])
        
        alignments = align_sentences(
            input_sentences,
            ref_sentences,
            similarity_matrix,
            threshold=0.7
        )
        
        assert len(alignments) == 1  # Only one pair above threshold
        assert alignments[0][0] == 0  # First input sentence
        assert alignments[0][1] == 0  # First reference sentence
        assert alignments[0][2] > 0.9  # High similarity
    
    def test_calculate_document_similarity(self):
        """Test document similarity calculation."""
        alignments = [(0, 0, 0.9), (1, 1, 0.8), (2, 2, 0.85)]
        num_input_sentences = 5
        
        doc_sim = calculate_document_similarity(alignments, num_input_sentences)
        
        assert 0.0 <= doc_sim <= 1.0
        assert doc_sim > 0.5  # Should be relatively high
    
    def test_calculate_document_similarity_no_matches(self):
        """Test document similarity with no matches."""
        alignments = []
        num_input_sentences = 5
        
        doc_sim = calculate_document_similarity(alignments, num_input_sentences)
        assert doc_sim == 0.0
    
    def test_get_plagiarism_verdict(self):
        """Test plagiarism verdict generation."""
        assert "High" in get_plagiarism_verdict(0.9)
        assert "Moderate" in get_plagiarism_verdict(0.6)
        assert "Low" in get_plagiarism_verdict(0.4)
        assert "No significant" in get_plagiarism_verdict(0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
