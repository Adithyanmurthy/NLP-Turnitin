"""
Utility functions for plagiarism detection
"""

import re
import numpy as np
from typing import List, Tuple
import nltk
from nltk.tokenize import sent_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


def preprocess_text(text: str, lowercase: bool = True, remove_extra_spaces: bool = True) -> str:
    """
    Preprocess text for plagiarism detection.
    
    Args:
        text: Input text
        lowercase: Convert to lowercase
        remove_extra_spaces: Remove extra whitespace
    
    Returns:
        Preprocessed text
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    if remove_extra_spaces:
        text = re.sub(r'\s+', ' ', text)
    
    # Convert to lowercase
    if lowercase:
        text = text.lower()
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def split_sentences(text: str) -> List[str]:
    """
    Split text into sentences using NLTK.
    
    Args:
        text: Input text
    
    Returns:
        List of sentences
    """
    if not text:
        return []
    
    sentences = sent_tokenize(text)
    # Filter out very short sentences (likely noise)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    
    return sentences


def compute_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
    
    Returns:
        Cosine similarity score (0.0 to 1.0)
    """
    if vec1.ndim == 1:
        vec1 = vec1.reshape(1, -1)
    if vec2.ndim == 1:
        vec2 = vec2.reshape(1, -1)
    
    dot_product = np.dot(vec1, vec2.T)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    similarity = dot_product / (norm1 * norm2)
    return float(similarity[0][0]) if similarity.ndim > 0 else float(similarity)


def create_shingles(text: str, k: int = 3) -> List[str]:
    """
    Create k-shingles (k-grams) from text for MinHash.
    
    Args:
        text: Input text
        k: Shingle size (number of words)
    
    Returns:
        List of shingles
    """
    words = text.split()
    if len(words) < k:
        return [text]
    
    shingles = []
    for i in range(len(words) - k + 1):
        shingle = ' '.join(words[i:i+k])
        shingles.append(shingle)
    
    return shingles


def compute_overlap_ratio(text1: str, text2: str) -> float:
    """
    Compute word-level overlap ratio between two texts.
    
    Args:
        text1: First text
        text2: Second text
    
    Returns:
        Overlap ratio (0.0 to 1.0)
    """
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union) if union else 0.0


def align_sentences(
    input_sentences: List[str],
    reference_sentences: List[str],
    similarity_scores: np.ndarray,
    threshold: float = 0.7
) -> List[Tuple[int, int, float]]:
    """
    Align input sentences with reference sentences based on similarity scores.
    
    Args:
        input_sentences: List of input sentences
        reference_sentences: List of reference sentences
        similarity_scores: Matrix of similarity scores (input x reference)
        threshold: Minimum similarity threshold
    
    Returns:
        List of (input_idx, reference_idx, score) tuples
    """
    alignments = []
    
    for i in range(len(input_sentences)):
        for j in range(len(reference_sentences)):
            score = similarity_scores[i, j]
            if score >= threshold:
                alignments.append((i, j, float(score)))
    
    # Sort by score (descending)
    alignments.sort(key=lambda x: x[2], reverse=True)
    
    return alignments


def calculate_document_similarity(alignments: List[Tuple[int, int, float]], num_input_sentences: int) -> float:
    """
    Calculate overall document similarity based on sentence alignments.
    
    Args:
        alignments: List of (input_idx, reference_idx, score) tuples
        num_input_sentences: Total number of input sentences
    
    Returns:
        Document similarity score (0.0 to 1.0)
    """
    if not alignments or num_input_sentences == 0:
        return 0.0
    
    # Get unique input sentences that have matches
    matched_input_indices = set(a[0] for a in alignments)
    
    # Calculate average similarity of matched sentences
    avg_similarity = sum(a[2] for a in alignments) / len(alignments)
    
    # Calculate coverage (percentage of input sentences matched)
    coverage = len(matched_input_indices) / num_input_sentences
    
    # Combined score: weighted average of similarity and coverage
    document_score = 0.7 * avg_similarity + 0.3 * coverage
    
    return float(document_score)


def format_plagiarism_report(matches: List[dict], overall_score: float) -> dict:
    """
    Format plagiarism detection results into a standardized report.
    
    Args:
        matches: List of match dictionaries
        overall_score: Overall plagiarism score
    
    Returns:
        Formatted report dictionary
    """
    report = {
        "score": float(overall_score),
        "num_matches": len(matches),
        "matches": matches,
        "verdict": get_plagiarism_verdict(overall_score)
    }
    
    return report


def get_plagiarism_verdict(score: float) -> str:
    """
    Get plagiarism verdict based on score.
    
    Args:
        score: Plagiarism score (0.0 to 1.0)
    
    Returns:
        Verdict string
    """
    if score >= 0.8:
        return "High plagiarism detected"
    elif score >= 0.5:
        return "Moderate plagiarism detected"
    elif score >= 0.3:
        return "Low plagiarism detected"
    else:
        return "No significant plagiarism detected"
