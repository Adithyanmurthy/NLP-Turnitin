"""
Person 2: Plagiarism Detection Engine
Core implementation modules
"""

from .reference_index import ReferenceIndexBuilder, ReferenceIndexQuery
from .similarity_models import SentenceBERTModel, SimCSEModel, CrossEncoderModel, LongformerSimilarity
from .plagiarism_detector import PlagiarismDetector
from .utils import preprocess_text, split_sentences, compute_cosine_similarity

__all__ = [
    'ReferenceIndexBuilder',
    'ReferenceIndexQuery',
    'SentenceBERTModel',
    'SimCSEModel',
    'CrossEncoderModel',
    'LongformerSimilarity',
    'PlagiarismDetector',
    'preprocess_text',
    'split_sentences',
    'compute_cosine_similarity'
]

__version__ = '1.0.0'
