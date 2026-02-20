"""
Module wrappers for Person 1, 2, and 3
These bridge the actual implementations into Person 4's pipeline
"""

from .ai_detector import AIDetector
from .plagiarism_detector import PlagiarismDetector
from .humanizer import Humanizer

__all__ = ['AIDetector', 'PlagiarismDetector', 'Humanizer']
