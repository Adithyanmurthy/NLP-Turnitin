"""
Person 1 — Data Pipeline & AI Detection Module

Provides:
    - DataLoader: Shared data loading utility for all team members
    - AIDetector: AI detection ensemble (DeBERTa + RoBERTa + Longformer + XLM-RoBERTa + meta-classifier)
    - detect(text) → float: Convenience function returning AI probability 0.0-1.0
"""

from person_1.ai_detector import AIDetector, detect
from person_1.data_loader import DataLoader

__all__ = ["AIDetector", "detect", "DataLoader"]
