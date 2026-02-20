"""
Person 3 - Humanization & Content Transformation Module

This module provides humanization capabilities for the Content Integrity platform.
It transforms AI-generated text into naturally human-written text.

Main API:
    from person3.humanizer import humanize
    
    result = humanize("Your AI-generated text here")
    
    # Returns:
    # {
    #     "text": "humanized text",
    #     "ai_score_before": 0.85,
    #     "ai_score_after": 0.15,
    #     "iterations": 2,
    #     "diversity_used": 70,
    #     "reorder_used": 50
    # }
"""

__version__ = "1.0.0"
__author__ = "Person 3"
__module__ = "Humanization & Content Transformation"

# Import main API function
try:
    from .humanizer import humanize, Humanizer
    __all__ = ["humanize", "Humanizer"]
except ImportError:
    # Models not trained yet
    __all__ = []
    
    def humanize(text):
        raise RuntimeError(
            "Humanizer not available. Please train models first:\n"
            "  cd person3\n"
            "  python run_all.py"
        )
