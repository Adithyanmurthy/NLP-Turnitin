"""
Utility functions for the Content Integrity Platform
Person 4: Helper functions and utilities
"""

import logging
import time
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional
from functools import wraps
import torch


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """
    Setup logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
    """
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance"""
    return logging.getLogger(name)


def timing_decorator(func):
    """Decorator to measure function execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        logger = get_logger(func.__module__)
        logger.info(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        
        return result
    return wrapper


def validate_text_input(text: str, min_length: int = 10, max_length: int = 50000) -> tuple[bool, str]:
    """
    Validate input text
    
    Args:
        text: Input text to validate
        min_length: Minimum text length
        max_length: Maximum text length
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not text or not isinstance(text, str):
        return False, "Text must be a non-empty string"
    
    text = text.strip()
    
    if len(text) < min_length:
        return False, f"Text too short (minimum {min_length} characters)"
    
    if len(text) > max_length:
        return False, f"Text too long (maximum {max_length} characters)"
    
    return True, ""


def compute_text_hash(text: str) -> str:
    """
    Compute SHA256 hash of text for caching
    
    Args:
        text: Input text
    
    Returns:
        Hex digest of hash
    """
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def save_json(data: Dict[Any, Any], filepath: Path):
    """Save data to JSON file"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(filepath: Path) -> Dict[Any, Any]:
    """Load data from JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_device() -> str:
    """
    Get the best available device for PyTorch
    
    Returns:
        Device string ('cuda' or 'cpu')
    """
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def format_percentage(value: float) -> str:
    """Format float as percentage string"""
    return f"{value * 100:.1f}%"


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to maximum length
    
    Args:
        text: Input text
        max_length: Maximum length
        suffix: Suffix to add if truncated
    
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


class ProgressTracker:
    """Simple progress tracker for long operations"""
    
    def __init__(self, total_steps: int, description: str = "Processing"):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.start_time = time.time()
        self.logger = get_logger(__name__)
    
    def update(self, step: int = 1):
        """Update progress"""
        self.current_step += step
        progress = (self.current_step / self.total_steps) * 100
        elapsed = time.time() - self.start_time
        
        self.logger.info(
            f"{self.description}: {progress:.1f}% "
            f"({self.current_step}/{self.total_steps}) - "
            f"Elapsed: {elapsed:.1f}s"
        )
    
    def finish(self):
        """Mark as finished"""
        elapsed = time.time() - self.start_time
        self.logger.info(
            f"{self.description} completed in {elapsed:.1f}s"
        )


def format_report(report: Dict[Any, Any], format_type: str = "text") -> str:
    """
    Format analysis report for display
    
    Args:
        report: Report dictionary
        format_type: 'text' or 'json'
    
    Returns:
        Formatted report string
    """
    if format_type == "json":
        return json.dumps(report, indent=2, ensure_ascii=False)
    
    # Text format
    lines = []
    lines.append("=" * 60)
    lines.append("  CONTENT ANALYSIS REPORT")
    lines.append("=" * 60)
    lines.append("")
    
    # AI Detection
    if 'ai_detection' in report:
        ai_score = report['ai_detection']['score']
        lines.append(f"  AI Detection Score:        {format_percentage(ai_score)}")
        lines.append(f"  Classification:            {'AI-generated' if ai_score > 0.5 else 'Human-written'}")
        lines.append("")
    
    # Plagiarism
    if 'plagiarism' in report:
        plag_score = report['plagiarism']['score']
        matches = report['plagiarism'].get('matches', [])
        lines.append(f"  Plagiarism Score:          {format_percentage(plag_score)}")
        lines.append(f"  Sources Found:             {len(matches)}")
        
        if matches:
            lines.append("")
            lines.append("  Matched Sources:")
            for i, match in enumerate(matches[:5], 1):  # Show top 5
                lines.append(f"    {i}. {match.get('source', 'Unknown')} — "
                           f"{format_percentage(match.get('similarity', 0))} similarity")
        lines.append("")
    
    # Humanization
    if 'humanization' in report:
        lines.append("  Humanization Results:")
        lines.append(f"    Before AI Score:         {format_percentage(report['humanization'].get('ai_score_before', 0))}")
        lines.append(f"    After AI Score:          {format_percentage(report['humanization'].get('ai_score_after', 0))}")
        lines.append(f"    Iterations:              {report['humanization'].get('iterations', 0)}")
        lines.append("")
    
    # Processing info
    if 'processing_time' in report:
        lines.append(f"  Processing Time:           {report['processing_time']:.2f}s")
    
    lines.append("=" * 60)
    
    return "\n".join(lines)
