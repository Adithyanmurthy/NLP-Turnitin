"""
Content Integrity Platform - Source Package
Person 4: Main package initialization
"""

__version__ = "1.0.0"
__author__ = "Your Team"

from src.pipeline import ContentIntegrityPipeline
from src.config import PipelineConfig, load_config

__all__ = ['ContentIntegrityPipeline', 'PipelineConfig', 'load_config']
