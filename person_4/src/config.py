"""
Configuration module for the Content Integrity Platform
Person 4: System configuration and settings

Paths point to the actual person_1/, person_2/, person_3/ directories
so the pipeline loads real checkpoints produced by each team member.
"""

import os
from pathlib import Path
from typing import Dict, Any
import yaml
from dataclasses import dataclass, field


# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent          # person_4/
WORKSPACE_ROOT = PROJECT_ROOT.parent                           # top-level workspace
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"

# Person-specific directories (where checkpoints actually live)
PERSON1_DIR = WORKSPACE_ROOT / "person_1"
PERSON2_DIR = WORKSPACE_ROOT / "person_2"
PERSON3_DIR = WORKSPACE_ROOT / "person_3"

# Create local directories if they don't exist
for dir_path in [DATA_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


@dataclass
class AIDetectorConfig:
    """Configuration for AI Detection module (Person 1)"""
    # Points to person_1/checkpoints where Person 1 saves trained models
    model_dir: Path = field(default_factory=lambda: PERSON1_DIR / "checkpoints")
    threshold: float = 0.5
    max_length: int = 512
    batch_size: int = 8
    device: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"


@dataclass
class PlagiarismDetectorConfig:
    """Configuration for Plagiarism Detection module (Person 2)"""
    # Points to person_2 directories where Person 2 saves models & index
    model_dir: Path = field(default_factory=lambda: PERSON2_DIR / "checkpoints")
    index_dir: Path = field(default_factory=lambda: PERSON2_DIR / "reference_index")
    similarity_threshold: float = 0.85
    lsh_threshold: float = 0.7
    max_candidates: int = 10
    device: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"


@dataclass
class HumanizerConfig:
    """Configuration for Humanization module (Person 3)"""
    # Points to person_3/checkpoints where Person 3 saves trained models
    model_dir: Path = field(default_factory=lambda: PERSON3_DIR / "checkpoints")
    target_ai_score: float = 0.05
    max_iterations: int = 10
    initial_diversity: int = 60
    initial_reorder: int = 40
    diversity_step: int = 10
    reorder_step: int = 10
    device: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"


@dataclass
class PipelineConfig:
    """Main pipeline configuration"""
    # Pipeline settings
    enable_caching: bool = True
    log_level: str = "INFO"
    
    # Processing limits
    max_text_length: int = 50000  # characters
    min_text_length: int = 10
    
    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1
    rate_limit_per_minute: int = 10
    max_request_size: int = 1024 * 1024  # 1MB
    
    def __post_init__(self):
        """Initialize sub-configs after dataclass initialization"""
        self.ai_detector = AIDetectorConfig()
        self.plagiarism_detector = PlagiarismDetectorConfig()
        self.humanizer = HumanizerConfig()
        self.cache_dir = PROJECT_ROOT / ".cache"


def load_config(config_path: str = None) -> PipelineConfig:
    """
    Load configuration from YAML file or use defaults
    
    Args:
        config_path: Path to YAML config file (optional)
    
    Returns:
        PipelineConfig instance
    """
    config = PipelineConfig()
    
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
            # Update config with YAML values
            # (simplified - in production, use proper config merging)
            for key, value in yaml_config.items():
                if hasattr(config, key):
                    setattr(config, key, value)
    
    return config


def save_config(config: PipelineConfig, config_path: str):
    """Save configuration to YAML file"""
    config_dict = {
        'ai_detector': config.ai_detector.__dict__,
        'plagiarism_detector': config.plagiarism_detector.__dict__,
        'humanizer': config.humanizer.__dict__,
        'enable_caching': config.enable_caching,
        'log_level': config.log_level,
        'max_text_length': config.max_text_length,
        'min_text_length': config.min_text_length,
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)


# Global config instance
CONFIG = load_config()
