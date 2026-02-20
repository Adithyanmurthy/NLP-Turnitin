"""
Setup script for Content Integrity Platform
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    requirements = requirements_file.read_text().strip().split('\n')
    requirements = [r.strip() for r in requirements if r.strip() and not r.startswith('#')]

setup(
    name="content-integrity-platform",
    version="1.0.0",
    description="AI Detection, Plagiarism Detection, and Humanization Platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Team",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/content-integrity-platform",
    packages=find_packages(exclude=["tests", "tests.*"]),
    install_requires=requirements,
    python_requires=">=3.10",
    entry_points={
        'console_scripts': [
            'content-integrity=main:main',
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
    ],
    keywords="ai-detection plagiarism nlp machine-learning text-analysis",
    project_urls={
        "Documentation": "https://github.com/yourusername/content-integrity-platform",
        "Source": "https://github.com/yourusername/content-integrity-platform",
        "Bug Reports": "https://github.com/yourusername/content-integrity-platform/issues",
    },
)
