#!/usr/bin/env python3
"""
Setup script for Genome Pathogenicity AI Predictor
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README file
README_PATH = Path(__file__).parent / "README.md"
with open(README_PATH, "r", encoding="utf-8") as f:
    long_description = f.read()

# Read requirements
REQUIREMENTS_PATH = Path(__file__).parent / "requirements.txt"
with open(REQUIREMENTS_PATH, "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f.readlines() if line.strip() and not line.startswith("#")]

setup(
    name="genome-pathogenicity-ai",
    version="1.0.0",
    author="Genome AI Team",
    author_email="contact@example.com",
    description="AI-powered application for predicting pathogenicity from genome sequences",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/genome-pathogenicity-ai",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.910",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "gpu": [
            "cupy-cuda11x>=9.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "genome-pathogenicity=src.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["config/*.yaml", "config/*.yml"],
    },
    zip_safe=False,
    keywords="bioinformatics, genomics, pathogenicity, machine learning, AI",
    project_urls={
        "Bug Reports": "https://github.com/example/genome-pathogenicity-ai/issues",
        "Source": "https://github.com/example/genome-pathogenicity-ai",
        "Documentation": "https://genome-pathogenicity-ai.readthedocs.io/",
    },
)