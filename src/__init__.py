"""
Genome Pathogenicity AI Predictor

An AI-powered application for analyzing genome sequences and predicting pathogenicity.
"""

__version__ = "1.0.0"
__author__ = "Genome AI Team"
__email__ = "contact@example.com"

from .genome_analyzer import GenomeAnalyzer
from .pathogenicity_predictor import PathogenicityPredictor

__all__ = [
    "GenomeAnalyzer",
    "PathogenicityPredictor",
]