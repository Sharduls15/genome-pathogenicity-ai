#!/usr/bin/env python3
"""
Training Script for Genome Pathogenicity AI Predictor

This script demonstrates how to train custom models with your own genomic data.
"""

import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from genome_analyzer import GenomeAnalyzer
from pathogenicity_predictor import PathogenicityPredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_training_data() -> Dict[str, Any]:
    """
    Create sample training data for demonstration.
    In practice, you would load this from real genomic databases.
    """
    logger.info("Creating sample training data...")
    
    # Sample genomic sequences with known pathogenicity
    training_sequences = [
        # Benign sequences (normal, well-structured)
        {
            "sequence": "ATGCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATC",
            "pathogenicity_score": 0.1,
            "pathogenicity_class": "benign"
        },
        {
            "sequence": "ATGAAACGTCGTCGTCGTCGTCGTCGTCGTCGTCGTCGTCGTCGTCGTCGTCGTCGTCGTCGTCGTCGTCGTCGTCGTCGTCGTCGTCGTCGTCGTCGTCGTCGTCGTCGTCGTCGTTAG",
            "pathogenicity_score": 0.15,
            "pathogenicity_class": "benign"
        },
        {
            "sequence": "ATGGGCGGCGGCGGCGGCGGCGGCGGCGGCGGCGGCGGCGGCGGCGGCGGCGGCGGCGGCGGCGGCGGCGGCGGCGGCGGCGGCGGCGGCGGCGGCGGCGGCGGCGGCGGCGGCTAG",
            "pathogenicity_score": 0.2,
            "pathogenicity_class": "benign"
        },
        
        # Likely benign sequences
        {
            "sequence": "ATGCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGAT",
            "pathogenicity_score": 0.25,
            "pathogenicity_class": "likely_benign"
        },
        {
            "sequence": "ATGAAACGTCGTCGTCGTCGTCGTCGTCGTCGTCGTCGTCGTCGTCGTCGTCGTCGTCGTCGTCGTCGTCGTCGTCGTCGTCGTCGTCGTCGTCGTCGTCGTCGTCGTCGTCGTCGT",
            "pathogenicity_score": 0.35,
            "pathogenicity_class": "likely_benign"
        },
        
        # Uncertain significance sequences (ambiguous features)
        {
            "sequence": "ATGCGATCGATCGATCGATCGATCGATCGATCGATCGATCNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN",
            "pathogenicity_score": 0.5,
            "pathogenicity_class": "uncertain_significance"
        },
        {
            "sequence": "ATGCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCNNN",
            "pathogenicity_score": 0.55,
            "pathogenicity_class": "uncertain_significance"
        },
        
        # Likely pathogenic sequences (some problematic features)
        {
            "sequence": "ATGCGATCGATCGATCGATCGATCGATCGATCGATCGATCTAATAATAATAATAATAATAATAATAATAATAATAATAATAATAATAATAATAATAATAATAATAATAATAATAATAG",
            "pathogenicity_score": 0.75,
            "pathogenicity_class": "likely_pathogenic"
        },
        {
            "sequence": "ATGCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCTAA",
            "pathogenicity_score": 0.8,
            "pathogenicity_class": "likely_pathogenic"
        },
        
        # Pathogenic sequences (clearly problematic)
        {
            "sequence": "ATGAAACCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC",
            "pathogenicity_score": 0.9,
            "pathogenicity_class": "pathogenic"
        },
        {
            "sequence": "ATGTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT",
            "pathogenicity_score": 0.95,
            "pathogenicity_class": "pathogenic"
        },
        {
            "sequence": "ATGAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
            "pathogenicity_score": 0.88,
            "pathogenicity_class": "pathogenic"
        },
    ]
    
    return training_sequences


def extract_features_from_sequences(sequences: List[Dict[str, Any]]) -> Tuple[List[Dict[str, float]], List[float], List[str]]:
    """
    Extract features from training sequences.
    
    Args:
        sequences: List of sequence dictionaries
        
    Returns:
        Tuple of (features_list, scores_list, classes_list)
    """
    logger.info("Extracting features from training sequences...")
    
    analyzer = GenomeAnalyzer()
    features_list = []
    scores_list = []
    classes_list = []
    
    for i, seq_data in enumerate(sequences):
        logger.info(f"Processing sequence {i+1}/{len(sequences)}")
        
        sequence = seq_data["sequence"]
        score = seq_data["pathogenicity_score"]
        pathogenicity_class = seq_data["pathogenicity_class"]
        
        # Extract features
        features = analyzer.extract_features(sequence)
        
        features_list.append(features)
        scores_list.append(score)
        classes_list.append(pathogenicity_class)
    
    logger.info(f"Extracted features for {len(sequences)} sequences")
    return features_list, scores_list, classes_list


def train_custom_model(model_name: str = "custom_trained") -> None:
    """
    Train a custom pathogenicity prediction model.
    
    Args:
        model_name: Name for the trained model
    """
    logger.info(f"Starting training for model: {model_name}")
    
    # Create or load training data
    training_sequences = create_sample_training_data()
    
    # Extract features
    features_list, scores_list, classes_list = extract_features_from_sequences(training_sequences)
    
    # Prepare training data
    training_data = {
        "features": features_list,
        "scores": scores_list,
        "classes": classes_list
    }
    
    # Initialize and train the model
    predictor = PathogenicityPredictor(model_name=model_name)
    predictor.train_model(training_data)
    
    logger.info(f"Training completed for model: {model_name}")
    
    # Test the trained model
    test_sequence = "ATGCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATC"
    analyzer = GenomeAnalyzer()
    test_features = analyzer.extract_features(test_sequence)
    
    score, classification = predictor.predict(test_features)
    confidence = predictor.get_prediction_confidence(test_features)
    
    logger.info(f"Test prediction: Score={score:.3f}, Class={classification}, Confidence={confidence:.3f}")


def evaluate_model(model_name: str, test_sequences: List[Dict[str, Any]]) -> None:
    """
    Evaluate a trained model on test data.
    
    Args:
        model_name: Name of the model to evaluate
        test_sequences: Test sequences with known labels
    """
    logger.info(f"Evaluating model: {model_name}")
    
    predictor = PathogenicityPredictor(model_name=model_name)
    analyzer = GenomeAnalyzer()
    
    correct_classifications = 0
    total_sequences = len(test_sequences)
    
    for seq_data in test_sequences:
        sequence = seq_data["sequence"]
        true_class = seq_data["pathogenicity_class"]
        
        # Get prediction
        features = analyzer.extract_features(sequence)
        score, predicted_class = predictor.predict(features)
        
        if predicted_class == true_class:
            correct_classifications += 1
        
        logger.info(f"True: {true_class}, Predicted: {predicted_class}, Score: {score:.3f}")
    
    accuracy = correct_classifications / total_sequences
    logger.info(f"Model accuracy: {accuracy:.3f} ({correct_classifications}/{total_sequences})")


def main():
    """Main training script."""
    logger.info("Starting Genome Pathogenicity AI Model Training")
    
    # Train a custom model
    train_custom_model("my_custom_model")
    
    # Create some test data (different from training)
    test_sequences = [
        {
            "sequence": "ATGCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCTAG",
            "pathogenicity_class": "benign"
        },
        {
            "sequence": "ATGAAACCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC",
            "pathogenicity_class": "pathogenic"
        }
    ]
    
    # Evaluate the model
    evaluate_model("my_custom_model", test_sequences)
    
    logger.info("Training and evaluation completed!")


if __name__ == "__main__":
    main()