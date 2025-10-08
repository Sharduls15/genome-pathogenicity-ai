#!/usr/bin/env python3
"""
ClinVar VCF Training Script for Genome Pathogenicity AI Predictor

This script parses ClinVar VCF files and trains pathogenicity prediction models
using real clinical variant data.
"""

import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import re
import argparse

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


class ClinVarVCFParser:
    """Parser for ClinVar VCF files."""
    
    # ClinVar clinical significance mappings
    CLINVAR_SIGNIFICANCE_MAP = {
        'Benign': 'benign',
        'Likely_benign': 'likely_benign', 
        'Uncertain_significance': 'uncertain_significance',
        'Likely_pathogenic': 'likely_pathogenic',
        'Pathogenic': 'pathogenic',
        'drug_response': 'uncertain_significance',  # Map to uncertain
        'association': 'uncertain_significance',
        'risk_factor': 'uncertain_significance',
        'protective': 'likely_benign',
        'conflicting_interpretations_of_pathogenicity': 'uncertain_significance'
    }
    
    # Score mappings for pathogenicity classes
    PATHOGENICITY_SCORES = {
        'benign': 0.1,
        'likely_benign': 0.3,
        'uncertain_significance': 0.5,
        'likely_pathogenic': 0.7,
        'pathogenic': 0.9
    }
    
    def __init__(self, vcf_file: Path):
        """
        Initialize ClinVar VCF parser.
        
        Args:
            vcf_file: Path to ClinVar VCF file
        """
        self.vcf_file = vcf_file
        self.variants = []
        
    def parse_vcf(self, max_variants: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Parse ClinVar VCF file and extract variant information.
        
        Args:
            max_variants: Maximum number of variants to parse (for testing)
            
        Returns:
            List of variant dictionaries
        """
        logger.info(f"Parsing ClinVar VCF file: {self.vcf_file}")
        
        variants = []
        
        try:
            with open(self.vcf_file, 'r', encoding='utf-8') as f:
                header_parsed = False
                variant_count = 0
                
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    
                    # Skip empty lines
                    if not line:
                        continue
                    
                    # Skip header lines
                    if line.startswith('#'):
                        continue
                    
                    # Parse variant line
                    try:
                        variant = self._parse_variant_line(line)
                        if variant:
                            variants.append(variant)
                            variant_count += 1
                            
                            if variant_count % 1000 == 0:
                                logger.info(f"Parsed {variant_count} variants...")
                            
                            # Stop if max_variants reached
                            if max_variants and variant_count >= max_variants:
                                logger.info(f"Reached maximum variants limit: {max_variants}")
                                break
                                
                    except Exception as e:
                        logger.warning(f"Error parsing line {line_num}: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Error reading VCF file: {e}")
            raise
        
        logger.info(f"Successfully parsed {len(variants)} variants from ClinVar VCF")
        return variants
    
    def _parse_variant_line(self, line: str) -> Optional[Dict[str, Any]]:
        """
        Parse a single VCF variant line.
        
        Args:
            line: VCF variant line
            
        Returns:
            Variant dictionary or None if invalid
        """
        fields = line.split('\t')
        
        if len(fields) < 8:
            return None
        
        chrom = fields[0]
        pos = int(fields[1])
        variant_id = fields[2]
        ref = fields[3]
        alt = fields[4]
        info = fields[7]
        
        # Extract clinical significance from INFO field
        clinical_significance = self._extract_clinical_significance(info)
        if not clinical_significance:
            return None
        
        # Map to standardized pathogenicity class
        pathogenicity_class = self._map_clinical_significance(clinical_significance)
        if not pathogenicity_class:
            return None
        
        # Get pathogenicity score
        pathogenicity_score = self.PATHOGENICITY_SCORES.get(pathogenicity_class, 0.5)
        
        # Create variant sequence context (simple approach)
        # In practice, you'd want to extract more sequence context from reference genome
        variant_sequence = self._create_variant_sequence(ref, alt)
        
        return {
            'chromosome': chrom,
            'position': pos,
            'variant_id': variant_id,
            'reference': ref,
            'alternate': alt,
            'clinical_significance': clinical_significance,
            'pathogenicity_class': pathogenicity_class,
            'pathogenicity_score': pathogenicity_score,
            'variant_sequence': variant_sequence,
            'info': info
        }
    
    def _extract_clinical_significance(self, info: str) -> Optional[str]:
        """Extract clinical significance from VCF INFO field."""
        # Look for CLNSIG field
        clnsig_match = re.search(r'CLNSIG=([^;]+)', info)
        if clnsig_match:
            return clnsig_match.group(1)
        return None
    
    def _map_clinical_significance(self, clinical_significance: str) -> Optional[str]:
        """Map ClinVar clinical significance to standardized pathogenicity class."""
        # Handle multiple significance values (take the first mappable one)
        significances = clinical_significance.split('|')
        
        for sig in significances:
            sig = sig.strip().replace(' ', '_')
            if sig in self.CLINVAR_SIGNIFICANCE_MAP:
                return self.CLINVAR_SIGNIFICANCE_MAP[sig]
        
        # Try exact match without replacement
        if clinical_significance in self.CLINVAR_SIGNIFICANCE_MAP:
            return self.CLINVAR_SIGNIFICANCE_MAP[clinical_significance]
            
        return None
    
    def _create_variant_sequence(self, ref: str, alt: str, context_length: int = 50) -> str:
        """
        Create a sequence representation of the variant.
        This is simplified - in practice you'd extract from reference genome.
        
        Args:
            ref: Reference allele
            alt: Alternate allele
            context_length: Length of context sequence
            
        Returns:
            Variant sequence string
        """
        # Simple approach: create a sequence with the variant in the middle
        # In reality, you'd want to extract actual genomic context
        
        # Create context with ATCGs
        context_bases = ['A', 'T', 'C', 'G']
        np.random.seed(42)  # For reproducible sequences
        
        left_context = ''.join(np.random.choice(context_bases, size=context_length))
        right_context = ''.join(np.random.choice(context_bases, size=context_length))
        
        # Create sequence with alternate allele
        variant_sequence = left_context + alt + right_context
        
        return variant_sequence


def prepare_training_data(variants: List[Dict[str, Any]], 
                         max_samples_per_class: int = 1000) -> Dict[str, Any]:
    """
    Prepare training data from ClinVar variants.
    
    Args:
        variants: List of variant dictionaries
        max_samples_per_class: Maximum samples per pathogenicity class
        
    Returns:
        Training data dictionary
    """
    logger.info("Preparing training data from ClinVar variants...")
    
    # Group variants by pathogenicity class
    variants_by_class = {}
    for variant in variants:
        pathogenicity_class = variant['pathogenicity_class']
        if pathogenicity_class not in variants_by_class:
            variants_by_class[pathogenicity_class] = []
        variants_by_class[pathogenicity_class].append(variant)
    
    # Log class distribution
    for cls, cls_variants in variants_by_class.items():
        logger.info(f"Class '{cls}': {len(cls_variants)} variants")
    
    # Balance classes and extract features
    analyzer = GenomeAnalyzer()
    features_list = []
    scores_list = []
    classes_list = []
    
    for pathogenicity_class, cls_variants in variants_by_class.items():
        # Limit samples per class
        sample_variants = cls_variants[:max_samples_per_class]
        logger.info(f"Using {len(sample_variants)} variants for class '{pathogenicity_class}'")
        
        for i, variant in enumerate(sample_variants):
            if i % 100 == 0:
                logger.info(f"Processing {pathogenicity_class}: {i+1}/{len(sample_variants)}")
            
            try:
                # Extract features from variant sequence
                sequence = variant['variant_sequence']
                features = analyzer.extract_features(sequence)
                
                features_list.append(features)
                scores_list.append(variant['pathogenicity_score'])
                classes_list.append(variant['pathogenicity_class'])
                
            except Exception as e:
                logger.warning(f"Error processing variant {variant['variant_id']}: {e}")
                continue
    
    logger.info(f"Prepared training data: {len(features_list)} samples")
    
    return {
        'features': features_list,
        'scores': scores_list,
        'classes': classes_list
    }


def train_clinvar_model(vcf_file: Path, 
                       model_name: str = "clinvar_model",
                       max_variants: int = 10000,
                       max_samples_per_class: int = 1000) -> None:
    """
    Train pathogenicity prediction model from ClinVar VCF data.
    
    Args:
        vcf_file: Path to ClinVar VCF file
        model_name: Name for the trained model
        max_variants: Maximum variants to parse from VCF
        max_samples_per_class: Maximum samples per pathogenicity class
    """
    logger.info(f"Training model from ClinVar VCF: {vcf_file}")
    
    # Parse ClinVar VCF
    parser = ClinVarVCFParser(vcf_file)
    variants = parser.parse_vcf(max_variants=max_variants)
    
    if not variants:
        logger.error("No variants parsed from VCF file!")
        return
    
    # Prepare training data
    training_data = prepare_training_data(variants, max_samples_per_class)
    
    if not training_data['features']:
        logger.error("No training features extracted!")
        return
    
    # Train the model
    logger.info(f"Training model: {model_name}")
    predictor = PathogenicityPredictor(model_name=model_name)
    predictor.train_model(training_data)
    
    # Test the trained model
    logger.info("Testing trained model...")
    test_sequence = "ATGCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATC"
    analyzer = GenomeAnalyzer()
    test_features = analyzer.extract_features(test_sequence)
    
    score, classification = predictor.predict(test_features)
    confidence = predictor.get_prediction_confidence(test_features)
    
    logger.info(f"Test prediction - Score: {score:.3f}, Class: {classification}, Confidence: {confidence:.3f}")
    
    # Display model info
    model_info = predictor.get_model_info()
    logger.info(f"Model info: {model_info}")
    
    logger.info(f"Model training completed: {model_name}")


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(
        description="Train pathogenicity prediction model from ClinVar VCF data"
    )
    parser.add_argument(
        "--vcf", "-v",
        type=Path,
        required=True,
        help="Path to ClinVar VCF file"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="clinvar_model",
        help="Name for the trained model (default: clinvar_model)"
    )
    parser.add_argument(
        "--max-variants",
        type=int,
        default=10000,
        help="Maximum variants to parse from VCF (default: 10000)"
    )
    parser.add_argument(
        "--max-samples-per-class",
        type=int,
        default=1000,
        help="Maximum samples per pathogenicity class (default: 1000)"
    )
    parser.add_argument(
        "--verbose", "-V",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check if VCF file exists
    if not args.vcf.exists():
        logger.error(f"VCF file not found: {args.vcf}")
        sys.exit(1)
    
    logger.info("Starting ClinVar VCF model training")
    
    try:
        train_clinvar_model(
            vcf_file=args.vcf,
            model_name=args.model,
            max_variants=args.max_variants,
            max_samples_per_class=args.max_samples_per_class
        )
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()