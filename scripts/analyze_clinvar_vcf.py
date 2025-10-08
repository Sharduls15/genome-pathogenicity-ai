#!/usr/bin/env python3
"""
ClinVar VCF Analysis Utility

This script analyzes ClinVar VCF files to show distribution of pathogenicity classes
and help you understand your data before training.
"""

import sys
import logging
import argparse
from pathlib import Path
from collections import Counter
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def analyze_vcf(vcf_file: Path, max_lines: int = 10000) -> None:
    """
    Analyze ClinVar VCF file and show statistics.
    
    Args:
        vcf_file: Path to ClinVar VCF file
        max_lines: Maximum lines to analyze
    """
    logger.info(f"Analyzing ClinVar VCF file: {vcf_file}")
    
    # Statistics counters
    total_variants = 0
    clinical_significance_counts = Counter()
    pathogenicity_class_counts = Counter()
    chromosomes = Counter()
    variant_types = Counter()
    
    # ClinVar significance mapping
    CLINVAR_SIGNIFICANCE_MAP = {
        'Benign': 'benign',
        'Likely_benign': 'likely_benign', 
        'Uncertain_significance': 'uncertain_significance',
        'Likely_pathogenic': 'likely_pathogenic',
        'Pathogenic': 'pathogenic',
        'drug_response': 'uncertain_significance',
        'association': 'uncertain_significance',
        'risk_factor': 'uncertain_significance',
        'protective': 'likely_benign',
        'conflicting_interpretations_of_pathogenicity': 'uncertain_significance'
    }
    
    try:
        with open(vcf_file, 'r', encoding='utf-8') as f:
            line_count = 0
            
            for line in f:
                line = line.strip()
                line_count += 1
                
                # Skip empty lines and headers
                if not line or line.startswith('#'):
                    continue
                
                # Stop if max_lines reached
                if total_variants >= max_lines:
                    logger.info(f"Reached maximum analysis limit: {max_lines} variants")
                    break
                
                # Parse variant line
                fields = line.split('\t')
                if len(fields) < 8:
                    continue
                
                total_variants += 1
                
                # Extract basic variant info
                chrom = fields[0]
                ref = fields[3]
                alt = fields[4]
                info = fields[7]
                
                # Count chromosome
                chromosomes[chrom] += 1
                
                # Determine variant type
                if len(ref) == 1 and len(alt) == 1:
                    variant_types['SNV'] += 1
                elif len(ref) > len(alt):
                    variant_types['Deletion'] += 1
                elif len(ref) < len(alt):
                    variant_types['Insertion'] += 1
                else:
                    variant_types['Complex'] += 1
                
                # Extract clinical significance
                clnsig_match = re.search(r'CLNSIG=([^;]+)', info)
                if clnsig_match:
                    clinical_significance = clnsig_match.group(1)
                    clinical_significance_counts[clinical_significance] += 1
                    
                    # Map to pathogenicity class
                    significances = clinical_significance.split('|')
                    for sig in significances:
                        sig = sig.strip().replace(' ', '_')
                        if sig in CLINVAR_SIGNIFICANCE_MAP:
                            pathogenicity_class = CLINVAR_SIGNIFICANCE_MAP[sig]
                            pathogenicity_class_counts[pathogenicity_class] += 1
                            break
                
                # Progress logging
                if total_variants % 1000 == 0:
                    logger.info(f"Analyzed {total_variants} variants...")
    
    except Exception as e:
        logger.error(f"Error reading VCF file: {e}")
        return
    
    # Display results
    print("\n" + "="*60)
    print("ClinVar VCF Analysis Results")
    print("="*60)
    
    print(f"\nTotal variants analyzed: {total_variants:,}")
    
    print(f"\nTop 10 chromosomes:")
    for chrom, count in chromosomes.most_common(10):
        print(f"  {chrom}: {count:,} ({count/total_variants*100:.1f}%)")
    
    print(f"\nVariant types:")
    for var_type, count in variant_types.most_common():
        print(f"  {var_type}: {count:,} ({count/total_variants*100:.1f}%)")
    
    print(f"\nClinical significance (raw ClinVar values):")
    for sig, count in clinical_significance_counts.most_common(15):
        print(f"  {sig}: {count:,} ({count/total_variants*100:.1f}%)")
    
    print(f"\nMapped pathogenicity classes:")
    for cls, count in pathogenicity_class_counts.most_common():
        print(f"  {cls}: {count:,} ({count/total_variants*100:.1f}%)")
    
    # Training recommendations
    print(f"\n" + "="*60)
    print("Training Recommendations")
    print("="*60)
    
    total_trainable = sum(pathogenicity_class_counts.values())
    print(f"Total trainable variants: {total_trainable:,}")
    
    if total_trainable == 0:
        print("❌ No trainable variants found! Check VCF format and CLNSIG fields.")
        return
    
    # Check class balance
    min_class_size = min(pathogenicity_class_counts.values()) if pathogenicity_class_counts else 0
    max_class_size = max(pathogenicity_class_counts.values()) if pathogenicity_class_counts else 0
    
    if min_class_size == 0:
        print("⚠️  Some pathogenicity classes are missing. Consider using --max-variants to analyze more data.")
    elif max_class_size / min_class_size > 10:
        print("⚠️  Significant class imbalance detected. Consider balancing classes during training.")
    else:
        print("✅ Reasonable class distribution for training.")
    
    # Suggest training parameters
    suggested_samples_per_class = min(1000, min_class_size) if min_class_size > 0 else 100
    suggested_max_variants = min(50000, total_variants * 2)
    
    print(f"\nSuggested training parameters:")
    print(f"  --max-variants {suggested_max_variants}")
    print(f"  --max-samples-per-class {suggested_samples_per_class}")
    
    print(f"\nTo train a model with your ClinVar data, run:")
    print(f"python scripts/train_from_clinvar.py --vcf \"{vcf_file}\" --max-variants {suggested_max_variants} --max-samples-per-class {suggested_samples_per_class}")


def main():
    """Main analysis script."""
    parser = argparse.ArgumentParser(
        description="Analyze ClinVar VCF file for training preparation"
    )
    parser.add_argument(
        "vcf",
        type=Path,
        help="Path to ClinVar VCF file"
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=10000,
        help="Maximum lines to analyze (default: 10000)"
    )
    parser.add_argument(
        "--verbose", "-v",
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
    
    analyze_vcf(args.vcf, args.max_lines)


if __name__ == "__main__":
    main()