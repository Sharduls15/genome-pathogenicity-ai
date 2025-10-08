#!/usr/bin/env python3
"""
Main application entry point for Genome Pathogenicity AI Predictor.

This module provides command-line interface and main application logic
for analyzing genome sequences and predicting pathogenicity.
"""

import argparse
import sys
from pathlib import Path
import logging
from typing import Optional, Dict, Any

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from genome_analyzer import GenomeAnalyzer
from pathogenicity_predictor import PathogenicityPredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="AI-powered genome pathogenicity prediction",
        epilog="Example: python main.py --sequence ATCGATCG... --format fasta"
    )
    
    parser.add_argument(
        "--sequence", "-s",
        type=str,
        help="Raw genome sequence string"
    )
    
    parser.add_argument(
        "--file", "-f",
        type=Path,
        help="Path to sequence file (FASTA, FASTQ, or plain text)"
    )
    
    parser.add_argument(
        "--format",
        choices=["fasta", "fastq", "text"],
        default="fasta",
        help="Input sequence format (default: fasta)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output file path for results (default: stdout)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="default",
        help="Model name to use for prediction (default: default)"
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Pathogenicity threshold for classification (default: 0.5)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process multiple sequences in batch mode"
    )
    
    return parser.parse_args()


def load_sequence_from_file(file_path: Path, format_type: str) -> str:
    """Load sequence from file based on format type."""
    try:
        with open(file_path, 'r') as f:
            content = f.read().strip()
            
        if format_type == "fasta":
            # Extract sequence from FASTA format (skip header lines)
            lines = content.split('\n')
            sequence = ''.join([line for line in lines if not line.startswith('>')])
            return sequence
            
        elif format_type == "fastq":
            # Extract sequence from FASTQ format (every 4th line starting from line 2)
            lines = content.split('\n')
            sequences = [lines[i] for i in range(1, len(lines), 4) if i < len(lines)]
            return ''.join(sequences)
            
        else:  # text format
            # Remove whitespace and newlines
            return ''.join(content.split())
            
    except Exception as e:
        logger.error(f"Error loading sequence from file {file_path}: {e}")
        raise


def format_results(results: Dict[str, Any], verbose: bool = False) -> str:
    """Format prediction results for output."""
    output_lines = []
    
    output_lines.append("=== Genome Pathogenicity Prediction Results ===")
    output_lines.append("")
    
    # Basic results
    output_lines.append(f"Pathogenicity Score: {results['pathogenicity_score']:.4f}")
    output_lines.append(f"Pathogenicity Type: {results['pathogenicity_type']}")
    output_lines.append(f"Confidence: {results.get('confidence', 'N/A')}")
    output_lines.append("")
    
    # Sequence info
    if 'sequence_info' in results:
        seq_info = results['sequence_info']
        output_lines.append("Sequence Information:")
        output_lines.append(f"  Length: {seq_info.get('length', 'N/A')} bp")
        output_lines.append(f"  GC Content: {seq_info.get('gc_content', 'N/A'):.2%}")
        output_lines.append("")
    
    # Detailed results if verbose
    if verbose and 'detailed_results' in results:
        output_lines.append("Detailed Analysis:")
        for key, value in results['detailed_results'].items():
            output_lines.append(f"  {key}: {value}")
        output_lines.append("")
    
    # Model info
    if 'model_info' in results:
        model_info = results['model_info']
        output_lines.append("Model Information:")
        output_lines.append(f"  Model: {model_info.get('name', 'N/A')}")
        output_lines.append(f"  Version: {model_info.get('version', 'N/A')}")
        output_lines.append("")
    
    return '\n'.join(output_lines)


def analyze_sequence(sequence: str, args: argparse.Namespace) -> Dict[str, Any]:
    """Analyze a single genome sequence."""
    logger.info(f"Analyzing sequence of length {len(sequence)}")
    
    # Initialize components
    analyzer = GenomeAnalyzer()
    predictor = PathogenicityPredictor(model_name=args.model)
    
    # Extract features from sequence
    logger.info("Extracting genomic features...")
    features = analyzer.extract_features(sequence)
    
    # Get sequence information
    sequence_info = analyzer.get_sequence_info(sequence)
    
    # Make prediction
    logger.info("Making pathogenicity prediction...")
    pathogenicity_score, pathogenicity_type = predictor.predict(
        features, 
        threshold=args.threshold
    )
    
    # Get additional prediction details
    confidence = predictor.get_prediction_confidence(features)
    model_info = predictor.get_model_info()
    
    # Compile results
    results = {
        'pathogenicity_score': pathogenicity_score,
        'pathogenicity_type': pathogenicity_type,
        'confidence': confidence,
        'sequence_info': sequence_info,
        'model_info': model_info
    }
    
    if args.verbose:
        detailed_results = predictor.get_detailed_prediction(features)
        results['detailed_results'] = detailed_results
    
    return results


def main():
    """Main application entry point."""
    try:
        args = parse_arguments()
        
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.debug("Verbose mode enabled")
        
        # Validate arguments
        if not args.sequence and not args.file:
            logger.error("Either --sequence or --file must be provided")
            sys.exit(1)
        
        # Get sequence
        if args.file:
            logger.info(f"Loading sequence from file: {args.file}")
            sequence = load_sequence_from_file(args.file, args.format)
        else:
            sequence = args.sequence
        
        # Validate sequence
        if not sequence or len(sequence) == 0:
            logger.error("Empty sequence provided")
            sys.exit(1)
        
        # Analyze sequence
        results = analyze_sequence(sequence, args)
        
        # Format and output results
        formatted_output = format_results(results, args.verbose)
        
        if args.output:
            logger.info(f"Writing results to: {args.output}")
            with open(args.output, 'w') as f:
                f.write(formatted_output)
        else:
            print(formatted_output)
        
        logger.info("Analysis completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()