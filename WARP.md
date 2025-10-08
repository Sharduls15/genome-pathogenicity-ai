# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Development Commands

### Environment Setup
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Running the Application
```bash
# Basic sequence analysis
python src/main.py --sequence "ATCGATCGATCG..." --format fasta

# Analyze from file
python src/main.py --file path/to/sequence.fasta --format fasta --verbose

# Save results to file
python src/main.py --file input.fasta --output results.txt --verbose

# Use specific model and threshold
python src/main.py --sequence "ATCG..." --model custom_model --threshold 0.7

# Using console script (after pip install)
genome-pathogenicity --file sequence.fasta --verbose
```

### Testing and Quality
```bash
# Run tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ --cov=src --cov-report=html

# Code formatting
black src/ tests/

# Linting
flake8 src/ tests/

# Type checking
mypy src/
```

### Model Operations
```bash
# Train a new model (requires training data in data/ directory)
python -c "
from src.pathogenicity_predictor import PathogenicityPredictor
from src.genome_analyzer import GenomeAnalyzer
# Training code would go here
"

# Test model performance
python src/main.py --sequence "ATCGATCGATCG" --verbose --model default
```

### Development Utilities
```bash
# Start Jupyter notebook for analysis
jupyter notebook notebooks/

# Run development server (if API endpoints added)
uvicorn src.api:app --host localhost --port 8000 --reload
```

## Architecture Overview

This is a Python-based bioinformatics application that predicts genomic pathogenicity using machine learning. The codebase follows a modular architecture with clear separation of concerns.

### Core Components

**GenomeAnalyzer** (`src/genome_analyzer.py`)
- Processes raw genomic sequences (FASTA, FASTQ, plain text)
- Extracts comprehensive features: k-mers, protein features, ORFs, sequence complexity
- Handles sequence cleaning, translation, and validation
- Key methods: `extract_features()`, `get_sequence_info()`, `find_orfs()`

**PathogenicityPredictor** (`src/pathogenicity_predictor.py`) 
- ML prediction engine using sklearn Random Forest models
- Dual prediction: regression (pathogenicity scores) + classification (5-class pathogenicity types)
- Falls back to rule-based prediction when ML models unavailable
- Model persistence via pickle serialization
- Key methods: `predict()`, `train_model()`, `get_prediction_confidence()`

**Main Application** (`src/main.py`)
- CLI interface with argparse for sequence analysis
- Handles multiple input formats and batch processing
- Integrates GenomeAnalyzer and PathogenicityPredictor
- Structured output formatting with verbose mode

### Data Flow Architecture

1. **Input Processing**: Raw sequences → cleaning/validation → feature extraction
2. **Feature Engineering**: Sequence → k-mers, protein features, ORFs, complexity metrics
3. **ML Pipeline**: Features → scaling → ensemble prediction (score + classification)
4. **Output Generation**: Predictions → formatted results with confidence metrics

### Configuration System

The application uses YAML-based configuration (`config/config.yaml`) for:
- Model parameters (Random Forest settings, thresholds)
- Feature extraction settings (k-mer sizes, ORF parameters)
- Performance tuning (batch sizes, memory limits)
- Output formatting and logging configuration

### Key Design Patterns

**Modular Feature Extraction**: GenomeAnalyzer uses separate methods for different feature types (basic, k-mer, protein, ORF, complexity), allowing selective feature engineering.

**Ensemble ML Approach**: Combines regression and classification models for robust pathogenicity assessment.

**Graceful Degradation**: Falls back to rule-based prediction when ML dependencies unavailable.

**Extensible Model System**: Model loading/saving infrastructure supports multiple model types and versions.

### Pathogenicity Classifications

The system uses standard clinical genomics classifications:
- `benign` 
- `likely_benign`
- `uncertain_significance` 
- `likely_pathogenic`
- `pathogenic`

### Development Notes

- Heavy use of bioinformatics libraries: Biopython, pysam, pyvcf for genomic data handling
- ML stack: scikit-learn (primary), TensorFlow/PyTorch (optional)
- Feature vectors can be high-dimensional (k-mers create many features)
- Model training requires labeled genomic variant data with known pathogenicity
- The codebase supports both development (dummy data) and production (real trained models) modes

### File Structure Context

- `src/`: Core application modules  
- `data/`: Training datasets and test sequences
- `models/`: Trained ML model files (.pkl format)
- `notebooks/`: Jupyter notebooks for analysis and experimentation
- `config/`: YAML configuration files
- `tests/`: Unit tests for all modules

### Performance Considerations

- Feature extraction is computationally intensive for long sequences
- K-mer feature explosion: 4-mers limited to top 50 to prevent memory issues
- Model inference designed for single sequences; batch processing available
- Memory usage scales with sequence length and feature complexity