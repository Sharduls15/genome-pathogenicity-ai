#!/usr/bin/env python3
"""
Pathogenicity Prediction Module

This module implements machine learning models for predicting pathogenicity
scores and classification types from genomic features.
"""

import logging
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional, Union
import joblib

# Import ML libraries (with fallbacks for when they're not installed)
try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import accuracy_score, classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

logger = logging.getLogger(__name__)


class PathogenicityPredictor:
    """
    Predicts pathogenicity scores and types from genomic features.
    """
    
    # Standard pathogenicity classifications
    PATHOGENICITY_CLASSES = [
        'benign',
        'likely_benign', 
        'uncertain_significance',
        'likely_pathogenic',
        'pathogenic'
    ]
    
    def __init__(self, model_name: str = 'default', models_dir: Optional[str] = None):
        """
        Initialize the pathogenicity predictor.
        
        Args:
            model_name: Name of the model to use
            models_dir: Directory containing model files
        """
        self.model_name = model_name
        self.models_dir = Path(models_dir) if models_dir else Path(__file__).parent.parent / 'models'
        
        # Initialize model components
        self.score_model = None
        self.classification_model = None
        self.feature_scaler = None
        self.feature_names = None
        
        # Model metadata
        self.model_info = {
            'name': model_name,
            'version': '1.0.0',
            'description': 'Genome pathogenicity prediction model',
            'trained': False
        }
        
        # Try to load existing model
        self._load_model()
        
        # If no model exists, create a default one
        if not self.model_info['trained']:
            self._create_default_model()
    
    def _load_model(self) -> bool:
        """
        Load a trained model from disk.
        
        Returns:
            True if model was loaded successfully, False otherwise
        """
        model_path = self.models_dir / f"{self.model_name}_model.pkl"
        
        if not model_path.exists():
            logger.debug(f"No existing model found at {model_path}")
            return False
        
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.score_model = model_data.get('score_model')
            self.classification_model = model_data.get('classification_model')
            self.feature_scaler = model_data.get('feature_scaler')
            self.feature_names = model_data.get('feature_names')
            self.model_info.update(model_data.get('model_info', {}))
            
            logger.info(f"Loaded model: {self.model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}")
            return False
    
    def _save_model(self) -> None:
        """Save the current model to disk."""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        model_path = self.models_dir / f"{self.model_name}_model.pkl"
        
        model_data = {
            'score_model': self.score_model,
            'classification_model': self.classification_model,
            'feature_scaler': self.feature_scaler,
            'feature_names': self.feature_names,
            'model_info': self.model_info
        }
        
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            logger.info(f"Saved model to {model_path}")
        except Exception as e:
            logger.error(f"Error saving model to {model_path}: {e}")
    
    def _create_default_model(self) -> None:
        """Create a default model when no trained model exists."""
        logger.info("Creating default pathogenicity prediction model")
        
        if SKLEARN_AVAILABLE:
            # Create simple sklearn models as placeholders
            self.score_model = RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
            
            self.classification_model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
            
            self.feature_scaler = StandardScaler()
            
            # Create some dummy training data for the default model
            self._create_dummy_training_data()
            
        else:
            logger.warning("scikit-learn not available. Using simple rule-based model.")
            self.score_model = None
            self.classification_model = None
            self.feature_scaler = None
    
    def _create_dummy_training_data(self) -> None:
        """Create dummy training data to initialize the default model."""
        logger.debug("Creating dummy training data for model initialization")
        
        # Generate synthetic feature data
        n_samples = 1000
        n_features = 50
        
        np.random.seed(42)
        X_dummy = np.random.randn(n_samples, n_features)
        
        # Generate synthetic pathogenicity scores (0-1 range)
        y_scores = np.random.beta(2, 2, n_samples)  # Beta distribution for realistic scores
        
        # Generate synthetic classifications based on scores
        y_classes = []
        for score in y_scores:
            if score < 0.1:
                y_classes.append('benign')
            elif score < 0.3:
                y_classes.append('likely_benign')
            elif score < 0.7:
                y_classes.append('uncertain_significance')
            elif score < 0.9:
                y_classes.append('likely_pathogenic')
            else:
                y_classes.append('pathogenic')
        
        # Train the models
        self.feature_scaler.fit(X_dummy)
        X_scaled = self.feature_scaler.transform(X_dummy)
        
        self.score_model.fit(X_scaled, y_scores)
        self.classification_model.fit(X_scaled, y_classes)
        
        # Store feature names (generic for dummy data)
        self.feature_names = [f'feature_{i}' for i in range(n_features)]
        
        self.model_info.update({
            'trained': True,
            'training_samples': n_samples,
            'features': n_features,
            'model_type': 'dummy_random_forest'
        })
        
        logger.info("Default model initialized with dummy training data")
    
    def _prepare_features(self, features: Dict[str, float]) -> np.ndarray:
        """
        Prepare features for prediction.
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            Numpy array of prepared features
        """
        if self.feature_names is None:
            # If no feature names stored, use current features
            self.feature_names = sorted(features.keys())
            logger.debug(f"Using {len(self.feature_names)} features for prediction")
        
        # Create feature vector in correct order
        feature_vector = []
        for feature_name in self.feature_names:
            value = features.get(feature_name, 0.0)  # Default to 0 if missing
            feature_vector.append(value)
        
        # Convert to numpy array and reshape
        X = np.array(feature_vector).reshape(1, -1)
        
        # Scale features if scaler is available
        if self.feature_scaler is not None:
            X = self.feature_scaler.transform(X)
        
        return X
    
    def _rule_based_prediction(self, features: Dict[str, float]) -> Tuple[float, str]:
        """
        Simple rule-based prediction when ML models are not available.
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            Tuple of (pathogenicity_score, pathogenicity_type)
        """
        # Simple heuristic based on common genomic features
        score = 0.5  # Default neutral score
        
        # Adjust score based on available features
        if 'gc_content' in features:
            # Very high or very low GC content might indicate pathogenicity
            gc_content = features['gc_content']
            if gc_content < 0.3 or gc_content > 0.7:
                score += 0.1
        
        if 'complexity' in features:
            # Low complexity sequences might be more pathogenic
            complexity = features['complexity']
            if complexity < 0.5:
                score += 0.15
        
        if 'entropy' in features:
            # Very low entropy might indicate pathogenicity
            entropy = features['entropy']
            if entropy < 1.5:
                score += 0.1
        
        # Check for ORF features
        if 'num_orfs' in features and features['num_orfs'] == 0:
            score += 0.1  # No ORFs might indicate disruption
        
        # Check for stop codons in reading frames
        for frame in range(3):
            stop_codon_feature = f'frame_{frame}_stop_codons'
            if stop_codon_feature in features and features[stop_codon_feature] > 0:
                score += 0.05  # Premature stop codons
        
        # Ensure score is in valid range
        score = max(0.0, min(1.0, score))\n        
        # Classify based on score
        if score < 0.2:
            classification = 'benign'
        elif score < 0.4:
            classification = 'likely_benign'\n        elif score < 0.6:
            classification = 'uncertain_significance'
        elif score < 0.8:
            classification = 'likely_pathogenic'
        else:
            classification = 'pathogenic'
        
        return score, classification
    
    def predict(self, features: Dict[str, float], threshold: float = 0.5) -> Tuple[float, str]:
        """
        Predict pathogenicity score and classification.
        
        Args:
            features: Dictionary of genomic features
            threshold: Classification threshold (not used for multi-class)
            
        Returns:
            Tuple of (pathogenicity_score, pathogenicity_type)
        """
        try:
            # Use ML models if available and trained
            if (self.score_model is not None and 
                self.classification_model is not None and 
                self.model_info['trained']):
                
                X = self._prepare_features(features)
                
                # Predict pathogenicity score
                pathogenicity_score = float(self.score_model.predict(X)[0])
                pathogenicity_score = max(0.0, min(1.0, pathogenicity_score))  # Clamp to [0,1]
                
                # Predict pathogenicity classification
                pathogenicity_type = self.classification_model.predict(X)[0]
                
                return pathogenicity_score, pathogenicity_type
            
            else:
                # Fall back to rule-based prediction
                logger.debug("Using rule-based prediction")
                return self._rule_based_prediction(features)
                
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            # Return default values on error
            return 0.5, 'uncertain_significance'
    
    def get_prediction_confidence(self, features: Dict[str, float]) -> float:
        """
        Get confidence score for the prediction.
        
        Args:
            features: Dictionary of genomic features
            
        Returns:
            Confidence score between 0 and 1
        """
        try:
            if (self.classification_model is not None and 
                hasattr(self.classification_model, 'predict_proba')):
                
                X = self._prepare_features(features)
                probabilities = self.classification_model.predict_proba(X)[0]
                confidence = float(np.max(probabilities))
                return confidence
            else:
                # Return moderate confidence for rule-based predictions
                return 0.7
                
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5
    
    def get_detailed_prediction(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Get detailed prediction information.
        
        Args:
            features: Dictionary of genomic features
            
        Returns:
            Dictionary with detailed prediction results
        """
        try:
            detailed_results = {}
            
            if (self.classification_model is not None and 
                hasattr(self.classification_model, 'predict_proba')):
                
                X = self._prepare_features(features)
                probabilities = self.classification_model.predict_proba(X)[0]
                
                # Get class probabilities
                class_names = self.classification_model.classes_
                for class_name, prob in zip(class_names, probabilities):
                    detailed_results[f'prob_{class_name}'] = float(prob)
                
                # Get feature importance if available
                if hasattr(self.classification_model, 'feature_importances_'):
                    importances = self.classification_model.feature_importances_
                    top_features = sorted(
                        zip(self.feature_names, importances), 
                        key=lambda x: x[1], 
                        reverse=True
                    )[:10]  # Top 10 features
                    
                    detailed_results['top_features'] = [
                        {'feature': name, 'importance': float(importance)} 
                        for name, importance in top_features
                    ]
            
            return detailed_results
            
        except Exception as e:
            logger.error(f"Error getting detailed prediction: {e}")
            return {}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return self.model_info.copy()
    
    def train_model(self, training_data: Dict[str, Any]) -> None:
        """
        Train the pathogenicity prediction model.
        
        Args:
            training_data: Dictionary containing training features and labels
        """
        logger.info("Training pathogenicity prediction model")
        
        if not SKLEARN_AVAILABLE:
            logger.error("scikit-learn not available for model training")
            return
        
        try:
            features = training_data['features']  # List of feature dictionaries
            scores = training_data['scores']      # List of pathogenicity scores
            classes = training_data['classes']    # List of pathogenicity classes
            
            # Convert features to matrix
            if not features:
                raise ValueError("No training features provided")
            
            # Get feature names from first sample
            self.feature_names = sorted(features[0].keys())
            
            # Create feature matrix
            X = []
            for feature_dict in features:
                feature_vector = [feature_dict.get(name, 0.0) for name in self.feature_names]
                X.append(feature_vector)
            
            X = np.array(X)
            y_scores = np.array(scores)
            y_classes = np.array(classes)
            
            # Scale features
            self.feature_scaler = StandardScaler()
            X_scaled = self.feature_scaler.fit_transform(X)
            
            # Train regression model for scores
            self.score_model = RandomForestRegressor(
                n_estimators=200,
                random_state=42,
                n_jobs=-1
            )
            self.score_model.fit(X_scaled, y_scores)
            
            # Train classification model
            self.classification_model = RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
            self.classification_model.fit(X_scaled, y_classes)
            
            # Update model info
            self.model_info.update({
                'trained': True,
                'training_samples': len(features),
                'features': len(self.feature_names),
                'model_type': 'random_forest',
                'classes': list(set(y_classes))
            })
            
            # Evaluate model performance
            score_cv = cross_val_score(self.score_model, X_scaled, y_scores, cv=5)
            class_cv = cross_val_score(self.classification_model, X_scaled, y_classes, cv=5)
            
            self.model_info.update({
                'score_cv_mean': float(np.mean(score_cv)),
                'score_cv_std': float(np.std(score_cv)),
                'classification_cv_mean': float(np.mean(class_cv)),
                'classification_cv_std': float(np.std(class_cv))
            })
            
            # Save the trained model
            self._save_model()
            
            logger.info(f"Model training completed. CV scores: {np.mean(score_cv):.3f} ± {np.std(score_cv):.3f} (regression), {np.mean(class_cv):.3f} ± {np.std(class_cv):.3f} (classification)")
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise