#!/usr/bin/env python3
"""
Train baseline XGBoost model for patentability prediction.
Creates dummy labeled dataset and trains model.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
import logging

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from services.patent_scoring import PatentScorer
    from services.vector_index import VectorIndex
except ImportError:
    print("Could not import services. Make sure you're running from project root.")
    sys.exit(1)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PatentabilityTrainer:
    def __init__(self, random_state: int = 42):
        """Initialize trainer with random state for reproducibility."""
        self.random_state = random_state
        self.model = None
        self.scaler = None
        
    def generate_dummy_dataset(self, n_samples: int = 1000) -> pd.DataFrame:
        """
        Generate dummy labeled dataset for training.
        This simulates real patent data with various characteristics.
        """
        np.random.seed(self.random_state)
        
        # Generate synthetic features similar to what PatentScorer computes
        data = []
        
        for i in range(n_samples):
            # Simulate similarity features (higher similarity = lower patentability)
            min_sim = np.random.beta(2, 5)  # Skewed toward lower values
            avg_sim = min_sim + np.random.beta(2, 3) * (0.8 - min_sim)
            max_sim = avg_sim + np.random.beta(2, 3) * (1.0 - avg_sim)
            std_sim = np.random.uniform(0.05, 0.3)
            
            # IPC overlap (higher overlap = lower patentability)
            ipc_overlap = np.random.beta(1, 4)  # Skewed toward lower values
            
            # Number of similar patents
            num_similar = np.random.poisson(5) + 1
            
            # Top 3 average similarity
            top3_avg = max_sim * np.random.uniform(0.8, 1.0)
            
            # Similarity variance
            sim_var = std_sim ** 2
            
            # Novelty score (inverse of max similarity)
            novelty_score = 1.0 - max_sim
            
            # Text features
            text_length = np.random.lognormal(6, 1)  # Log-normal distribution
            word_count = text_length / 6  # Rough approximation
            avg_word_length = np.random.normal(5.5, 1.5)
            tech_word_ratio = np.random.beta(2, 5)  # Tech words ratio
            sentence_count = max(1, int(word_count / 15))  # Rough sentences
            
            # Create patentability label based on features
            # Lower similarity and higher novelty = more patentable
            patentability_prob = (
                0.3 * novelty_score +  # High novelty is good
                0.2 * (1 - ipc_overlap) +  # Low IPC overlap is good
                0.2 * np.clip(tech_word_ratio * 2, 0, 1) +  # Tech words help
                0.15 * np.clip((text_length - 500) / 1000, 0, 1) +  # Detailed description helps
                0.15 * np.random.random()  # Some randomness
            )
            
            # Add some noise and threshold
            patentability_prob += np.random.normal(0, 0.1)
            is_patentable = patentability_prob > 0.5
            
            sample = {
                'min_cosine_similarity': min_sim,
                'avg_cosine_similarity': avg_sim,
                'max_cosine_similarity': max_sim,
                'std_cosine_similarity': std_sim,
                'ipc_overlap_ratio': ipc_overlap,
                'num_similar_patents': num_similar,
                'top3_avg_similarity': top3_avg,
                'similarity_variance': sim_var,
                'novelty_score': novelty_score,
                'text_length': text_length,
                'word_count': word_count,
                'avg_word_length': avg_word_length,
                'tech_word_ratio': tech_word_ratio,
                'sentence_count': sentence_count,
                'is_patentable': int(is_patentable),
                'patentability_score': patentability_prob * 100
            }
            
            data.append(sample)
        
        df = pd.DataFrame(data)
        
        logger.info(f"Generated {len(df)} samples")
        logger.info(f"Patentable samples: {df['is_patentable'].sum()} ({df['is_patentable'].mean():.2%})")
        
        return df
    
    def prepare_features(self, df: pd.DataFrame):
        """Prepare feature matrix and target vector."""
        feature_columns = [
            'min_cosine_similarity', 'avg_cosine_similarity', 'max_cosine_similarity',
            'std_cosine_similarity', 'ipc_overlap_ratio', 'num_similar_patents',
            'top3_avg_similarity', 'similarity_variance', 'novelty_score',
            'text_length', 'word_count', 'avg_word_length', 'tech_word_ratio',
            'sentence_count'
        ]
        
        X = df[feature_columns].values
        y = df['is_patentable'].values
        
        return X, y, feature_columns
    
    def train_model(self, X_train, y_train, X_val, y_val):
        """Train XGBoost model."""
        logger.info("Training XGBoost model...")
        
        # XGBoost parameters
        params = {
            'objective': 'binary:logistic',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': self.random_state
        }
        
        # Create and train model
        self.model = xgb.XGBClassifier(**params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=10,
            verbose=False
        )
        
        logger.info("Model training completed")
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate trained model."""
        logger.info("Evaluating model...")
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Test Accuracy: {accuracy:.3f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                  target_names=['Not Patentable', 'Patentable']))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        # Feature importance
        feature_importance = self.model.feature_importances_
        return accuracy, feature_importance
    
    def save_model(self, model_path: str = 'models/patentability_xgb.pkl'):
        """Save trained model and scaler."""
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': [
                'min_cosine_similarity', 'avg_cosine_similarity', 'max_cosine_similarity',
                'std_cosine_similarity', 'ipc_overlap_ratio', 'num_similar_patents',
                'top3_avg_similarity', 'similarity_variance', 'novelty_score',
                'text_length', 'word_count', 'avg_word_length', 'tech_word_ratio',
                'sentence_count'
            ]
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to: {model_path}")
    
    def full_training_pipeline(self, n_samples: int = 1000, test_size: float = 0.2):
        """Complete training pipeline."""
        logger.info("Starting full training pipeline...")
        
        # Generate dataset
        df = self.generate_dummy_dataset(n_samples)
        
        # Prepare features
        X, y, feature_names = self.prepare_features(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=self.random_state, stratify=y_train
        )
        
        # Scale features
        logger.info("Scaling features...")
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.train_model(X_train_scaled, y_train, X_val_scaled, y_val)
        
        # Evaluate model
        accuracy, feature_importance = self.evaluate_model(X_test_scaled, y_test)
        
        # Print feature importance
        print("\nFeature Importance:")
        for name, importance in zip(feature_names, feature_importance):
            print(f"  {name}: {importance:.3f}")
        
        # Cross-validation
        logger.info("Performing cross-validation...")
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        logger.info(f"CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Save model
        self.save_model()
        
        logger.info("Training pipeline completed successfully!")
        
        return {
            'test_accuracy': accuracy,
            'cv_accuracy_mean': cv_scores.mean(),
            'cv_accuracy_std': cv_scores.std(),
            'feature_importance': dict(zip(feature_names, feature_importance))
        }


def main():
    """Main training function."""
    trainer = PatentabilityTrainer(random_state=42)
    
    # Run full training pipeline
    results = trainer.full_training_pipeline(n_samples=2000)
    
    print(f"\nTraining Results:")
    print(f"Test Accuracy: {results['test_accuracy']:.3f}")
    print(f"CV Accuracy: {results['cv_accuracy_mean']:.3f} (+/- {results['cv_accuracy_std'] * 2:.3f})")
    
    print(f"\nTop 5 Most Important Features:")
    sorted_features = sorted(results['feature_importance'].items(), key=lambda x: x[1], reverse=True)
    for name, importance in sorted_features[:5]:
        print(f"  {name}: {importance:.3f}")


if __name__ == "__main__":
    main()