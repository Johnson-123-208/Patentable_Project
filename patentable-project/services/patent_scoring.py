"""
Patent scoring service for computing patentability scores based on novelty features.
"""

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging
from sklearn.preprocessing import StandardScaler

# Import local modules
try:
    from services.vector_index import VectorIndex, query_index
except ImportError:
    from vector_index import VectorIndex, query_index

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PatentScorer:
    def __init__(self, model_path: str = 'models/patentability_xgb.pkl', 
                 index_path: str = 'models/patent_index'):
        """
        Initialize patent scoring service.
        
        Args:
            model_path: Path to trained patentability model
            index_path: Path to vector index
        """
        self.model_path = model_path
        self.index_path = index_path
        self.model = None
        self.scaler = None
        self.vector_index = None
        
    def load_model(self):
        """Load trained patentability model."""
        if self.model is None:
            model_file = Path(self.model_path)
            if model_file.exists():
                logger.info(f"Loading patentability model from: {self.model_path}")
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    if isinstance(model_data, dict):
                        self.model = model_data.get('model')
                        self.scaler = model_data.get('scaler')
                    else:
                        self.model = model_data
                        self.scaler = StandardScaler()  # Default scaler
            else:
                logger.warning(f"Model file not found: {self.model_path}")
                raise FileNotFoundError(f"Model not found at {self.model_path}")
    
    def load_vector_index(self):
        """Load vector index for similarity computation."""
        if self.vector_index is None:
            self.vector_index = VectorIndex()
            try:
                self.vector_index.load_index(self.index_path)
            except FileNotFoundError:
                logger.warning(f"Vector index not found at {self.index_path}")
                raise FileNotFoundError(f"Vector index not found. Run build_index() first.")
    
    def extract_ipc_codes(self, text: str) -> List[str]:
        """Extract potential IPC codes from project text."""
        import re
        # Simple pattern matching for IPC-like codes in text
        ipc_pattern = r'\b[A-H]\d{2}[A-Z]\d+/\d+\b'
        matches = re.findall(ipc_pattern, text.upper())
        return list(set(matches))  # Remove duplicates
    
    def compute_ipc_overlap(self, project_text: str, patent_ipc_codes: List[str]) -> float:
        """Compute IPC code overlap between project and patents."""
        if not patent_ipc_codes:
            return 0.0
        
        project_ipc = self.extract_ipc_codes(project_text)
        if not project_ipc:
            return 0.0
        
        # Flatten patent IPC codes
        all_patent_ipc = []
        for patent_codes in patent_ipc_codes:
            if patent_codes:
                codes = patent_codes.split(';')
                all_patent_ipc.extend([code.strip() for code in codes])
        
        # Compute overlap
        project_set = set(project_ipc)
        patent_set = set(all_patent_ipc)
        
        if not patent_set:
            return 0.0
        
        intersection = project_set.intersection(patent_set)
        overlap_ratio = len(intersection) / len(patent_set)
        
        return overlap_ratio
    
    def compute_novelty_features(self, project_text: str, k: int = 10) -> Dict[str, float]:
        """
        Compute novelty features for a project.
        
        Args:
            project_text: Project description text
            k: Number of similar patents to consider
            
        Returns:
            Dictionary of novelty features
        """
        # Load vector index if not already loaded
        if self.vector_index is None:
            self.load_vector_index()
        
        # Query similar patents
        similar_patents = self.vector_index.query_index(project_text, k=k)
        
        if not similar_patents:
            # No similar patents found - high novelty
            return {
                'min_cosine_similarity': 0.0,
                'avg_cosine_similarity': 0.0,
                'max_cosine_similarity': 0.0,
                'std_cosine_similarity': 0.0,
                'ipc_overlap_ratio': 0.0,
                'num_similar_patents': 0,
                'top3_avg_similarity': 0.0,
                'similarity_variance': 0.0,
                'novelty_score': 1.0
            }
        
        # Extract similarity scores
        similarities = [patent['similarity_score'] for patent in similar_patents]
        
        # Extract IPC codes
        ipc_codes = [patent['ipc_codes'] for patent in similar_patents]
        
        # Compute features
        features = {
            'min_cosine_similarity': min(similarities),
            'avg_cosine_similarity': np.mean(similarities),
            'max_cosine_similarity': max(similarities),
            'std_cosine_similarity': np.std(similarities),
            'ipc_overlap_ratio': self.compute_ipc_overlap(project_text, ipc_codes),
            'num_similar_patents': len(similar_patents),
            'top3_avg_similarity': np.mean(similarities[:3]) if len(similarities) >= 3 else np.mean(similarities),
            'similarity_variance': np.var(similarities),
            'novelty_score': 1.0 - max(similarities)  # Simple novelty score
        }
        
        return features
    
    def compute_additional_features(self, project_text: str) -> Dict[str, float]:
        """Compute additional text-based features."""
        words = project_text.split()
        
        # Technical keywords that might indicate patentability
        tech_keywords = [
            'algorithm', 'method', 'system', 'device', 'apparatus', 'process',
            'technique', 'mechanism', 'circuit', 'software', 'hardware',
            'invention', 'novel', 'unique', 'innovative', 'improved', 'enhanced'
        ]
        
        tech_word_count = sum(1 for word in words if word.lower() in tech_keywords)
        
        features = {
            'text_length': len(project_text),
            'word_count': len(words),
            'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
            'tech_word_ratio': tech_word_count / len(words) if words else 0,
            'sentence_count': project_text.count('.') + project_text.count('!') + project_text.count('?'),
        }
        
        return features
    
    def predict_patentability(self, project_text: str) -> Dict[str, Any]:
        """
        Predict patentability score and label for a project.
        
        Args:
            project_text: Project description text
            
        Returns:
            Dictionary containing patentability_score (0-100) and label
        """
        try:
            # Load model if not already loaded
            if self.model is None:
                self.load_model()
            
            # Compute features
            novelty_features = self.compute_novelty_features(project_text)
            additional_features = self.compute_additional_features(project_text)
            
            # Combine all features
            all_features = {**novelty_features, **additional_features}
            
            # Convert to feature vector (ensure consistent ordering)
            feature_names = [
                'min_cosine_similarity', 'avg_cosine_similarity', 'max_cosine_similarity',
                'std_cosine_similarity', 'ipc_overlap_ratio', 'num_similar_patents',
                'top3_avg_similarity', 'similarity_variance', 'novelty_score',
                'text_length', 'word_count', 'avg_word_length', 'tech_word_ratio',
                'sentence_count'
            ]
            
            feature_vector = np.array([[all_features.get(name, 0.0) for name in feature_names]])
            
            # Scale features if scaler is available
            if self.scaler is not None:
                feature_vector = self.scaler.transform(feature_vector)
            
            # Make prediction
            if hasattr(self.model, 'predict_proba'):
                # For classifiers, get probability of being patentable
                probabilities = self.model.predict_proba(feature_vector)[0]
                if len(probabilities) == 2:  # Binary classification
                    patentability_score = probabilities[1] * 100  # Probability of positive class
                else:
                    patentability_score = max(probabilities) * 100
            else:
                # For regressors, direct prediction
                raw_prediction = self.model.predict(feature_vector)[0]
                patentability_score = max(0, min(100, raw_prediction * 100))  # Clamp to 0-100
            
            # Determine label based on score
            if patentability_score >= 70:
                label = "Highly Patentable"
            elif patentability_score >= 50:
                label = "Potentially Patentable"
            elif patentability_score >= 30:
                label = "Low Patentability"
            else:
                label = "Not Patentable"
            
            result = {
                'patentability_score': round(patentability_score, 2),
                'label': label,
                'features': all_features,
                'similar_patents': self.vector_index.query_index(project_text, k=5) if self.vector_index else []
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in patentability prediction: {e}")
            # Return default values in case of error
            return {
                'patentability_score': 50.0,
                'label': "Unknown",
                'features': {},
                'similar_patents': [],
                'error': str(e)
            }
    
    def batch_predict(self, project_texts: List[str]) -> List[Dict[str, Any]]:
        """Predict patentability for multiple projects."""
        results = []
        for i, text in enumerate(project_texts):
            logger.info(f"Processing project {i+1}/{len(project_texts)}")
            result = self.predict_patentability(text)
            results.append(result)
        return results


# Convenience functions
def compute_novelty_features(project_text: str, k: int = 10, 
                           index_path: str = 'models/patent_index') -> Dict[str, float]:
    """Compute novelty features for a project."""
    scorer = PatentScorer(index_path=index_path)
    return scorer.compute_novelty_features(project_text, k)


def predict_patentability(project_text: str, 
                         model_path: str = 'models/patentability_xgb.pkl',
                         index_path: str = 'models/patent_index') -> Dict[str, Any]:
    """Predict patentability score and label."""
    scorer = PatentScorer(model_path, index_path)
    return scorer.predict_patentability(project_text)


# Example usage and testing
if __name__ == "__main__":
    try:
        # Example project text
        sample_project = """
        A machine learning algorithm for real-time image recognition using convolutional neural networks.
        The system can identify objects in images with high accuracy and process video streams efficiently.
        The novel approach combines transfer learning with custom optimization techniques.
        """
        
        print("Computing novelty features...")
        features = compute_novelty_features(sample_project)
        print("Novelty features:")
        for key, value in features.items():
            print(f"  {key}: {value:.4f}")
        
        print("\nPredicting patentability...")
        result = predict_patentability(sample_project)
        print(f"Patentability Score: {result['patentability_score']}")
        print(f"Label: {result['label']}")
        
        if result.get('similar_patents'):
            print(f"\nTop similar patents:")
            for patent in result['similar_patents'][:3]:
                print(f"  - {patent['title']} (similarity: {patent['similarity_score']:.3f})")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to run the training script and build the vector index first.")