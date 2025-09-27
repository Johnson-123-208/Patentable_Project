"""
Vector index service for semantic similarity search using sentence transformers and FAISS.
"""

import os
import numpy as np
import pandas as pd
import faiss
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorIndex:
    def __init__(self, model_name: str = 'all-mpnet-base-v2'):
        """Initialize vector index with sentence transformer model."""
        self.model_name = model_name
        self.model = None
        self.index = None
        self.patent_data = None
        self.embeddings = None
        
    def _load_model(self):
        """Load sentence transformer model."""
        if self.model is None:
            logger.info(f"Loading sentence transformer model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
        return self.model
    
    def build_index(self, patent_csv: str, model_name: str = None, index_path: str = 'models/patent_index'):
        """
        Build FAISS index from patent data.
        
        Args:
            patent_csv: Path to cleaned patents CSV
            model_name: Sentence transformer model name
            index_path: Path to save index and metadata
        """
        if model_name:
            self.model_name = model_name
        
        # Load model
        self.model = self._load_model()
        
        # Load patent data
        logger.info(f"Loading patent data from: {patent_csv}")
        self.patent_data = pd.read_csv(patent_csv)
        
        # Prepare texts for embedding
        texts = []
        for _, row in self.patent_data.iterrows():
            # Combine title, abstract, and claims for richer embeddings
            combined_text = f"{row.get('title', '')} {row.get('abstract', '')} {row.get('claims', '')}"
            texts.append(combined_text.strip())
        
        logger.info(f"Generating embeddings for {len(texts)} patents...")
        
        # Generate embeddings
        self.embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Build FAISS index
        logger.info("Building FAISS index...")
        dimension = self.embeddings.shape[1]
        
        # Use IndexFlatIP for cosine similarity (after normalization)
        self.index = faiss.IndexFlatIP(dimension)
        
        # Normalize embeddings for cosine similarity
        normalized_embeddings = self.embeddings.copy()
        faiss.normalize_L2(normalized_embeddings)
        
        # Add to index
        self.index.add(normalized_embeddings.astype('float32'))
        
        # Save index and metadata
        self._save_index(index_path)
        
        logger.info(f"Index built successfully with {self.index.ntotal} patents")
        return self.index
    
    def _save_index(self, index_path: str):
        """Save FAISS index and metadata to disk."""
        index_path = Path(index_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(index_path.with_suffix('.index')))
        
        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'patent_data': self.patent_data,
            'embeddings': self.embeddings
        }
        
        with open(index_path.with_suffix('.pkl'), 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Index saved to: {index_path}")
    
    def load_index(self, index_path: str):
        """Load FAISS index and metadata from disk."""
        index_path = Path(index_path)
        
        # Load FAISS index
        index_file = index_path.with_suffix('.index')
        metadata_file = index_path.with_suffix('.pkl')
        
        if not index_file.exists() or not metadata_file.exists():
            raise FileNotFoundError(f"Index files not found at {index_path}")
        
        logger.info(f"Loading index from: {index_path}")
        
        self.index = faiss.read_index(str(index_file))
        
        # Load metadata
        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)
        
        self.model_name = metadata['model_name']
        self.patent_data = metadata['patent_data']
        self.embeddings = metadata['embeddings']
        
        # Load model
        self.model = self._load_model()
        
        logger.info(f"Index loaded successfully with {self.index.ntotal} patents")
    
    def query_index(self, project_text: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Query index for similar patents.
        
        Args:
            project_text: Text to search for
            k: Number of top results to return
            
        Returns:
            List of dictionaries containing patent info and similarity scores
        """
        if self.index is None:
            raise ValueError("Index not loaded. Call build_index() or load_index() first.")
        
        if self.model is None:
            self.model = self._load_model()
        
        # Generate embedding for query
        query_embedding = self.model.encode([project_text], convert_to_numpy=True)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search index
        similarities, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Prepare results
        results = []
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx == -1:  # FAISS returns -1 for empty results
                continue
                
            patent = self.patent_data.iloc[idx]
            
            result = {
                'rank': i + 1,
                'similarity_score': float(similarity),
                'patent_id': patent.get('id', ''),
                'title': patent.get('title', ''),
                'abstract': patent.get('abstract', ''),
                'claims': patent.get('claims', ''),
                'ipc_codes': patent.get('ipc_codes', ''),
                'main_ipc_class': patent.get('main_ipc_class', ''),
                'link': patent.get('link', ''),
                'combined_text': patent.get('combined_text', '')
            }
            results.append(result)
        
        return results
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single text."""
        if self.model is None:
            self.model = self._load_model()
        return self.model.encode([text], convert_to_numpy=True)[0]
    
    def compute_similarity_matrix(self, texts: List[str]) -> np.ndarray:
        """Compute pairwise similarity matrix for texts."""
        if self.model is None:
            self.model = self._load_model()
        
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        
        # Normalize embeddings
        faiss.normalize_L2(embeddings)
        
        # Compute similarity matrix
        similarity_matrix = np.dot(embeddings, embeddings.T)
        return similarity_matrix


# Convenience functions
def build_index(patent_csv: str, model_name: str = 'all-mpnet-base-v2', index_path: str = 'models/patent_index'):
    """Build and save vector index."""
    vector_index = VectorIndex(model_name)
    return vector_index.build_index(patent_csv, model_name, index_path)


def query_index(project_text: str, k: int = 5, index_path: str = 'models/patent_index') -> List[Dict[str, Any]]:
    """Query vector index for similar patents."""
    vector_index = VectorIndex()
    vector_index.load_index(index_path)
    return vector_index.query_index(project_text, k)


# Example usage and testing
if __name__ == "__main__":
    # Example: Build index from sample data
    try:
        print("Building vector index...")
        build_index('data/clean/patents.csv')
        
        # Test query
        print("\nTesting query...")
        sample_project = "machine learning algorithm for image recognition and computer vision applications"
        results = query_index(sample_project, k=3)
        
        print(f"\nTop similar patents for: '{sample_project}'")
        for result in results:
            print(f"Rank {result['rank']}: {result['similarity_score']:.3f} - {result['title']}")
            print(f"  IPC: {result['main_ipc_class']}")
            print(f"  Abstract: {result['abstract'][:150]}...")
            print()
            
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to run extract_patents.py first to create data/clean/patents.csv")