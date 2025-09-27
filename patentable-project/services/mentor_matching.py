"""
Mentor matching service for recommending mentors based on project content.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import logging
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MentorMatcher:
    def __init__(self, model_name: str = 'all-mpnet-base-v2'):
        """
        Initialize mentor matching service.
        
        Args:
            model_name: Sentence transformer model for embeddings
        """
        self.model_name = model_name
        self.model = None
        self.mentors_df = None
        self.mentor_embeddings = None
        
    def load_model(self):
        """Load sentence transformer model."""
        if self.model is None:
            logger.info(f"Loading sentence transformer model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
        return self.model
    
    def load_mentors(self, mentors_csv: str):
        """
        Load mentor data from CSV.
        
        Args:
            mentors_csv: Path to mentors CSV file
        """
        logger.info(f"Loading mentors from: {mentors_csv}")
        
        self.mentors_df = pd.read_csv(mentors_csv)
        
        # Standardize column names
        columns_mapping = {}
        for col in self.mentors_df.columns:
            col_lower = col.lower()
            if 'name' in col_lower:
                columns_mapping[col] = 'name'
            elif 'domain' in col_lower or 'tag' in col_lower or 'expertise' in col_lower:
                columns_mapping[col] = 'domain_tags'
            elif 'bio' in col_lower or 'description' in col_lower or 'profile' in col_lower:
                columns_mapping[col] = 'bio'
        
        self.mentors_df = self.mentors_df.rename(columns=columns_mapping)
        
        # Ensure required columns exist
        required_columns = ['name', 'domain_tags', 'bio']
        for col in required_columns:
            if col not in self.mentors_df.columns:
                self.mentors_df[col] = ''
        
        # Clean and prepare text for embedding
        self.mentors_df['combined_text'] = self.mentors_df.apply(
            lambda row: self.prepare_mentor_text(row), axis=1
        )
        
        logger.info(f"Loaded {len(self.mentors_df)} mentors")
    
    def prepare_mentor_text(self, mentor_row) -> str:
        """Prepare combined text for mentor embedding."""
        name = str(mentor_row.get('name', ''))
        domain_tags = str(mentor_row.get('domain_tags', ''))
        bio = str(mentor_row.get('bio', ''))
        
        # Clean domain tags
        domain_tags = re.sub(r'[,;|]', ' ', domain_tags)
        
        # Combine texts with appropriate weighting
        # Domain tags are repeated to give them more weight
        combined = f"{name} {domain_tags} {domain_tags} {bio}"
        
        return combined.strip()
    
    def build_mentor_embeddings(self):
        """Build embeddings for all mentors."""
        if self.mentors_df is None:
            raise ValueError("Mentors data not loaded. Call load_mentors() first.")
        
        if self.model is None:
            self.load_model()
        
        logger.info("Building mentor embeddings...")
        
        mentor_texts = self.mentors_df['combined_text'].tolist()
        
        self.mentor_embeddings = self.model.encode(
            mentor_texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        logger.info("Mentor embeddings built successfully")
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract key technical terms and concepts from project text."""
        # Common technical/domain keywords
        tech_domains = [
            'machine learning', 'artificial intelligence', 'deep learning', 'neural networks',
            'computer vision', 'natural language processing', 'data science', 'robotics',
            'blockchain', 'cybersecurity', 'cloud computing', 'mobile development',
            'web development', 'software engineering', 'database', 'networking',
            'iot', 'embedded systems', 'electronics', 'mechanical engineering',
            'biomedical', 'healthcare', 'finance', 'marketing', 'business',
            'education', 'research', 'innovation', 'startup', 'product management'
        ]
        
        text_lower = text.lower()
        found_domains = []
        
        for domain in tech_domains:
            if domain in text_lower:
                found_domains.append(domain)
        
        return found_domains
    
    def compute_domain_overlap(self, project_text: str, mentor_domains: str) -> float:
        """Compute domain overlap between project and mentor expertise."""
        project_keywords = set(self.extract_keywords(project_text))
        mentor_keywords = set(self.extract_keywords(mentor_domains))
        
        if not mentor_keywords:
            return 0.0
        
        overlap = project_keywords.intersection(mentor_keywords)
        overlap_score = len(overlap) / len(mentor_keywords)
        
        return overlap_score
    
    def recommend_mentors(self, project_text: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Recommend top-k mentors for a given project.
        
        Args:
            project_text: Project description text
            k: Number of mentors to recommend
            
        Returns:
            List of recommended mentors with similarity scores
        """
        if self.mentors_df is None:
            raise ValueError("Mentors data not loaded. Call load_mentors() first.")
        
        if self.mentor_embeddings is None:
            self.build_mentor_embeddings()
        
        if self.model is None:
            self.load_model()
        
        # Generate embedding for project
        project_embedding = self.model.encode([project_text], convert_to_numpy=True)
        
        # Compute similarities
        similarities = np.dot(self.mentor_embeddings, project_embedding.T).flatten()
        
        # Normalize similarities to 0-1 range
        similarities = (similarities + 1) / 2  # Convert from [-1,1] to [0,1]
        
        # Get top-k mentors
        top_indices = np.argsort(similarities)[::-1][:k]
        
        recommendations = []
        
        for rank, idx in enumerate(top_indices):
            mentor = self.mentors_df.iloc[idx]
            similarity_score = similarities[idx]
            
            # Compute additional scoring factors
            domain_overlap = self.compute_domain_overlap(
                project_text, 
                str(mentor.get('domain_tags', ''))
            )
            
            # Combined score (weighted)
            combined_score = 0.7 * similarity_score + 0.3 * domain_overlap
            
            recommendation = {
                'rank': rank + 1,
                'name': mentor.get('name', ''),
                'domain_tags': mentor.get('domain_tags', ''),
                'bio': mentor.get('bio', ''),
                'similarity_score': round(float(similarity_score), 3),
                'domain_overlap_score': round(float(domain_overlap), 3),
                'combined_score': round(float(combined_score), 3),
                'match_keywords': list(set(self.extract_keywords(project_text)).intersection(
                    set(self.extract_keywords(str(mentor.get('domain_tags', ''))))))
            }
            
            recommendations.append(recommendation)
        
        return recommendations
    
    def get_mentor_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded mentors."""
        if self.mentors_df is None:
            return {}
        
        # Extract all domain tags
        all_tags = []
        for tags in self.mentors_df['domain_tags']:
            if pd.notna(tags):
                tag_list = re.split(r'[,;|]', str(tags))
                all_tags.extend([tag.strip().lower() for tag in tag_list if tag.strip()])
        
        # Count occurrences
        from collections import Counter
        tag_counts = Counter(all_tags)
        
        stats = {
            'total_mentors': len(self.mentors_df),
            'avg_bio_length': self.mentors_df['bio'].str.len().mean(),
            'most_common_domains': tag_counts.most_common(10),
            'unique_domains': len(tag_counts),
            'mentors_with_bio': self.mentors_df['bio'].notna().sum()
        }
        
        return stats


# Convenience functions
def recommend_mentors(project_text: str, mentors_csv: str = 'data/sample/mentors.csv', 
                     k: int = 3) -> List[Dict[str, Any]]:
    """
    Recommend mentors for a project.
    
    Args:
        project_text: Project description
        mentors_csv: Path to mentors CSV file
        k: Number of mentors to recommend
        
    Returns:
        List of recommended mentors
    """
    matcher = MentorMatcher()
    matcher.load_mentors(mentors_csv)
    return matcher.recommend_mentors(project_text, k)


def get_mentor_recommendations_batch(project_texts: List[str], 
                                   mentors_csv: str = 'data/sample/mentors.csv',
                                   k: int = 3) -> List[List[Dict[str, Any]]]:
    """Get mentor recommendations for multiple projects."""
    matcher = MentorMatcher()
    matcher.load_mentors(mentors_csv)
    
    recommendations = []
    for i, text in enumerate(project_texts):
        logger.info(f"Processing project {i+1}/{len(project_texts)} for mentor matching")
        recs = matcher.recommend_mentors(text, k)
        recommendations.append(recs)
    
    return recommendations


# Example usage and testing
if __name__ == "__main__":
    try:
        # Example project text
        sample_project = """
        Developing a machine learning system for automated medical image analysis. 
        The project focuses on using convolutional neural networks to detect anomalies 
        in X-ray images and CT scans. The goal is to assist radiologists in faster 
        and more accurate diagnosis.
        """
        
        print("Finding mentor recommendations...")
        
        # Test with sample data
        recommendations = recommend_mentors(sample_project, k=3)
        
        print(f"\nTop mentor recommendations for project:")
        print(f"'{sample_project[:100]}...'")
        print()
        
        for rec in recommendations:
            print(f"Rank {rec['rank']}: {rec['name']}")
            print(f"  Domain Tags: {rec['domain_tags']}")
            print(f"  Similarity Score: {rec['similarity_score']:.3f}")
            print(f"  Domain Overlap: {rec['domain_overlap_score']:.3f}")
            print(f"  Combined Score: {rec['combined_score']:.3f}")
            if rec['match_keywords']:
                print(f"  Matching Keywords: {', '.join(rec['match_keywords'])}")
            print(f"  Bio: {rec['bio'][:150]}...")
            print()
        
        # Test mentor stats
        matcher = MentorMatcher()
        matcher.load_mentors('data/sample/mentors.csv')
        stats = matcher.get_mentor_stats()
        
        print("Mentor Database Statistics:")
        for key, value in stats.items():
            if key == 'most_common_domains':
                print(f"  {key}: {value[:5]}")  # Show top 5 only
            else:
                print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the mentors CSV file exists with proper columns.")