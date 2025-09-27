#!/usr/bin/env python3
"""
Complete demonstration of the Patentable Project Discovery & Mentor Recommender system.
This script runs the entire pipeline from data extraction to recommendations.
"""

import os
import sys
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def setup_directories():
    """Create necessary directories."""
    directories = [
        'data/sample',
        'data/clean', 
        'models'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    print("✓ Created project directories")

def run_data_extraction():
    """Run data extraction scripts."""
    print("\n=== DATA EXTRACTION ===")
    
    # Extract projects
    try:
        from scripts.extract_projects import ProjectExtractor
        extractor = ProjectExtractor()
        projects_df = extractor.process_file('data/sample/projects.csv', 'data/clean/projects.csv')
        print(f"✓ Extracted {len(projects_df)} projects")
    except Exception as e:
        print(f"✗ Error extracting projects: {e}")
        return False
    
    # Extract patents
    try:
        from scripts.extract_patents import PatentExtractor
        extractor = PatentExtractor()
        patents_df = extractor.process_csv('data/sample/patents.csv', 'data/clean/patents.csv')
        print(f"✓ Extracted {len(patents_df)} patents")
    except Exception as e:
        print(f"✗ Error extracting patents: {e}")
        return False
    
    return True

def build_vector_index():
    """Build vector index for similarity search."""
    print("\n=== BUILDING VECTOR INDEX ===")
    
    try:
        from services.vector_index import build_index
        index = build_index('data/clean/patents.csv', index_path='models/patent_index')
        print(f"✓ Built vector index with {index.ntotal} patents")
        return True
    except Exception as e:
        print(f"✗ Error building vector index: {e}")
        return False

def train_patentability_model():
    """Train the patentability prediction model."""
    print("\n=== TRAINING PATENTABILITY MODEL ===")
    
    try:
        from train.patentability_train import PatentabilityTrainer
        trainer = PatentabilityTrainer()
        results = trainer.full_training_pipeline(n_samples=1000)
        print(f"✓ Trained model with {results['test_accuracy']:.3f} test accuracy")
        return True
    except Exception as e:
        print(f"✗ Error training model: {e}")
        return False

def demo_patent_scoring():
    """Demonstrate patent scoring functionality."""
    print("\n=== PATENT SCORING DEMO ===")
    
    try:
        from services.patent_scoring import predict_patentability
        
        # Load sample projects
        projects_df = pd.read_csv('data/clean/projects.csv')
        
        print("Analyzing patentability for sample projects:")
        print("-" * 80)
        
        for idx, project in projects_df.iterrows():
            project_text = f"{project['title']} {project['abstract']}"
            result = predict_patentability(project_text)
            
            print(f"\nProject: {project['title']}")
            print(f"Score: {result['patentability_score']}/100")
            print(f"Label: {result['label']}")
            
            if result.get('similar_patents'):
                print("Most similar patent:")
                similar = result['similar_patents'][0]
                print(f"  - {similar['title']} (similarity: {similar['similarity_score']:.3f})")
        
        return True
    except Exception as e:
        print(f"✗ Error in patent scoring: {e}")
        return False

def demo_mentor_matching():
    """Demonstrate mentor matching functionality."""
    print("\n=== MENTOR MATCHING DEMO ===")
    
    try:
        from services.mentor_matching import recommend_mentors
        
        # Load sample projects
        projects_df = pd.read_csv('data/clean/projects.csv')
        
        print("Finding mentor recommendations for sample projects:")
        print("-" * 80)
        
        for idx, project in projects_df.iterrows():
            project_text = f"{project['title']} {project['abstract']}"
            recommendations = recommend_mentors(project_text, k=2)
            
            print(f"\nProject: {project['title']}")
            print("Recommended mentors:")
            
            for rec in recommendations:
                print(f"  {rec['rank']}. {rec['name']} (score: {rec['combined_score']:.3f})")
                print(f"     Expertise: {rec['domain_tags']}")
                if rec['match_keywords']:
                    print(f"     Matching: {', '.join(rec['match_keywords'])}")
        
        return True
    except Exception as e:
        print(f"✗ Error in mentor matching: {e}")
        return False

def demo_complete_pipeline():
    """Demonstrate complete pipeline for a new project."""
    print("\n=== COMPLETE PIPELINE DEMO ===")
    
    try:
        new_project = """
        IoT-based Smart Agriculture System: An integrated platform that uses IoT sensors 
        to monitor soil conditions, weather patterns, and crop health. The system employs 
        machine learning algorithms to predict optimal irrigation and fertilization schedules. 
        Key features include automated drone surveys, predictive analytics for disease detection, 
        and mobile alerts for farmers. The platform integrates with existing farm management 
        software and provides actionable insights to maximize crop yield while minimizing 
        resource usage.
        """
        
        print(f"Analyzing new project: {new_project[:100]}...")
        print("-" * 80)
        
        # Patent analysis
        from services.patent_scoring import predict_patentability
        patent_result = predict_patentability(new_project)
        
        print(f"PATENTABILITY ANALYSIS:")
        print(f"  Score: {patent_result['patentability_score']}/100")
        print(f"  Label: {patent_result['label']}")
        
        if patent_result.get('similar_patents'):
            print(f"  Most similar patents:")
            for i, patent in enumerate(patent_result['similar_patents'][:2], 1):
                print(f"    {i}. {patent['title']} ({patent['similarity_score']:.3f})")
        
        # Mentor matching
        from services.mentor_matching import recommend_mentors
        mentor_results = recommend_mentors(new_project, k=3)
        
        print(f"\nMENTOR RECOMMENDATIONS:")
        for rec in mentor_results:
            print(f"  {rec['rank']}. {rec['name']} (score: {rec['combined_score']:.3f})")
            print(f"      {rec['domain_tags']}")
        
        return True
    except Exception as e:
        print(f"✗ Error in complete pipeline: {e}")
        return False

def main():
    """Main demonstration function."""
    print("🚀 Patent Discovery & Mentor Recommender System Demo")
    print("=" * 60)
    
    # Setup
    setup_directories()
    
    # Check if sample data exists
    sample_files = [
        'data/sample/projects.csv',
        'data/sample/patents.csv', 
        'data/sample/mentors.csv'
    ]
    
    missing_files = [f for f in sample_files if not Path(f).exists()]
    if missing_files:
        print(f"\n⚠️  Missing sample data files: {missing_files}")
        print("Please create the sample data files first.")
        return
    
    # Run pipeline
    steps = [
        ("Data Extraction", run_data_extraction),
        ("Vector Index Building", build_vector_index), 
        ("Model Training", train_patentability_model),
        ("Patent Scoring Demo", demo_patent_scoring),
        ("Mentor Matching Demo", demo_mentor_matching),
        ("Complete Pipeline Demo", demo_complete_pipeline)
    ]
    
    results = {}
    for step_name, step_func in steps:
        try:
            success = step_func()
            results[step_name] = success
            if not success:
                print(f"\n⚠️  {step_name} failed, but continuing...")
        except KeyboardInterrupt:
            print(f"\n⏸️  Demo interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error in {step_name}: {e}")
            results[step_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 DEMO SUMMARY")
    print("=" * 60)
    
    for step_name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{status}: {step_name}")
    
    successful_steps = sum(results.values())
    total_steps = len(results)
    
    print(f"\nOverall: {successful_steps}/{total_steps} steps completed successfully")
    
    if successful_steps == total_steps:
        print("\n🎉 All systems operational! The ML core is ready for integration.")
    else:
        print(f"\n⚠️  Some steps failed. Check the error messages above for details.")
    
    print("\n📁 Generated files:")
    output_files = [
        'data/clean/projects.csv',
        'data/clean/patents.csv', 
        'models/patent_index.index',
        'models/patent_index.pkl',
        'models/patentability_xgb.pkl'
    ]
    
    for file_path in output_files:
        if Path(file_path).exists():
            size = Path(file_path).stat().st_size
            print(f"  ✓ {file_path} ({size:,} bytes)")
        else:
            print(f"  ✗ {file_path} (missing)")

if __name__ == "__main__":
    main()