"""
API Routes for FastAPI Backend
File: app/backend/routes.py
"""

from fastapi import APIRouter, HTTPException, Depends, status
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import pandas as pd
import json
import time
import os
from typing import List

from .models import (
    Project, ProjectResponse, InferenceRequest, InferenceResponse, 
    Mentor, PatentDetails, PatentMatch, ErrorResponse
)

# Import ML services (these would be your actual ML modules)
try:
    from services.vector_index import VectorIndex
    from services.patent_scoring import PatentScorer
    from services.mentor_matching import MentorMatcher
except ImportError:
    # Mock implementations for development
    class VectorIndex:
        def find_similar_patents(self, project_description: str, top_k: int = 5):
            return [
                {
                    "patent_id": f"US101234{i}",
                    "title": f"Similar Patent {i}",
                    "similarity_score": 0.9 - (i * 0.1),
                    "abstract": f"Abstract for patent {i}...",
                    "inventors": [f"Inventor {i}A", f"Inventor {i}B"],
                    "filing_date": f"2020-0{i+1}-15"
                }
                for i in range(top_k)
            ]
    
    class PatentScorer:
        def calculate_patentability_score(self, project_data: dict, similar_patents: list):
            return {
                "novelty_score": 0.78,
                "non_obviousness_score": 0.82,
                "utility_score": 0.95,
                "overall_patentability": 0.85,
                "recommendations": [
                    "Focus on unique algorithmic approach",
                    "Emphasize novel hardware integration",
                    "Document technical advantages clearly"
                ]
            }
            
        def score_patent_matches(self, patents: list):
            for patent in patents:
                patent["patentability_score"] = 0.8 - (0.05 * patents.index(patent))
            return patents
    
    class MentorMatcher:
        def find_matching_mentors(self, project_data: dict, top_k: int = 3):
            mentors = [
                {
                    "name": "Dr. Sarah Johnson",
                    "expertise": ["Artificial Intelligence", "Machine Learning", "Patent Strategy"],
                    "experience_years": 15,
                    "success_rate": 0.88,
                    "contact_info": {
                        "email": "sarah.johnson@techventures.com",
                        "linkedin": "linkedin.com/in/sarahjohnson",
                        "phone": "+1-555-0123"
                    },
                    "bio": "Leading AI researcher and patent strategist with 15+ years of experience in tech commercialization.",
                    "match_score": 0.94
                },
                {
                    "name": "Prof. Michael Chen",
                    "expertise": ["IoT", "Smart Systems", "Innovation Management"],
                    "experience_years": 12,
                    "success_rate": 0.82,
                    "contact_info": {
                        "email": "m.chen@university.edu",
                        "linkedin": "linkedin.com/in/michaelchen"
                    },
                    "bio": "Professor and consultant specializing in IoT innovation and intellectual property.",
                    "match_score": 0.89
                },
                {
                    "name": "Dr. Emily Rodriguez",
                    "expertise": ["Technology Transfer", "Startup Strategy", "Patent Portfolio"],
                    "experience_years": 20,
                    "success_rate": 0.91,
                    "contact_info": {
                        "email": "emily@innovateip.com",
                        "linkedin": "linkedin.com/in/emilyrodriguez"
                    },
                    "bio": "Veteran technology transfer expert with extensive experience in patent commercialization.",
                    "match_score": 0.87
                }
            ]
            return mentors[:top_k]

# Initialize services
vector_index = VectorIndex()
patent_scorer = PatentScorer()
mentor_matcher = MentorMatcher()

router = APIRouter()

# Database setup
SQLITE_DATABASE_URL = "sqlite:///./projects.db"
engine = create_engine(SQLITE_DATABASE_URL, connect_args={"check_same_thread": False})
Base = declarative_base()

class ProjectDB(Base):
    __tablename__ = "projects"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    description = Column(Text)
    technology_areas = Column(Text)  # JSON string
    innovation_level = Column(String)
    market_potential = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

def get_db():
    from sqlalchemy.orm import sessionmaker
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/ingest/project", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
async def ingest_project(project: Project, db: Session = Depends(get_db)):
    """
    Save a new project to the database
    """
    try:
        # Create database entry
        db_project = ProjectDB(
            title=project.title,
            description=project.description,
            technology_areas=json.dumps(project.technology_areas),
            innovation_level=project.innovation_level,
            market_potential=project.market_potential
        )
        
        db.add(db_project)
        db.commit()
        db.refresh(db_project)
        
        # Convert back to response model
        return ProjectResponse(
            id=db_project.id,
            title=db_project.title,
            description=db_project.description,
            technology_areas=json.loads(db_project.technology_areas),
            innovation_level=db_project.innovation_level,
            market_potential=db_project.market_potential,
            created_at=db_project.created_at
        )
        
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save project: {str(e)}"
        )

@router.post("/infer", response_model=InferenceResponse)
async def run_inference(request: InferenceRequest, db: Session = Depends(get_db)):
    """
    Run complete inference pipeline: vector search + patent scoring + mentor matching
    """
    start_time = time.time()
    
    try:
        project_data = None
        project_id = None
        
        # Get project data
        if request.project_id:
            # Load existing project
            db_project = db.query(ProjectDB).filter(ProjectDB.id == request.project_id).first()
            if not db_project:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Project with ID {request.project_id} not found"
                )
            
            project_data = {
                "title": db_project.title,
                "description": db_project.description,
                "technology_areas": json.loads(db_project.technology_areas),
                "innovation_level": db_project.innovation_level,
                "market_potential": db_project.market_potential
            }
            project_id = db_project.id
            
        elif request.project:
            # Use provided project data
            project_data = request.project.dict()
            
            # Save to database for future reference
            db_project = ProjectDB(
                title=request.project.title,
                description=request.project.description,
                technology_areas=json.dumps(request.project.technology_areas),
                innovation_level=request.project.innovation_level,
                market_potential=request.project.market_potential
            )
            
            db.add(db_project)
            db.commit()
            db.refresh(db_project)
            project_id = db_project.id
            
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either project_id or project data must be provided"
            )
        
        # Step 1: Vector-based patent search
        similar_patents = vector_index.find_similar_patents(
            project_data["description"], 
            top_k=request.top_k_patents
        )
        
        # Step 2: Patent scoring
        scored_patents = patent_scorer.score_patent_matches(similar_patents)
        patentability_analysis = patent_scorer.calculate_patentability_score(
            project_data, 
            scored_patents
        )
        
        # Step 3: Mentor matching
        recommended_mentors = mentor_matcher.find_matching_mentors(
            project_data, 
            top_k=request.top_k_mentors
        )
        
        # Convert to response models
        patent_matches = [
            PatentMatch(
                patent_id=patent["patent_id"],
                title=patent["title"],
                similarity_score=patent["similarity_score"],
                patentability_score=patent["patentability_score"],
                abstract=patent.get("abstract"),
                inventors=patent.get("inventors", []),
                filing_date=patent.get("filing_date")
            )
            for patent in scored_patents
        ]
        
        mentor_objects = [
            Mentor(**mentor) for mentor in recommended_mentors
        ]
        
        processing_time = time.time() - start_time
        
        return InferenceResponse(
            project_id=project_id,
            patent_matches=patent_matches,
            recommended_mentors=mentor_objects,
            patentability_analysis=patentability_analysis,
            processing_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference pipeline failed: {str(e)}"
        )

@router.get("/patent/{patent_id}", response_model=PatentDetails)
async def get_patent_details(patent_id: str):
    """
    Get detailed patent information from CSV data
    """
    try:
        # Load patent data from CSV (adjust path as needed)
        csv_path = "data/patents.csv"
        
        if not os.path.exists(csv_path):
            # Mock data for development
            if patent_id.startswith("US"):
                return PatentDetails(
                    patent_id=patent_id,
                    title=f"Patent Title for {patent_id}",
                    abstract=f"Detailed abstract for patent {patent_id}. This patent describes innovative technology solutions in the relevant field.",
                    inventors=["Dr. John Smith", "Jane Doe", "Prof. Bob Wilson"],
                    assignee="TechCorp Industries",
                    filing_date="2020-03-15",
                    publication_date="2021-09-20",
                    patent_family_size=3,
                    technology_classification=["G06F", "H04L", "G05B"],
                    claims_count=18,
                    citations_count=12
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Patent {patent_id} not found"
                )
        
        # Load and search CSV
        patents_df = pd.read_csv(csv_path)
        patent_row = patents_df[patents_df['patent_id'] == patent_id]
        
        if patent_row.empty:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Patent {patent_id} not found in database"
            )
        
        row = patent_row.iloc[0]
        
        return PatentDetails(
            patent_id=row['patent_id'],
            title=row['title'],
            abstract=row['abstract'],
            inventors=row.get('inventors', '').split(';') if row.get('inventors') else [],
            assignee=row.get('assignee'),
            filing_date=row.get('filing_date', ''),
            publication_date=row.get('publication_date'),
            patent_family_size=row.get('patent_family_size'),
            technology_classification=row.get('technology_classification', '').split(';') if row.get('technology_classification') else [],
            claims_count=row.get('claims_count'),
            citations_count=row.get('citations_count')
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve patent details: {str(e)}"
        )

@router.get("/projects", response_model=List[ProjectResponse])
async def get_all_projects(db: Session = Depends(get_db)):
    """
    Get all projects from database
    """
    try:
        projects = db.query(ProjectDB).all()
        
        return [
            ProjectResponse(
                id=project.id,
                title=project.title,
                description=project.description,
                technology_areas=json.loads(project.technology_areas),
                innovation_level=project.innovation_level,
                market_potential=project.market_potential,
                created_at=project.created_at
            )
            for project in projects
        ]
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve projects: {str(e)}"
        )

@router.get("/projects/{project_id}", response_model=ProjectResponse)
async def get_project(project_id: int, db: Session = Depends(get_db)):
    """
    Get a specific project by ID
    """
    try:
        project = db.query(ProjectDB).filter(ProjectDB.id == project_id).first()
        
        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Project with ID {project_id} not found"
            )
        
        return ProjectResponse(
            id=project.id,
            title=project.title,
            description=project.description,
            technology_areas=json.loads(project.technology_areas),
            innovation_level=project.innovation_level,
            market_potential=project.market_potential,
            created_at=project.created_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve project: {str(e)}"
        )