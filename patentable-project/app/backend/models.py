"""
Pydantic Models for FastAPI Backend
File: app/backend/models.py
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class Project(BaseModel):
    """Model for project data"""
    title: str = Field(..., description="Project title", min_length=1, max_length=200)
    description: str = Field(..., description="Project description", min_length=10)
    technology_areas: List[str] = Field(..., description="List of technology areas")
    innovation_level: str = Field(..., description="Innovation level (low, medium, high)")
    market_potential: str = Field(..., description="Market potential (low, medium, high)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "title": "AI-Powered Smart Home Assistant",
                "description": "An intelligent home automation system that uses machine learning to predict user preferences and optimize energy consumption.",
                "technology_areas": ["Artificial Intelligence", "IoT", "Machine Learning"],
                "innovation_level": "high",
                "market_potential": "high"
            }
        }

class ProjectResponse(BaseModel):
    """Response model for created project"""
    id: int
    title: str
    description: str
    technology_areas: List[str]
    innovation_level: str
    market_potential: str
    created_at: datetime
    
    class Config:
        from_attributes = True

class InferenceRequest(BaseModel):
    """Request model for inference endpoint"""
    project_id: Optional[int] = Field(None, description="Existing project ID")
    project: Optional[Project] = Field(None, description="New project data")
    top_k_patents: int = Field(5, description="Number of top patents to return", ge=1, le=20)
    top_k_mentors: int = Field(3, description="Number of top mentors to return", ge=1, le=10)
    
    class Config:
        json_schema_extra = {
            "example": {
                "project": {
                    "title": "AI-Powered Smart Home Assistant",
                    "description": "An intelligent home automation system that uses machine learning to predict user preferences and optimize energy consumption.",
                    "technology_areas": ["Artificial Intelligence", "IoT", "Machine Learning"],
                    "innovation_level": "high",
                    "market_potential": "high"
                },
                "top_k_patents": 5,
                "top_k_mentors": 3
            }
        }

class PatentMatch(BaseModel):
    """Model for patent matching results"""
    patent_id: str = Field(..., description="Patent ID")
    title: str = Field(..., description="Patent title")
    similarity_score: float = Field(..., description="Similarity score", ge=0.0, le=1.0)
    patentability_score: float = Field(..., description="Patentability score", ge=0.0, le=1.0)
    abstract: Optional[str] = Field(None, description="Patent abstract")
    inventors: Optional[List[str]] = Field(None, description="List of inventors")
    filing_date: Optional[str] = Field(None, description="Filing date")
    
class Mentor(BaseModel):
    """Model for mentor information"""
    name: str = Field(..., description="Mentor name")
    expertise: List[str] = Field(..., description="Areas of expertise")
    experience_years: int = Field(..., description="Years of experience", ge=0)
    success_rate: float = Field(..., description="Success rate", ge=0.0, le=1.0)
    contact_info: Dict[str, str] = Field(..., description="Contact information")
    bio: Optional[str] = Field(None, description="Mentor biography")
    match_score: Optional[float] = Field(None, description="Match score with project", ge=0.0, le=1.0)
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "Dr. Jane Smith",
                "expertise": ["Artificial Intelligence", "Machine Learning", "IoT"],
                "experience_years": 15,
                "success_rate": 0.85,
                "contact_info": {
                    "email": "jane.smith@example.com",
                    "linkedin": "linkedin.com/in/janesmith"
                },
                "bio": "Expert in AI and IoT with 15+ years of experience in patent strategy",
                "match_score": 0.92
            }
        }

class InferenceResponse(BaseModel):
    """Response model for inference results"""
    project_id: int = Field(..., description="Project ID")
    patent_matches: List[PatentMatch] = Field(..., description="Similar patents found")
    recommended_mentors: List[Mentor] = Field(..., description="Recommended mentors")
    patentability_analysis: Dict[str, Any] = Field(..., description="Patentability analysis results")
    processing_time: float = Field(..., description="Processing time in seconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "project_id": 1,
                "patent_matches": [
                    {
                        "patent_id": "US10123456",
                        "title": "Smart Home Automation System",
                        "similarity_score": 0.85,
                        "patentability_score": 0.75,
                        "abstract": "A system for automated home control...",
                        "inventors": ["John Doe", "Jane Roe"],
                        "filing_date": "2020-01-15"
                    }
                ],
                "recommended_mentors": [
                    {
                        "name": "Dr. Jane Smith",
                        "expertise": ["AI", "IoT"],
                        "experience_years": 15,
                        "success_rate": 0.85,
                        "contact_info": {"email": "jane@example.com"},
                        "match_score": 0.92
                    }
                ],
                "patentability_analysis": {
                    "novelty_score": 0.78,
                    "non_obviousness_score": 0.82,
                    "utility_score": 0.95,
                    "overall_patentability": 0.85,
                    "recommendations": ["Focus on unique AI learning algorithm"]
                },
                "processing_time": 2.45
            }
        }

class PatentDetails(BaseModel):
    """Model for detailed patent information"""
    patent_id: str
    title: str
    abstract: str
    inventors: List[str]
    assignee: Optional[str] = None
    filing_date: str
    publication_date: Optional[str] = None
    patent_family_size: Optional[int] = None
    technology_classification: List[str]
    claims_count: Optional[int] = None
    citations_count: Optional[int] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "patent_id": "US10123456",
                "title": "Smart Home Automation System with AI Learning",
                "abstract": "A comprehensive system for automated home control using artificial intelligence...",
                "inventors": ["John Doe", "Jane Roe", "Bob Smith"],
                "assignee": "TechCorp Inc.",
                "filing_date": "2020-01-15",
                "publication_date": "2021-07-20",
                "patent_family_size": 5,
                "technology_classification": ["G06F", "H04L", "G05B"],
                "claims_count": 20,
                "citations_count": 15
            }
        }

class ErrorResponse(BaseModel):
    """Model for error responses"""
    error: str
    message: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)