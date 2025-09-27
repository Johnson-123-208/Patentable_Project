"""
Test Suite for FastAPI Backend
File: tests/test_api.py
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from unittest.mock import patch, MagicMock
import json
import tempfile
import os

# Import the FastAPI app and database dependencies
from app.backend.main import app, get_db, Base, ProjectDB
from app.backend.models import Project, InferenceRequest

# Create a test database
SQLITE_TEST_DATABASE_URL = "sqlite:///./test_projects.db"
test_engine = create_engine(SQLITE_TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

# Override the database dependency
app.dependency_overrides[get_db] = override_get_db

# Create test client
client = TestClient(app)

@pytest.fixture(scope="function")
def setup_test_db():
    """Create and clean up test database for each test"""
    # Create tables
    Base.metadata.create_all(bind=test_engine)
    yield
    # Clean up
    Base.metadata.drop_all(bind=test_engine)
    if os.path.exists("test_projects.db"):
        os.remove("test_projects.db")

class TestAPI:
    """Test suite for the FastAPI backend"""
    
    def test_root_endpoint(self, setup_test_db):
        """Test the root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert data["version"] == "1.0.0"
    
    def test_health_check(self, setup_test_db):
        """Test the health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
    
    def test_ingest_project(self, setup_test_db):
        """Test project ingestion endpoint"""
        project_data = {
            "title": "AI-Powered Smart Home Assistant",
            "description": "An intelligent home automation system that uses machine learning to predict user preferences and optimize energy consumption. The system includes voice recognition, predictive analytics, and automated device control.",
            "technology_areas": ["Artificial Intelligence", "IoT", "Machine Learning", "Home Automation"],
            "innovation_level": "high",
            "market_potential": "high"
        }
        
        response = client.post("/ingest/project", json=project_data)
        assert response.status_code == 201
        
        data = response.json()
        assert data["id"] == 1
        assert data["title"] == project_data["title"]
        assert data["description"] == project_data["description"]
        assert data["technology_areas"] == project_data["technology_areas"]
        assert data["innovation_level"] == project_data["innovation_level"]
        assert data["market_potential"] == project_data["market_potential"]
        assert "created_at" in data
    
    def test_ingest_project_validation_error(self, setup_test_db):
        """Test project ingestion with invalid data"""
        # Test with missing required fields
        invalid_project = {
            "title": "",  # Empty title should fail validation
            "description": "Short",  # Too short description
            "technology_areas": [],  # Empty list
            "innovation_level": "invalid",  # Not a valid level
            "market_potential": "high"
        }
        
        response = client.post("/ingest/project", json=invalid_project)
        assert response.status_code == 422
    
    def test_infer_endpoint_with_new_project(self, setup_test_db):
        """Test inference endpoint with new project data"""
        inference_request = {
            "project": {
                "title": "Blockchain-Based Supply Chain Tracker",
                "description": "A revolutionary supply chain management system using blockchain technology to ensure transparency, traceability, and authenticity of products from manufacture to delivery. The system integrates IoT sensors for real-time monitoring and smart contracts for automated compliance checking.",
                "technology_areas": ["Blockchain", "Supply Chain", "IoT", "Smart Contracts"],
                "innovation_level": "high",
                "market_potential": "medium"
            },
            "top_k_patents": 5,
            "top_k_mentors": 3
        }
        
        response = client.post("/infer", json=inference_request)
        assert response.status_code == 200
        
        data = response.json()
        
        # Validate response structure
        assert "project_id" in data
        assert "patent_matches" in data
        assert "recommended_mentors" in data
        assert "patentability_analysis" in data
        assert "processing_time" in data
        
        # Validate patent matches
        assert len(data["patent_matches"]) <= 5
        if data["patent_matches"]:
            patent = data["patent_matches"][0]
            assert "patent_id" in patent
            assert "title" in patent
            assert "similarity_score" in patent
            assert "patentability_score" in patent
            assert 0.0 <= patent["similarity_score"] <= 1.0
            assert 0.0 <= patent["patentability_score"] <= 1.0
        
        # Validate recommended mentors
        assert len(data["recommended_mentors"]) <= 3
        if data["recommended_mentors"]:
            mentor = data["recommended_mentors"][0]
            assert "name" in mentor
            assert "expertise" in mentor
            assert "experience_years" in mentor
            assert "success_rate" in mentor
            assert "contact_info" in mentor
            assert mentor["experience_years"] >= 0
            assert 0.0 <= mentor["success_rate"] <= 1.0
        
        # Validate patentability analysis
        analysis = data["patentability_analysis"]
        assert "novelty_score" in analysis
        assert "non_obviousness_score" in analysis
        assert "utility_score" in analysis
        assert "overall_patentability" in analysis
        assert "recommendations" in analysis
        
        # Validate processing time
        assert data["processing_time"] > 0
    
    def test_infer_endpoint_with_existing_project(self, setup_test_db):
        """Test inference endpoint with existing project ID"""
        # First, create a project
        project_data = {
            "title": "Quantum Computing Optimization Algorithm",
            "description": "Advanced quantum algorithm for solving complex optimization problems in logistics and resource allocation. Uses quantum annealing and variational quantum eigensolver approaches.",
            "technology_areas": ["Quantum Computing", "Optimization", "Algorithms"],
            "innovation_level": "high",
            "market_potential": "high"
        }
        
        create_response = client.post("/ingest/project", json=project_data)
        assert create_response.status_code == 201
        project_id = create_response.json()["id"]
        
        # Now test inference with existing project ID
        inference_request = {
            "project_id": project_id,
            "top_k_patents": 3,
            "top_k_mentors": 2
        }
        
        response = client.post("/infer", json=inference_request)
        assert response.status_code == 200
        
        data = response.json()
        assert data["project_id"] == project_id
        assert len(data["patent_matches"]) <= 3
        assert len(data["recommended_mentors"]) <= 2
    
    def test_infer_endpoint_invalid_project_id(self, setup_test_db):
        """Test inference endpoint with non-existent project ID"""
        inference_request = {
            "project_id": 999,  # Non-existent ID
            "top_k_patents": 5,
            "top_k_mentors": 3
        }
        
        response = client.post("/infer", json=inference_request)
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]
    
    def test_infer_endpoint_missing_project_data(self, setup_test_db):
        """Test inference endpoint without project data or ID"""
        inference_request = {
            "top_k_patents": 5,
            "top_k_mentors": 3
        }
        
        response = client.post("/infer", json=inference_request)
        assert response.status_code == 400
        assert "must be provided" in response.json()["detail"]
    
    def test_get_patent_details(self, setup_test_db):
        """Test getting patent details"""
        patent_id = "US10123456"
        
        response = client.get(f"/patent/{patent_id}")
        assert response.status_code == 200
        
        data = response.json()
        assert data["patent_id"] == patent_id
        assert "title" in data
        assert "abstract" in data
        assert "inventors" in data
        assert "filing_date" in data
        assert "technology_classification" in data
    
    def test_get_patent_details_not_found(self, setup_test_db):
        """Test getting details for non-existent patent"""
        patent_id = "INVALID123"
        
        response = client.get(f"/patent/{patent_id}")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]
    
    def test_get_all_projects_empty(self, setup_test_db):
        """Test getting all projects when database is empty"""
        response = client.get("/projects")
        assert response.status_code == 200
        assert response.json() == []
    
    def test_get_all_projects_with_data(self, setup_test_db):
        """Test getting all projects with data"""
        # Create multiple projects
        projects_data = [
            {
                "title": "Project 1",
                "description": "Description for project 1 with sufficient length to meet validation requirements.",
                "technology_areas": ["AI", "ML"],
                "innovation_level": "high",
                "market_potential": "medium"
            },
            {
                "title": "Project 2",
                "description": "Description for project 2 with sufficient length to meet validation requirements.",
                "technology_areas": ["Blockchain", "Crypto"],
                "innovation_level": "medium",
                "market_potential": "high"
            }
        ]
        
        # Create projects
        for project_data in projects_data:
            response = client.post("/ingest/project", json=project_data)
            assert response.status_code == 201
        
        # Get all projects
        response = client.get("/projects")
        assert response.status_code == 200
        
        data = response.json()
        assert len(data) == 2
        assert data[0]["title"] == "Project 1"
        assert data[1]["title"] == "Project 2"
    
    def test_get_project_by_id(self, setup_test_db):
        """Test getting a specific project by ID"""
        project_data = {
            "title": "Specific Project",
            "description": "A specific project description with enough content to pass validation requirements.",
            "technology_areas": ["IoT", "Edge Computing"],
            "innovation_level": "medium",
            "market_potential": "high"
        }
        
        # Create project
        create_response = client.post("/ingest/project", json=project_data)
        assert create_response.status_code == 201
        project_id = create_response.json()["id"]
        
        # Get project by ID
        response = client.get(f"/projects/{project_id}")
        assert response.status_code == 200
        
        data = response.json()
        assert data["id"] == project_id
        assert data["title"] == project_data["title"]
        assert data["description"] == project_data["description"]
    
    def test_get_project_by_id_not_found(self, setup_test_db):
        """Test getting non-existent project by ID"""
        response = client.get("/projects/999")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

class TestInferenceEndpointEdgeCases:
    """Test edge cases and error conditions for the inference endpoint"""
    
    def test_infer_with_boundary_values(self, setup_test_db):
        """Test inference with boundary values for top_k parameters"""
        project_data = {
            "title": "Test Project",
            "description": "A test project with adequate description length for validation purposes.",
            "technology_areas": ["Test Tech"],
            "innovation_level": "low",
            "market_potential": "low"
        }
        
        # Test with minimum values
        inference_request = {
            "project": project_data,
            "top_k_patents": 1,
            "top_k_mentors": 1
        }
        
        response = client.post("/infer", json=inference_request)
        assert response.status_code == 200
        
        data = response.json()
        assert len(data["patent_matches"]) <= 1
        assert len(data["recommended_mentors"]) <= 1
        
        # Test with maximum values
        inference_request = {
            "project": project_data,
            "top_k_patents": 20,
            "top_k_mentors": 10
        }
        
        response = client.post("/infer", json=inference_request)
        assert response.status_code == 200
        
        data = response.json()
        assert len(data["patent_matches"]) <= 20
        assert len(data["recommended_mentors"]) <= 10
    
    def test_infer_with_invalid_top_k_values(self, setup_test_db):
        """Test inference with invalid top_k values"""
        project_data = {
            "title": "Test Project",
            "description": "A test project with adequate description length for validation purposes.",
            "technology_areas": ["Test Tech"],
            "innovation_level": "medium",
            "market_potential": "medium"
        }
        
        # Test with values outside allowed range
        inference_request = {
            "project": project_data,
            "top_k_patents": 0,  # Below minimum
            "top_k_mentors": 15  # Above maximum
        }
        
        response = client.post("/infer", json=inference_request)
        assert response.status_code == 422  # Validation error
    
    def test_comprehensive_inference_workflow(self, setup_test_db):
        """Test complete workflow from project creation to inference"""
        # Step 1: Create a comprehensive project
        project_data = {
            "title": "AI-Enhanced Medical Diagnostic Tool",
            "description": "An advanced medical diagnostic system that combines deep learning, computer vision, and natural language processing to analyze medical images and patient records. The system provides real-time diagnostic suggestions with confidence scores and supporting evidence from medical literature.",
            "technology_areas": [
                "Artificial Intelligence",
                "Deep Learning", 
                "Computer Vision",
                "Natural Language Processing",
                "Medical Technology"
            ],
            "innovation_level": "high",
            "market_potential": "high"
        }
        
        # Step 2: Ingest project
        create_response = client.post("/ingest/project", json=project_data)
        assert create_response.status_code == 201
        project_id = create_response.json()["id"]
        
        # Step 3: Run inference
        inference_request = {
            "project_id": project_id,
            "top_k_patents": 10,
            "top_k_mentors": 5
        }
        
        inference_response = client.post("/infer", json=inference_request)
        assert inference_response.status_code == 200
        
        inference_data = inference_response.json()
        
        # Step 4: Validate comprehensive results
        assert inference_data["project_id"] == project_id
        
        # Check patent matches quality
        patents = inference_data["patent_matches"]
        assert len(patents) > 0
        
        # Patents should be sorted by similarity score (descending)
        if len(patents) > 1:
            for i in range(len(patents) - 1):
                assert patents[i]["similarity_score"] >= patents[i + 1]["similarity_score"]
        
        # Check mentor recommendations
        mentors = inference_data["recommended_mentors"]
        assert len(mentors) > 0
        
        # Mentors should have relevant expertise
        for mentor in mentors:
            assert len(mentor["expertise"]) > 0
            assert mentor["experience_years"] > 0
            assert 0.0 <= mentor["success_rate"] <= 1.0
        
        # Check patentability analysis completeness
        analysis = inference_data["patentability_analysis"]
        required_analysis_fields = [
            "novelty_score", "non_obviousness_score", 
            "utility_score", "overall_patentability", "recommendations"
        ]
        
        for field in required_analysis_fields:
            assert field in analysis
        
        # All scores should be between 0 and 1
        for score_field in ["novelty_score", "non_obviousness_score", "utility_score", "overall_patentability"]:
            assert 0.0 <= analysis[score_field] <= 1.0
        
        assert isinstance(analysis["recommendations"], list)
        assert len(analysis["recommendations"]) > 0
        
        # Step 5: Get detailed patent information
        if patents:
            patent_id = patents[0]["patent_id"]
            patent_response = client.get(f"/patent/{patent_id}")
            assert patent_response.status_code == 200
            
            patent_details = patent_response.json()
            assert patent_details["patent_id"] == patent_id
            assert len(patent_details["inventors"]) > 0
            assert len(patent_details["technology_classification"]) > 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])