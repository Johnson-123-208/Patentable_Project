import os

# Define the folder structure
folders = [
    "patentable-project/scripts",
    "patentable-project/scripts/extract_projects.py",
    "patentable-project/scripts/extract_patents.py",
    "patentable-project/services",
    "patentable-project/services/vector_index.py",
    "patentable-project/services/patent_scoring.py",
    "patentable-project/services/mentor_matching.py",
    "patentable-project/train",
    "patentable-project/train/patentability_train.py",
    "patentable-project/models",
    "patentable-project/models/patentability_xgb.pkl",
    "patentable-project/data/raw",
    "patentable-project/data/clean",
    "patentable-project/data/sample",
    "patentable-project/data/sample/projects.csv",
    "patentable-project/data/sample/patents.csv",
    "patentable-project/data/sample/mentors.csv",
    "patentable-project/app/backend",
    "patentable-project/app/backend/main.py",
    "patentable-project/app/backend/routes.py",
    "patentable-project/app/backend/models.py",
    "patentable-project/app/streamlit_app.py",
    "patentable-project/utils",
    "patentable-project/utils/export_report.py",
    "patentable-project/docs",
    "patentable-project/docs/handover.md",
    "patentable-project/docs/final_report.md",
    "patentable-project/presentation",
    "patentable-project/presentation/final_ppt.pptx",
    "patentable-project/tests",
    "patentable-project/tests/test_api.py",
    "patentable-project/requirements.txt",
    "patentable-project/Dockerfile",
    "patentable-project/docker-compose.yml",
    "patentable-project/README.md",
    "patentable-project/.gitignore"
]

for path in folders:
    # Check if path has a file extension (rudimentary check)
    if os.path.splitext(path)[1]:  
        # Create parent directories if they don't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Create empty file
        with open(path, 'w') as f:
            pass
    else:
        # Just a directory
        os.makedirs(path, exist_ok=True)

print("Project structure created successfully!")
