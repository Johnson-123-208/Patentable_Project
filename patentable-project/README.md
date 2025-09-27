# Patentable Project Discovery & Mentor Recommender - Backend

A FastAPI-based backend service for discovering patentable projects and recommending suitable mentors using AI-powered analysis.

## Features

- **Project Ingestion**: Store and manage project data with comprehensive metadata
- **AI-Powered Inference**: Vector-based patent similarity search, patentability scoring, and mentor matching
- **RESTful API**: Well-documented endpoints with OpenAPI/Swagger integration
- **Database Integration**: SQLite for rapid development and deployment
- **Docker Support**: Containerized deployment with Docker and docker-compose
- **Comprehensive Testing**: Full test suite with pytest
- **Production Ready**: CORS support, health checks, and optional Nginx reverse proxy

## API Endpoints

### Core Endpoints

- `POST /ingest/project` - Save a new project to the database
- `POST /infer` - Run complete inference pipeline (vector search + patent scoring + mentor matching)
- `GET /patent/{id}` - Retrieve detailed patent information
- `GET /projects` - Get all projects
- `GET /projects/{id}` - Get specific project by ID
- `GET /health` - Health check endpoint
- `GET /docs` - Interactive API documentation (Swagger UI)

### Project Ingestion

```bash
curl -X POST "http://localhost:8000/ingest/project" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "AI-Powered Smart Home Assistant",
    "description": "An intelligent home automation system using ML to predict user preferences",
    "technology_areas": ["AI", "IoT", "Machine Learning"],
    "innovation_level": "high",
    "market_potential": "high"
  }'
```

### Running Inference

```bash
curl -X POST "http://localhost:8000/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "project": {
      "title": "Blockchain Supply Chain Tracker",
      "description": "Revolutionary supply chain management using blockchain technology",
      "technology_areas": ["Blockchain", "Supply Chain", "IoT"],
      "innovation_level": "high",
      "market_potential": "medium"
    },
    "top_k_patents": 5,
    "top_k_mentors": 3
  }'
```

## Quick Start

### Local Development

1. **Clone and setup**:
   ```bash
   git clone <repository>
   cd app/backend
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the server**:
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

4. **Access the API**:
   - API: http://localhost:8000
   - Documentation: http://localhost:8000/docs
   - Alternative docs: http://localhost:8000/redoc

### Docker Deployment

1. **Build and run with Docker Compose**:
   ```bash
   docker-compose up -d
   ```

2. **View logs**:
   ```bash
   docker-compose logs -f backend
   ```

3. **Stop services**:
   ```bash
   docker-compose down
   ```

### Production Deployment

For production deployment with Nginx reverse proxy:

```bash
docker-compose --profile production up -d
```

## Testing

Run the comprehensive test suite:

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_api.py -v

# Run with coverage
pytest tests/ --cov=app.backend --cov-report=html
```

### Test Coverage

The test suite includes:
- Endpoint validation testing
- Database integration testing  
- Edge case and error condition testing
- Complete workflow testing
- Input validation testing
- Mock service integration testing

## Project Structure

```
app/backend/
├── main.py              # FastAPI application entry point
├── routes.py            # API route definitions
├── models.py            # Pydantic data models
├── requirements.txt     # Python dependencies
tests/
├── test_api.py          # Comprehensive API test suite
docker-compose.yml       # Docker orchestration
Dockerfile              # Container build instructions
nginx.conf              # Reverse proxy configuration (optional)
```

## Configuration

### Environment Variables

- `DATABASE_URL`: SQLite database connection string (default: `sqlite:///./projects.db`)
- `ENVIRONMENT`: Deployment environment (`development`, `production`)
- `LOG_LEVEL`: Logging level (`debug`, `info`, `warning`, `error`)
- `CORS_ORIGINS`: Allowed CORS origins for frontend integration

### Database Schema

The backend uses SQLite with the following schema:

```sql
CREATE TABLE projects (
    id INTEGER PRIMARY KEY,
    title STRING NOT NULL,
    description TEXT NOT NULL,
    technology_areas TEXT NOT NULL,  -- JSON array
    innovation_level STRING NOT NULL,
    market_potential STRING NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

## ML Service Integration

The backend integrates with three core ML services:

1. **Vector Index Service** (`services.vector_index`):
   - Performs semantic similarity search against patent database
   - Returns ranked list of similar patents with similarity scores

2. **Patent Scoring Service** (`services.patent_scoring`):
   - Analyzes patentability based on novelty, non-obviousness, and utility
   - Provides detailed scoring and recommendations

3. **Mentor Matching Service** (`services.mentor_matching`):
   - Matches projects with suitable mentors based on expertise and track record
   - Returns ranked mentor recommendations with match scores

## API Response Models

### Inference Response
```json
{
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
```

## Health Monitoring

- **Health Check**: `GET /health` returns service status
- **Metrics**: Prometheus-compatible metrics endpoint (optional)
- **Logging**: Structured logging with request tracing
- **Docker Health Check**: Automated container health monitoring

## Security Features

- **Input Validation**: Comprehensive Pydantic model validation
- **CORS Configuration**: Configurable cross-origin resource sharing
- **Rate Limiting**: Built-in request rate limiting via Nginx
- **Security Headers**: Standard security headers in responses
- **Non-root Container**: Docker container runs as non-privileged user
- **SQL Injection Protection**: SQLAlchemy ORM prevents SQL injection attacks

## Performance Optimization

- **Async Support**: Full async/await support for non-blocking operations
- **Database Connection Pooling**: Efficient database connection management
- **Response Caching**: Optional Redis integration for caching frequent queries
- **Batch Processing**: Support for bulk operations
- **Lazy Loading**: Efficient data loading strategies

## Error Handling

The API provides comprehensive error handling with structured responses:

```json
{
  "error": "ValidationError",
  "message": "Invalid project data: title is required",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

Common HTTP status codes:
- `200 OK`: Successful operation
- `201 Created`: Resource created successfully
- `400 Bad Request`: Invalid input data
- `404 Not Found`: Resource not found
- `422 Unprocessable Entity`: Validation errors
- `500 Internal Server Error`: Server-side errors

## Monitoring and Observability

### Logging
- Structured JSON logging with correlation IDs
- Request/response logging for debugging
- Error tracking and alerting
- Performance metrics logging

### Health Checks
- Deep health checks including database connectivity
- Service dependency health monitoring
- Kubernetes/Docker health check integration

### Metrics
- Request latency and throughput metrics
- Database query performance
- ML model inference timing
- Custom business metrics

## Development Guidelines

### Code Quality
- **Type Hints**: Full type annotation coverage
- **Documentation**: Comprehensive docstrings and API documentation
- **Testing**: Minimum 90% code coverage requirement
- **Linting**: Black, isort, and flake8 for code formatting and quality

### API Design Principles
- **RESTful**: Follows REST architectural principles
- **Versioning**: API versioning support for backward compatibility
- **Pagination**: Cursor-based pagination for large result sets
- **Filtering**: Query parameter filtering and sorting
- **Consistent Responses**: Standardized response formats

## Deployment Options

### Development
```bash
# Local development server
uvicorn app.backend.main:app --reload --host 0.0.0.0 --port 8000
```

### Docker
```bash
# Single container
docker build -t patent-discovery-backend .
docker run -p 8000:8000 patent-discovery-backend

# With docker-compose
docker-compose up -d
```

### Kubernetes
```yaml
# Example Kubernetes deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: patent-discovery-backend
spec:
  replicas: 3
  selector:
    matchLabels:
      app: patent-discovery-backend
  template:
    metadata:
      labels:
        app: patent-discovery-backend
    spec:
      containers:
      - name: backend
        image: patent-discovery-backend:latest
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: database-url
```

### Cloud Deployment
- **AWS**: ECS/Fargate or EKS deployment
- **GCP**: Cloud Run or GKE deployment  
- **Azure**: Container Instances or AKS deployment
- **Heroku**: Direct container deployment

## Database Migrations

For production deployments, use Alembic for database migrations:

```bash
# Initialize migrations
alembic init migrations

# Generate migration
alembic revision --autogenerate -m "Add new table"

# Apply migrations
alembic upgrade head

# Rollback migrations
alembic downgrade -1
```

## Troubleshooting

### Common Issues

1. **Database Connection Errors**:
   ```bash
   # Check database file permissions
   ls -la projects.db
   
   # Verify SQLite installation
   python -c "import sqlite3; print(sqlite3.version)"
   ```

2. **Import Errors for ML Services**:
   ```bash
   # Verify ML services are available
   python -c "from services.vector_index import VectorIndex"
   
   # Check Python path
   echo $PYTHONPATH
   ```

3. **Port Already in Use**:
   ```bash
   # Find process using port 8000
   lsof -i :8000
   
   # Kill process
   kill -9 <PID>
   ```

4. **Docker Build Issues**:
   ```bash
   # Clear Docker cache
   docker system prune -a
   
   # Rebuild without cache
   docker build --no-cache -t patent-discovery-backend .
   ```

### Debug Mode

Enable debug mode for detailed error information:

```python
# In main.py
app = FastAPI(debug=True)

# Or via environment
export DEBUG=true
```

### Logging Configuration

Adjust logging levels:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/new-feature`
3. **Write tests**: Ensure new features have comprehensive tests
4. **Run quality checks**: `black . && isort . && flake8 .`
5. **Run tests**: `pytest tests/ -v`
6. **Submit pull request**: Include detailed description and test results

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

- **Documentation**: Check `/docs` endpoint for interactive API documentation
- **Issues**: Report bugs and feature requests via GitHub issues
- **Discussions**: Join project discussions for questions and feedback

## Changelog

### v1.0.0 (Current)
- Initial release with core functionality
- Project ingestion and management
- AI-powered inference pipeline
- Comprehensive test suite
- Docker deployment support
- Production-ready configuration

### Roadmap
- Authentication and authorization system
- Advanced caching and performance optimization
- Real-time notifications via WebSocket
- Advanced analytics and reporting
- Multi-tenant support
- Enhanced ML model integration