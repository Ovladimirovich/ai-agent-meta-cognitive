# Containerization Solution for AI Agent with Meta-Cognition

This document describes the comprehensive containerization solution for the AI Agent with Meta-Cognition application.

## Overview

The containerization solution includes:

- Multi-stage optimized Docker images for both backend and frontend
- Production-ready docker-compose configuration
- Development environment with hot-reload capabilities
- Monitoring and observability stack
- Security best practices
- Resource optimization

## Architecture

### Services

1. **Frontend** - React/TypeScript UI served by Nginx
2. **AI Agent API** - FastAPI backend with meta-cognitive capabilities
3. **PostgreSQL** - Primary database for application data
4. **Redis** - Caching and session storage
5. **ChromaDB** - Vector database for embeddings
6. **Monitoring Stack** (optional) - Prometheus, Grafana, Jaeger, Loki

## Docker Images

### Backend API Image

The backend image uses a multi-stage build approach:

- **Builder stage**: Installs Python dependencies in a virtual environment
- **Runtime stage**: Production-optimized image with minimal attack surface
- **Development stage**: Includes development tools and debugging utilities

### Frontend Image

The frontend image also uses a multi-stage build:

- **Dependencies stage**: Installs production dependencies
- **Build stage**: Creates optimized production build
- **Runtime stage**: Serves static files with Nginx
- **Development stage**: Runs development server with hot-reload

## Configuration

### Production Environment

To run the application in production:

```bash
# Create secrets directory
mkdir -p secrets
echo "your_postgres_password" > secrets/postgres_password.txt
echo "your_openai_api_key" > secrets/openai_api_key.txt
echo "your_secret_key" > secrets/secret_key.txt
echo "your_jwt_secret_key" > secrets/jwt_secret_key.txt
echo "your_redis_password" > secrets/redis_password.txt

# Start the services
docker-compose up -d
```

### Development Environment

To run the application in development mode:

```bash
# Set environment variables
export OPENAI_API_KEY=your_openai_api_key

# Start with development overrides
docker-compose -f docker-compose.yml -f docker-compose.override.yml up -d
```

### Monitoring Stack

To run with monitoring services:

```bash
# Start the main application
docker-compose up -d

# In another terminal, start monitoring
docker-compose -f docker-compose.monitoring.yml up -d
```

## Security Best Practices

1. **Secrets Management**: All sensitive data (API keys, passwords) are stored in Docker secrets
2. **Non-root User**: Applications run as non-root user with minimal required permissions
3. **Resource Limits**: CPU and memory limits prevent resource exhaustion
4. **Network Isolation**: Services communicate through dedicated Docker network
5. **Minimal Base Images**: Using Alpine-based and slim images to reduce attack surface

## Performance Optimizations

1. **Multi-stage Builds**: Reduces final image size by excluding build dependencies
2. **Dependency Caching**: Leverages Docker layer caching for faster builds
3. **Resource Constraints**: Configured resource limits to prevent overconsumption
4. **Health Checks**: Built-in health checks for service reliability
5. **Environment-specific Configurations**: Optimized settings for dev/prod environments

## Development Workflow

### Running in Development Mode

The development configuration includes:

- Hot-reload for both frontend and backend
- Direct access to source files via volume mounts
- Development tools (debugger, linters, etc.)
- Exposed ports for debugging

### Building Images

To build the images manually:

```bash
# Build backend image
docker build -t ai-agent-backend .

# Build frontend image
docker build -t ai-agent-frontend ./frontend

# Build specific target (runtime or development)
docker build --target runtime -t ai-agent-backend .
docker build --target development -t ai-agent-backend-dev .
```

## Monitoring and Observability

The monitoring stack includes:

- **Prometheus**: Metrics collection and storage
- **Grafana**: Metrics visualization and dashboards
- **Jaeger**: Distributed tracing
- **Loki**: Log aggregation
- **Promtail**: Log shipping agent
- **Node Exporter**: System metrics
- **Alertmanager**: Alert management

## Troubleshooting

### Common Issues

1. **Permission Errors**: Ensure Docker secrets have correct permissions
2. **Port Conflicts**: Check for port availability before starting services
3. **Resource Limits**: Adjust resource constraints based on available hardware
4. **Network Issues**: Verify services can communicate through the Docker network

### Useful Commands

```bash
# View logs
docker-compose logs -f ai-agent
docker-compose logs -f frontend

# Execute commands in containers
docker-compose exec ai-agent bash
docker-compose exec postgres psql -U ai_agent

# Check service status
docker-compose ps

# Monitor resources
docker stats

# Build with no cache
docker-compose build --no-cache
```

## Scaling

The containerized application supports horizontal scaling:

- **API Service**: Can be scaled based on load
- **Database**: Consider using managed services for production
- **Cache**: Redis can be configured in cluster mode
- **Frontend**: Stateless service can be scaled independently

## Environment Variables

Key environment variables for configuration:

- `ENVIRONMENT`: Set to "production" or "development"
- `LOG_LEVEL`: Logging verbosity (INFO, DEBUG, ERROR)
- `API_WORKERS`: Number of API server workers (production)
- `OPENAI_API_KEY`: OpenAI API key for AI features
- `POSTGRES_PASSWORD`: Database password (should use secrets in production)

## File Structure

```
.
├── Dockerfile                    # Backend multi-stage Dockerfile
├── frontend/Dockerfile           # Frontend multi-stage Dockerfile
├── frontend/nginx.conf           # Nginx configuration for frontend
├── docker-compose.yml            # Production docker-compose
├── docker-compose.override.yml   # Development overrides
├── docker-compose.monitoring.yml # Monitoring services
├── .dockerignore                 # Backend .dockerignore
├── frontend/.dockerignore        # Frontend .dockerignore
└── CONTAINERIZATION.md           # This documentation
```

## Next Steps

1. Set up CI/CD pipeline for automated builds
2. Configure external load balancer for production
3. Set up backup and disaster recovery procedures
4. Implement security scanning in CI/CD
5. Configure automated monitoring alerts