#!/bin/bash

# Build and start containers
docker-compose up -d --build

# Run migrations
docker-compose exec web alembic upgrade head

# Check health
curl -s http://localhost:8000/health || echo "Health check failed"
