#!/usr/bin/env python3
"""
Celery Worker Configuration
Handles async task processing for generation and sanitization.
"""

import asyncio
import os
import logging
from celery import Celery
from app.core.generation import AIGenerator
from app.security.sanitizer import CodeSanitizer
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler("worker.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

app = Celery(
    'tasks',
    broker=os.getenv("REDIS_URL", "redis://redis:6379/0"),
    backend=os.getenv("REDIS_URL", "redis://redis:6379/1")
)

app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=300
)

generator = AIGenerator()
sanitizer = CodeSanitizer()

@app.task(name="generate_system_task")
def generate_system_task(description: str, complexity: int) -> dict:
    """Generate a system specification asynchronously."""
    try:
        logger.info(f"Starting generation task: {description[:50]}... (complexity: {complexity})")
        result = asyncio.run(generator.generate_system(description, complexity))
        logger.info(f"Generation task completed for {description[:50]}")
        return result
    except Exception as e:
        logger.error(f"Generation task failed: {str(e)}", exc_info=True)
        raise

@app.task(name="sanitize_project_task")
def sanitize_project_task(project_path: str) -> dict:
    """Sanitize a project directory asynchronously."""
    try:
        logger.info(f"Starting sanitization task for {project_path}")
        project_path_obj = Path(project_path)
        sanitizer.sanitize_project(project_path_obj)
        logger.info(f"Sanitization completed for {project_path}")
        return {"status": "sanitized", "path": str(project_path)}
    except Exception as e:
        logger.error(f"Sanitization task failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    app.worker_main(['worker', '--loglevel=INFO'])
