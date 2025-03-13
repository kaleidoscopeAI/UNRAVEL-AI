#!/usr/bin/env python3
"""
Main FastAPI Application
Entry point for the Kaleidoscope AI web service.
"""

from fastapi import FastAPI, Depends, HTTPException, status, APIRouter
from pydantic import BaseModel
import os
import logging
from datetime import datetime
from app.core.generation import AIGenerator
from app.commerce.subscriptions import SubscriptionTracker, SubscriptionTier, TIER_CONFIG
from app.security.auth import AuthHandler, oauth2_scheme
from app.security.sanitizer import CodeSanitizer
from app.worker.celery import generate_system_task, sanitize_project_task

app = FastAPI(title="Kaleidoscope AI")
router = APIRouter(prefix="/api/v1")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler("main.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

auth_handler = AuthHandler()
generator = AIGenerator()
sanitizer = CodeSanitizer()

class SystemRequest(BaseModel):
    description: str
    complexity: int

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@router.post("/systems", status_code=status.HTTP_202_ACCEPTED)
async def create_system(
    request: SystemRequest,
    user: dict = Depends(auth_handler.validate_token)
):
    """Generate a system specification asynchronously."""
    tracker = SubscriptionTracker(
        user_id=user["sub"],
        tier=SubscriptionTier(user.get("tier", "basic")),
        reset_date=datetime.fromisoformat(user.get("reset_date", datetime.now().isoformat()))
    )
    
    if not tracker.check_usage():
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail="Subscription limit reached"
        )
    
    if request.complexity > TIER_CONFIG[tracker.tier].max_complexity:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Complexity exceeds tier limit"
        )
    
    task = generate_system_task.delay(request.description, request.complexity)
    tracker.use_generation()
    logger.info(f"Task {task.id} queued for user {user['sub']}")
    return {"task_id": task.id}

@router.get("/systems/{task_id}")
def get_system_status(task_id: str, user: dict = Depends(auth_handler.validate_token)):
    """Get status of a generation task."""
    task = generate_system_task.AsyncResult(task_id)
    return {
        "status": task.status,
        "result": task.result if task.ready() else None
    }

@router.post("/systems/{system_id}/export")
async def export_system(
    system_id: str,
    user: dict = Depends(auth_handler.validate_token)
):
    """Export a sanitized system (PRO+ only)."""
    if user["tier"] == SubscriptionTier.BASIC.value:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Export requires PRO subscription or higher"
        )
    
    export_path = f"/app/sandboxes/{system_id}"
    task = sanitize_project_task.delay(export_path)
    logger.info(f"Export task {task.id} queued for {system_id}")
    return {"export_id": task.id}

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
