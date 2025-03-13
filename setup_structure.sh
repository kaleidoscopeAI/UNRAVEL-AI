#!/bin/bash

# Ensure the script exits on error
set -e

echo "Setting up Kaleidoscope AI Pro project structure..."

# Create directory structure
mkdir -p app/api app/commerce app/core app/db app/security app/services app/worker infrastructure/aws scripts tests

# Create and populate configuration files
echo "Creating alembic.ini..."
cat << 'EOF' > alembic.ini
[alembic]
script_location = alembic
sqlalchemy.url = postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@db:5432/${POSTGRES_DB}

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console
qualname =

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic
EOF

# Create empty __init__.py for api module
echo "Creating app/api/__init__.py..."
touch app/api/__init__.py

# Create and populate app/main.py
echo "Creating app/main.py..."
cat << 'EOF' > app/main.py
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
EOF

# Create and populate app/core/generation.py
echo "Creating app/core/generation.py..."
cat << 'EOF' > app/core/generation.py
#!/usr/bin/env python3
"""
AI Generation Engine
Generates technical specifications using LangChain and OpenAI.
"""

import asyncio
import json
import logging
import os
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential
from typing import Dict, Any

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler("generation.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class AIGenerator:
    def __init__(self):
        """Initialize the AI generator with OpenAI LLM."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set in environment")
        self.llm = OpenAI(
            temperature=0.7,
            model_name="gpt-4-1106-preview",
            max_tokens=2000,
            openai_api_key=api_key
        )
        self.prompt_template = PromptTemplate(
            input_variables=["description", "complexity"],
            template="""
            Generate a technical specification for a {complexity}/10 complexity system:
            {description}
            
            Include:
            - System components
            - Data flow diagram
            - API endpoints
            - Database schema
            - Security requirements
            """
        )
        logger.info("AIGenerator initialized")

    @retry(stop=stop_after_attempt(3), wait=wait_random_exponential(min=1, max=60))
    async def generate_system(self, description: str, complexity: int) -> Dict[str, Any]:
        """Generate a system specification."""
        try:
            logger.info(f"Generating system (complexity: {complexity}) for: {description[:50]}...")
            chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
            raw_output = await asyncio.wait_for(
                chain.arun({"description": description, "complexity": complexity}),
                timeout=120
            )
            return self._validate_output(raw_output)
        except asyncio.TimeoutError:
            logger.error("Generation timed out")
            raise RuntimeError("Generation request timed out")
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}", exc_info=True)
            raise

    def _validate_output(self, raw: str) -> Dict[str, Any]:
        """Validate and parse raw AI output."""
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Raw output not JSON, attempting basic parsing")
            components = raw.split("System components:")[1].split("Data flow")[0].strip() if "System components:" in raw else "N/A"
            return {
                "components": {"modules": components},
                "data_flow": "N/A",
                "api_endpoints": [],
                "database_schema": {},
                "security_requirements": "N/A"
            }
EOF

# Create and populate app/commerce/payments.py
echo "Creating app/commerce/payments.py..."
cat << 'EOF' > app/commerce/payments.py
#!/usr/bin/env python3
"""
Payment Processor
Handles Stripe subscription payments.
"""

import os
import stripe
from fastapi import HTTPException
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler("payments.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class PaymentProcessor:
    def __init__(self):
        """Initialize Stripe with API key."""
        stripe.api_key = os.getenv("STRIPE_KEY")
        if not stripe.api_key:
            raise ValueError("STRIPE_KEY not set in environment")
        logger.info("PaymentProcessor initialized")

    def create_subscription(self, user_id: str, tier: str) -> dict:
        """
        Create a Stripe subscription for a user.
        
        Args:
            user_id: Stripe customer ID
            tier: Subscription tier ("BASIC", "PRO", "ENTERPRISE")
        
        Returns:
            Subscription details
        """
        try:
            logger.info(f"Creating subscription for user {user_id}, tier {tier}")
            subscription = stripe.Subscription.create(
                customer=user_id,
                items=[{"price": self._get_price_id(tier)}],
                payment_behavior="default_incomplete",
                expand=["latest_invoice.payment_intent"]
            )
            logger.info(f"Subscription {subscription.id} created for user {user_id}")
            return subscription
        except stripe.error.StripeError as e:
            logger.error(f"Payment processing failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Payment processing failed: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

    def _get_price_id(self, tier: str) -> str:
        """Map tier to Stripe price ID."""
        price_ids = {
            "BASIC": "price_1P_basic_placeholder",
            "PRO": "price_1P_pro_placeholder",
            "ENTERPRISE": "price_1P_enterprise_placeholder"
        }
        return price_ids.get(tier, "price_1P_basic_placeholder")
EOF

# Create and populate app/commerce/subscriptions.py
echo "Creating app/commerce/subscriptions.py..."
cat << 'EOF' > app/commerce/subscriptions.py
#!/usr/bin/env python3
"""
Subscription Service
Manages user subscription tiers and usage tracking.
"""

from datetime import datetime, timedelta
from enum import Enum
from pydantic import BaseModel
import json
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler("subscriptions.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class SubscriptionTier(str, Enum):
    BASIC = "basic"
    PRO = "pro"
    ENTERPRISE = "enterprise"

class SubscriptionLimits(BaseModel):
    weekly_generations: int
    max_complexity: int
    support_level: str
    deployment_options: list

TIER_CONFIG = {
    SubscriptionTier.BASIC: SubscriptionLimits(
        weekly_generations=1,
        max_complexity=3,
        support_level="community",
        deployment_options=["docker"]
    ),
    SubscriptionTier.PRO: SubscriptionLimits(
        weekly_generations=5,
        max_complexity=7,
        support_level="email",
        deployment_options=["docker", "aws"]
    ),
    SubscriptionTier.ENTERPRISE: SubscriptionLimits(
        weekly_generations=999,
        max_complexity=10,
        support_level="24/7",
        deployment_options=["docker", "aws", "gcp", "azure"]
    )
}

class SubscriptionTracker(BaseModel):
    user_id: str
    tier: SubscriptionTier
    usage_count: int = 0
    reset_date: datetime

    def __init__(self, **data):
        super().__init__(**data)
        self._load_usage()

    def _load_usage(self):
        """Load usage from file or initialize."""
        usage_file = f"/app/sandboxes/{self.user_id}_usage.json"
        if os.path.exists(usage_file):
            try:
                with open(usage_file, 'r') as f:
                    data = json.load(f)
                    self.usage_count = data.get("count", 0)
                    self.reset_date = datetime.fromisoformat(data["reset_date"])
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Error loading usage for {self.user_id}: {e}")
                self._reset_counters()
        else:
            self._reset_counters()

    def _save_usage(self):
        """Save usage to file."""
        usage_file = f"/app/sandboxes/{self.user_id}_usage.json"
        try:
            with open(usage_file, 'w') as f:
                json.dump({"count": self.usage_count, "reset_date": self.reset_date.isoformat()}, f)
        except IOError as e:
            logger.error(f"Error saving usage for {self.user_id}: {e}")

    def check_usage(self) -> bool:
        """Check if user has available generations."""
        if self.reset_date < datetime.now():
            self._reset_counters()
        limit = TIER_CONFIG[self.tier].weekly_generations
        available = self.usage_count < limit
        logger.info(f"User {self.user_id} usage check: {available} ({self.usage_count}/{limit})")
        return available

    def use_generation(self) -> bool:
        """Use a generation if available."""
        if self.check_usage():
            self.usage_count += 1
            self._save_usage()
            logger.info(f"Generation used for {self.user_id}, count: {self.usage_count}")
            return True
        logger.warning(f"No generations available for {self.user_id}")
        return False

    def _reset_counters(self):
        """Reset usage counters."""
        self.usage_count = 0
        self.reset_date = datetime.now() + timedelta(weeks=1)
        self._save_usage()
        logger.info(f"Reset counters for {self.user_id}")
EOF

# Create and populate app/db/models.py
echo "Creating app/db/models.py..."
cat << 'EOF' > app/db/models.py
#!/usr/bin/env python3
"""
Database Models for Kaleidoscope AI
"""

from sqlalchemy import Column, Integer, String, DateTime, Enum, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
import enum
from datetime import datetime

Base = declarative_base()

class SubscriptionTierEnum(str, enum.Enum):
    BASIC = "basic"
    PRO = "pro"
    ENTERPRISE = "enterprise"

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    email = Column(String(255), unique=True, nullable=False)
    subscription_tier = Column(Enum(SubscriptionTierEnum), default=SubscriptionTierEnum.BASIC)
    usage_count = Column(Integer, default=0)
    reset_date = Column(DateTime, default=datetime.utcnow)

class GeneratedSystem(Base):
    __tablename__ = "generated_systems"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    description = Column(String(1000), nullable=False)
    complexity = Column(Integer, nullable=False)
    generated_at = Column(DateTime, default=func.now())
EOF

# Create and populate app/security/auth.py
echo "Creating app/security/auth.py..."
cat << 'EOF' > app/security/auth.py
#!/usr/bin/env python3
"""
Authentication Handler
Manages JWT-based authentication.
"""

from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
import os
import logging
from fastapi import HTTPException, Depends
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler("auth.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class AuthHandler:
    def __init__(self):
        """Initialize with secret key."""
        self.SECRET_KEY = os.getenv("SECRET_KEY")
        self.ALGORITHM = "HS256"
        if not self.SECRET_KEY:
            raise ValueError("SECRET_KEY not set in environment")
        logger.info("AuthHandler initialized")

    def create_access_token(self, data: dict, expires_delta: int = 3600) -> str:
        """Create a JWT access token."""
        to_encode = data.copy()
        to_encode.update({"exp": int(time.time()) + expires_delta})
        try:
            token = jwt.encode(to_encode, self.SECRET_KEY, algorithm=self.ALGORITHM)
            logger.info(f"Token created for {data.get('sub', 'unknown')}")
            return token
        except Exception as e:
            logger.error(f"Error creating token: {str(e)}")
            raise

    def validate_token(self, token: str = Depends(oauth2_scheme)) -> dict:
        """Validate a JWT token."""
        try:
            payload = jwt.decode(token, self.SECRET_KEY, algorithms=[self.ALGORITHM])
            logger.info(f"Token validated for {payload.get('sub', 'unknown')}")
            return payload
        except JWTError as e:
            logger.error(f"Invalid token: {str(e)}")
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
EOF

# Create and populate app/security/sanitizer.py
echo "Creating app/security/sanitizer.py..."
cat << 'EOF' > app/security/sanitizer.py
#!/usr/bin/env python3
"""
Code Sanitizer
Removes branding and sensitive markers from generated code.
"""

import re
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler("sanitizer.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class CodeSanitizer:
    BRAND_PATTERNS = [
        (r'//\s*Generated by Kaleidoscope AI', ''),
        (r'<!-- KALEIDOSCOPE_MARKER -->', ''),
        (r'KALEIDOSCOPE_VERSION=\d+\.\d+\.\d+', '')
    ]

    def sanitize_project(self, path: Path):
        """Sanitize all files in a project directory."""
        if not path.exists():
            logger.error(f"Project path {path} does not exist")
            return
        logger.info(f"Sanitizing project at {path}")
        for file in path.glob('**/*'):
            if file.is_file():
                self._sanitize_file(file)

    def _sanitize_file(self, file: Path):
        """Sanitize a single file."""
        try:
            with open(file, 'r+', encoding='utf-8') as f:
                content = f.read()
                original_size = len(content)
                for pattern, replacement in self.BRAND_PATTERNS:
                    content = re.sub(pattern, replacement, content)
                if len(content) != original_size:
                    f.seek(0)
                    f.write(content)
                    f.truncate()
                    logger.info(f"Sanitized {file}")
        except IOError as e:
            logger.error(f"Error sanitizing {file}: {str(e)}")
EOF

# Create and populate app/services/enterprise.py
echo "Creating app/services/enterprise.py..."
cat << 'EOF' > app/services/enterprise.py
#!/usr/bin/env python3
"""
Enterprise Feature Module
Adds enterprise-specific enhancements to generated systems.
"""

from typing import Dict, Any
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler("enterprise.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class EnterpriseEnhancer:
    @staticmethod
    def add_sso(system_config: Dict[str, Any]) -> Dict[str, Any]:
        """Add Single Sign-On configuration."""
        try:
            logger.info("Adding SSO configuration")
            system_config["auth"] = {
                "sso": True,
                "providers": ["google", "microsoft", "saml"]
            }
            return system_config
        except KeyError as e:
            logger.error(f"Error adding SSO: {str(e)}")
            raise

    @staticmethod
    def add_audit_logging(system_config: Dict[str, Any]) -> Dict[str, Any]:
        """Add audit logging configuration."""
        try:
            logger.info("Adding audit logging")
            system_config["monitoring"] = {
                "audit_logs": True,
                "retention_days": 365
            }
            return system_config
        except KeyError as e:
            logger.error(f"Error adding audit logging: {str(e)}")
            raise

    @classmethod
    def upgrade_to_enterprise(cls, system_config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply all enterprise enhancements."""
        return cls.add_sso(cls.add_audit_logging(system_config))
EOF

# Create and populate app/worker/celery.py
echo "Creating app/worker/celery.py..."
cat << 'EOF' > app/worker/celery.py
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
EOF

# Create and populate docker-compose.yml
echo "Creating docker-compose.yml..."
cat << 'EOF' > docker-compose.yml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "8000:8000"
    env_file:
      - .env
    depends_on:
      - redis
      - db
    networks:
      - kaleidoscope-net

  redis:
    image: redis:7-alpine
    volumes:
      - redis-data:/data
    networks:
      - kaleidoscope-net

  db:
    image: postgres:15-alpine
    env_file:
      - .env
    volumes:
      - postgres-data:/var/lib/postgresql/data
    networks:
      - kaleidoscope-net

volumes:
  redis-data:
  postgres-data:

networks:
  kaleidoscope-net:
    driver: bridge
EOF

# Create and populate Dockerfile
echo "Creating Dockerfile..."
cat << 'EOF' > Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
EOF

# Create and populate requirements.txt
echo "Creating requirements.txt..."
cat << 'EOF' > requirements.txt
fastapi==0.108.0
uvicorn==0.25.0
langchain==0.1.14
openai==1.12.0
celery==5.3.6
redis==5.0.1
psycopg2-binary==2.9.9
python-dotenv==1.0.0
python-jose==3.3.0
tenacity==8.2.3
sqlalchemy==2.0.28
alembic==1.13.1
stripe==8.5.0
EOF

# Create and populate scripts/deploy.sh
echo "Creating scripts/deploy.sh..."
cat << 'EOF' > scripts/deploy.sh
#!/bin/bash

# Build and start containers
docker-compose up -d --build

# Run migrations
docker-compose exec web alembic upgrade head

# Check health
curl -s http://localhost:8000/health || echo "Health check failed"
EOF
chmod +x scripts/deploy.sh

# Create and populate tests/test_generator.py
echo "Creating tests/test_generator.py..."
cat << 'EOF' > tests/test_generator.py
#!/usr/bin/env python3
"""
Tests for AIGenerator
"""

import pytest
from app.core.generation import AIGenerator

@pytest.mark.asyncio
async def test_generation():
    generator = AIGenerator()
    spec = await generator.generate_system("Test system", 1)
    assert "components" in spec
EOF

# Create and populate tests/test_subscriptions.py
echo "Creating tests/test_subscriptions.py..."
cat << 'EOF' > tests/test_subscriptions.py
#!/usr/bin/env python3
"""
Tests for SubscriptionTracker
"""

import pytest
from app.commerce.subscriptions import SubscriptionTracker, SubscriptionTier

def test_subscription_usage():
    tracker = SubscriptionTracker(user_id="test_user", tier=SubscriptionTier.BASIC)
    assert tracker.check_usage() is True
    tracker.use_generation()
    assert tracker.check_usage() is False
EOF

# Install dependencies (assuming virtualenv is active)
echo "Installing dependencies..."
pip install -r requirements.txt

# Final instructions
echo "Setup complete!"
echo "Next steps:"
echo "1. Create a .env file with the following variables:"
echo "   - OPENAI_API_KEY"
echo "   - STRIPE_KEY"
echo "   - SECRET_KEY"
echo "   - POSTGRES_USER"
echo "   - POSTGRES_PASSWORD"
echo "   - POSTGRES_DB"
echo "2. Run 'alembic upgrade head' to apply database migrations."
echo "3. Start the services with 'docker-compose up -d'."
echo "4. Run tests with 'pytest tests/'."
