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
