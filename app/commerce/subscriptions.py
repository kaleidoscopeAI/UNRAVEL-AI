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
