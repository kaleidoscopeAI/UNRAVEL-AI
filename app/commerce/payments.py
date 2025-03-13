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
