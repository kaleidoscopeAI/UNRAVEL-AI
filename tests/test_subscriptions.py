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
