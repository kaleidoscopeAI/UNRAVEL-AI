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
