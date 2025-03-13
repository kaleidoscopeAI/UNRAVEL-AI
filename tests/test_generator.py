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
