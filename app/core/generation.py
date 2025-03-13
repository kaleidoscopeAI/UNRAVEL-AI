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
