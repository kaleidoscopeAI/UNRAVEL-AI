#!/usr/bin/env python3
"""
Kaleidoscope AI - LLM Integration Module
========================================
Provides robust integration with Language Model APIs for software analysis,
including chunking, specialized prompting, and streaming capabilities to handle
large files that exceed context windows.

This module implements the techniques described by Geoffrey Huntley for
using LLMs to analyze and transform software.
"""

import os
import sys
import json
import time
import asyncio
import logging
import hashlib
import tiktoken
from typing import Dict, List, Any, Optional, Union, Tuple, Callable, AsyncGenerator
import aiohttp
import backoff
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("llm_integration.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class TokenCounter:
    """Utility for counting tokens in prompts"""
    
    def __init__(self, model: str = "gpt-4"):
        """
        Initialize the token counter
        
        Args:
            model: The model to use for token counting
        """
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            logger.warning(f"Model '{model}' not found, using cl100k_base encoding")
            self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        return len(self.encoding.encode(text))
    
    def count_messages_tokens(self, messages: List[Dict[str, str]]) -> int:
        """
        Count tokens in a messages array (for ChatCompletions)
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Total token count
        """
        total_tokens = 0
        
        for message in messages:
            # Count message metadata tokens (role, etc.)
            total_tokens += 4  # approximation for message formatting
            
            # Count content tokens
            if "content" in message and message["content"]:
                total_tokens += self.count_tokens(message["content"])
        
        # Add tokens for the assistant's reply format
        total_tokens += 2  # approximation for reply formatting
        
        return total_tokens

class LLMChunkManager:
    """Manages chunking of large files for LLM processing"""
    
    def __init__(self, max_tokens: int = 8000, overlap: int = 500):
        """
        Initialize the chunk manager
        
        Args:
            max_tokens: Maximum tokens per chunk
            overlap: Number of tokens to overlap between chunks
        """
        self.max_tokens = max_tokens
        self.overlap = overlap
        self.token_counter = TokenCounter()
    
    def split_text_into_chunks(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        # Split text into lines
        lines = text.split('\n')
        
        chunks = []
        current_chunk = []
        current_token_count = 0
        
        for line in lines:
            line_token_count = self.token_counter.count_tokens(line)
            
            # If adding this line would exceed the limit, finalize the current chunk
            if current_token_count + line_token_count > self.max_tokens and current_chunk:
                chunks.append('\n'.join(current_chunk))
                
                # Start a new chunk with overlap
                overlap_start = max(0, len(current_chunk) - self.calculate_overlap_lines(current_chunk))
                current_chunk = current_chunk[overlap_start:]
                current_token_count = self.token_counter.count_tokens('\n'.join(current_chunk))
            
            # Add the current line
            current_chunk.append(line)
            current_token_count += line_token_count
        
        # Add the final chunk if it's not empty
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks
    
    def calculate_overlap_lines(self, lines: List[str]) -> int:
        """
        Calculate how many lines to include in the overlap
        
        Args:
            lines: Current chunk lines
            
        Returns:
            Number of lines for overlap
        """
        target_tokens = self.overlap
        total_lines = len(lines)
        token_count = 0
        lines_needed = 0
        
        # Count backward from the end to find overlap lines
        for i in range(total_lines - 1, -1, -1):
            line_tokens = self.token_counter.count_tokens(lines[i])
            if token_count + line_tokens > target_tokens:
                break
            
            token_count += line_tokens
            lines_needed += 1
            
            if lines_needed >= total_lines // 2:
                # Don't use more than half the chunk for overlap
                break
        
        return lines_needed

class LLMProvider:
    """Base class for LLM providers"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the LLM provider
        
        Args:
            api_key: API key for the provider
        """
        self.api_key = api_key or os.environ.get(self._get_api_key_env_var())
        if not self.api_key:
            logger.warning(f"No API key provided for {self.__class__.__name__}")
    
    def _get_api_key_env_var(self) -> str:
        """Get the environment variable name for the API key"""
        raise NotImplementedError("Subclasses must implement this method")
    
    async def generate_completion(self, prompt: str, **kwargs) -> str:
        """
        Generate a completion for a prompt
        
        Args:
            prompt: Prompt text
            **kwargs: Additional parameters
            
        Returns:
            Generated completion
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    async def generate_chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a chat completion
        
        Args:
            messages: List of message dictionaries
            **kwargs: Additional parameters
            
        Returns:
            Response from the LLM
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    async def stream_chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Stream a chat completion
        
        Args:
            messages: List of message dictionaries
            **kwargs: Additional parameters
            
        Yields:
            Chunks of the generated response
        """
        raise NotImplementedError("Subclasses must implement this method")

class OpenAIProvider(LLMProvider):
    """OpenAI API provider"""
    
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        model: str = "gpt-4-1106-preview", 
        organization: Optional[str] = None
    ):
        """
        Initialize the OpenAI provider
        
        Args:
            api_key: OpenAI API key
            model: Model to use
            organization: OpenAI organization ID
        """
        super().__init__(api_key)
        self.model = model
        self.organization = organization
        self.base_url = "https://api.openai.com/v1"
        self.token_counter = TokenCounter(model)
    
    def _get_api_key_env_var(self) -> str:
        return "OPENAI_API_KEY"
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        if self.organization:
            headers["OpenAI-Organization"] = self.organization
            
        return headers
    
    @retry(
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30)
    )
    async def generate_completion(self, prompt: str, **kwargs) -> str:
        """
        Generate a completion using the OpenAI API
        
        Args:
            prompt: Prompt text
            **kwargs: Additional parameters
            
        Returns:
            Generated completion
        """
        url = f"{self.base_url}/completions"
        
        # Set default parameters
        params = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": kwargs.get("max_tokens", 1000),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 1.0),
            "n": 1,
            "stream": False,
        }
        
        # Update with any additional parameters
        params.update({k: v for k, v in kwargs.items() if k not in ["model", "prompt"]})
        
        logger.debug(f"Sending completion request to OpenAI: {prompt[:100]}...")
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, 
                headers=self._get_headers(), 
                json=params,
                timeout=300  # 5-minute timeout
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"OpenAI API error: {response.status} - {error_text}")
                    response.raise_for_status()
                
                result = await response.json()
                
                if "choices" in result and len(result["choices"]) > 0:
                    return result["choices"][0].get("text", "")
                else:
                    logger.error(f"Unexpected response format: {result}")
                    return ""
    
    @retry(
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30)
    )
    async def generate_chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a chat completion using the OpenAI API
        
        Args:
            messages: List of message dictionaries
            **kwargs: Additional parameters
            
        Returns:
            Response from the API
        """
        url = f"{self.base_url}/chat/completions"
        
        # Count tokens to validate against model limits
        token_count = self.token_counter.count_messages_tokens(messages)
        logger.debug(f"Chat completion request with {token_count} tokens")
        
        # Set default parameters
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 1.0),
            "n": 1,
            "stream": False,
        }
        
        # Update with any additional parameters
        params.update({k: v for k, v in kwargs.items() if k not in ["model", "messages"]})
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, 
                headers=self._get_headers(), 
                json=params,
                timeout=300  # 5-minute timeout
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"OpenAI API error: {response.status} - {error_text}")
                    response.raise_for_status()
                
                result = await response.json()
                return result
    
    async def stream_chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Stream a chat completion from the OpenAI API
        
        Args:
            messages: List of message dictionaries
            **kwargs: Additional parameters
            
        Yields:
            Chunks of the generated response
        """
        url = f"{self.base_url}/chat/completions"
        
        # Set default parameters
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 1.0),
            "n": 1,
            "stream": True,
        }
        
        # Update with any additional parameters
        params.update({k: v for k, v in kwargs.items() if k not in ["model", "messages", "stream"]})
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, 
                    headers=self._get_headers(), 
                    json=params,
                    timeout=600  # 10-minute timeout
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"OpenAI API error: {response.status} - {error_text}")
                        response.raise_for_status()
                    
                    # Process the streaming response
                    async for line in response.content.iter_any():
                        line = line.decode('utf-8')
                        
                        # Handle SSE format
                        if line.startswith('data: '):
                            data = line[6:]
                            
                            if data.strip() == "[DONE]":
                                break
                            
                            try:
                                chunk = json.loads(data)
                                if "choices" in chunk and len(chunk["choices"]) > 0:
                                    delta = chunk["choices"][0].get("delta", {})
                                    if "content" in delta:
                                        yield delta["content"]
                            except json.JSONDecodeError:
                                logger.error(f"Error parsing chunk: {data}")
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.error(f"Error in streaming: {str(e)}")
            raise

class AnthropicProvider(LLMProvider):
    """Anthropic Claude API provider"""
    
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        model: str = "claude-2"
    ):
        """
        Initialize the Anthropic provider
        
        Args:
            api_key: Anthropic API key
            model: Model to use
        """
        super().__init__(api_key)
        self.model = model
        self.base_url = "https://api.anthropic.com/v1"
    
    def _get_api_key_env_var(self) -> str:
        return "ANTHROPIC_API_KEY"
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers"""
        return {
            "Content-Type": "application/json",
            "X-API-Key": self.api_key,
            "anthropic-version": "2023-06-01"
        }
    
    @retry(
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30)
    )
    async def generate_chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a chat completion using the Anthropic API
        
        Args:
            messages: List of message dictionaries
            **kwargs: Additional parameters
            
        Returns:
            Response from the API
        """
        url = f"{self.base_url}/messages"
        
        # Convert messages to Anthropic format
        anthropic_messages = []
        for msg in messages:
            role = "assistant" if msg["role"] == "assistant" else "user"
            anthropic_messages.append({
                "role": role,
                "content": msg["content"]
            })
        
        # Set default parameters
        params = {
            "model": self.model,
            "messages": anthropic_messages,
            "max_tokens": kwargs.get("max_tokens", 1000),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 1.0),
            "stream": False,
        }
        
        # Update with any additional parameters
        params.update({k: v for k, v in kwargs.items() if k not in ["model", "messages"]})
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, 
                headers=self._get_headers(), 
                json=params,
                timeout=300  # 5-minute timeout
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Anthropic API error: {response.status} - {error_text}")
                    response.raise_for_status()
                
                result = await response.json()
                
                # Format the response to be similar to OpenAI
                return {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": result.get("content", [{"text": ""}])[0].get("text", "")
                            },
                            "finish_reason": result.get("stop_reason", "")
                        }
                    ],
                    "usage": result.get("usage", {})
                }
    
    async def generate_completion(self, prompt: str, **kwargs) -> str:
        """
        Generate a completion using the Anthropic API (via chat)
        
        Args:
            prompt: Prompt text
            **kwargs: Additional parameters
            
        Returns:
            Generated completion
        """
        messages = [{"role": "user", "content": prompt}]
        response = await self.generate_chat_completion(messages, **kwargs)
        
        if "choices" in response and len(response["choices"]) > 0:
            return response["choices"][0]["message"]["content"]
        return ""
    
    async def stream_chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Stream a chat completion from the Anthropic API
        
        Args:
            messages: List of message dictionaries
            **kwargs: Additional parameters
            
        Yields:
            Chunks of the generated response
        """
        url = f"{self.base_url}/messages"
        
        # Convert messages to Anthropic format
        anthropic_messages = []
        for msg in messages:
            role = "assistant" if msg["role"] == "assistant" else "user"
            anthropic_messages.append({
                "role": role,
                "content": msg["content"]
            })
        
        # Set default parameters
        params = {
            "model": self.model,
            "messages": anthropic_messages,
            "max_tokens": kwargs.get("max_tokens", 1000),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 1.0),
            "stream": True,
        }
        
        # Update with any additional parameters
        params.update({k: v for k, v in kwargs.items() if k not in ["model", "messages", "stream"]})
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, 
                    headers=self._get_headers(), 
                    json=params,
                    timeout=600  # 10-minute timeout
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Anthropic API error: {response.status} - {error_text}")
                        response.raise_for_status()
                    
                    # Process the streaming response
                    async for line in response.content.iter_any():
                        line = line.decode('utf-8')
                        
                        # Handle SSE format
                        if line.startswith('data: '):
                            data = line[6:]
                            
                            if data.strip() == "[DONE]":
                                break
                            
                            try:
                                chunk = json.loads(data)
                                content_block = chunk.get("delta", {}).get("text", "")
                                if content_block:
                                    yield content_block
                            except json.JSONDecodeError:
                                logger.error(f"Error parsing chunk: {data}")
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.error(f"Error in streaming: {str(e)}")
            raise

class LLMPromptManager:
    """Manages specialized prompts for software analysis tasks"""
    
    def __init__(self):
        """Initialize the prompt manager"""
        # Load prompt templates
        self.templates = {
            "decompilation": self._load_prompt_template("decompilation"),
            "structure_analysis": self._load_prompt_template("structure_analysis"),
            "specification": self._load_prompt_template("specification"),
            "reconstruction": self._load_prompt_template("reconstruction"),
            "mimicry": self._load_prompt_template("mimicry")
        }
    
    def _load_prompt_template(self, template_name: str) -> str:
        """
        Load a prompt template
        
        Args:
            template_name: Name of the template
            
        Returns:
            Template string
        """
        # Define default templates
        default_templates = {
            "decompilation": """
You are an expert software reverse engineer. Analyze the following code thoroughly:

{code}

First, identify the purpose and functionality of this code.
Then, extract the core logic and algorithms.
Finally, provide a clean, restructured version with meaningful variable and function names.

Your response should include:
1. An overview of the code's purpose
2. Key structures and functions
3. A clean, readable version of the code
""",
            
            "structure_analysis": """
You are an expert software architect. Analyze the following code and extract its structure:

{code}

Your response should include:
1. High-level architecture overview
2. Main classes/modules and their responsibilities
3. Key dependencies and relationships
4. Data flow and control flow
5. Any design patterns identified

Format the output as a structured analysis that could be used to recreate this software from scratch.
""",
            
            "specification": """
Based on the previous analysis of the software, create a comprehensive specification document.

The specification should:
1. Define the purpose and scope of the software
2. List all functional requirements
3. Describe data structures and relationships
4. Detail APIs and interfaces
5. Outline key algorithms and logic

This specification will be used as the foundation for reconstructing the software, so it should be complete and precise.
""",
            
            "reconstruction": """
You are tasked with reconstructing software based on the following specification:

{specification}

Create clean, well-structured code that implements this specification. Your code should:
1. Follow modern best practices for the target language
2. Be maintainable and readable
3. Include appropriate error handling and validation
4. Be well-documented with comments

Target language: {language}
""",
            
            "mimicry": """
You are tasked with creating an enhanced version of the following software:

{specification}

Create a new implementation in {language} that:
1. Maintains the same core functionality but with improved architecture
2. Uses modern language features and best practices
3. Adds better error handling and logging
4. Improves performance where possible
5. Enhances security and data validation
6. Follows a clean, maintainable structure

In essence, you should build a better version of this software by applying modern software engineering principles while keeping the original functionality intact.
"""
        }
        
        # Return the default template
        return default_templates.get(template_name, "")
    
    def get_prompt(self, template_name: str, **kwargs) -> str:
        """
        Get a formatted prompt
        
        Args:
            template_name: Name of the template
            **kwargs: Variables to format the template with
            
        Returns:
            Formatted prompt
        """
        template = self.templates.get(template_name, "")
        if not template:
            logger.warning(f"Template '{template_name}' not found")
            return ""
        
        return template.format(**kwargs)
    
    def get_chunked_prompts(
        self, 
        template_name: str, 
        code: str, 
        chunk_manager: LLMChunkManager, 
        **kwargs
    ) -> List[str]:
        """
        Get prompts for chunked code
        
        Args:
            template_name: Name of the template
            code: Code to analyze (will be chunked)
            chunk_manager: Chunk manager instance
            **kwargs: Additional template variables
            
        Returns:
            List of formatted prompts, one for each chunk
        """
        # Split code into chunks
        code_chunks = chunk_manager.split_text_into_chunks(code)
        
        # Create prompts for each chunk
        prompts = []
        
        for i, chunk in enumerate(code_chunks):
            # Create context for this chunk
            chunk_context = f"CHUNK {i+1} OF {len(code_chunks)}\n\n"
            
            # Format the template with this chunk
            kwargs_copy = kwargs.copy()
            kwargs_copy["code"] = chunk_context + chunk
            
            prompts.append(self.get_prompt(template_name, **kwargs_copy))
        
        return prompts

class LLMCache:
    """Caches LLM responses to avoid redundant API calls"""
    
    def __init__(self, cache_dir: str = ".llm_cache"):
        """
        Initialize the cache
        
        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_key(self, prompt: str, params: Dict[str, Any]) -> str:
        """
        Generate a cache key for a prompt and parameters
        
        Args:
            prompt: Prompt text
            params: Additional parameters
            
        Returns:
            Cache key
        """
        # Create a string representation of the parameters
        param_str = json.dumps(params, sort_keys=True)
        
        # Create a hash of the prompt and parameters
        combined = prompt + param_str
        return hashlib.md5(combined.encode()).hexdigest()
    
    def get_cached_response(self, prompt: str, params: Dict[str, Any]) -> Optional[str]:
        """
        Get a cached response if available
        
        Args:
            prompt: Prompt text
            params: Additional parameters
            
        Returns:
            Cached response or None
        """
        cache_key = self._get_cache_key(prompt, params)
        cache_path = os.path.join(self.cache_dir, cache_key)
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                logger.error(f"Error reading cache: {str(e)}")
                return None
        
        return None
    
    def cache_response(self, prompt: str, params: Dict[str, Any], response: str) -> bool:
        """
        Cache a response
        
        Args:
            prompt: Prompt text
            params: Additional parameters
            response: Response to cache
            
        Returns:
            Success flag
        """
        cache_key = self._get_cache_key(prompt, params)
        cache_path = os.path.join(self.cache_dir, cache_key)
        
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                f.write(response)
            return True
        except Exception as e:
            logger.error(f"Error writing cache: {str(e)}")
            return False

class LLMIntegration:
    """Main integration class for using LLMs in software analysis"""
    
    def __init__(
        self, 
        provider: str = "openai", 
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        cache_dir: str = ".llm_cache",
        max_tokens_per_chunk: int = 8000,
        overlap_tokens: int = 500
    ):
        """
        Initialize the LLM integration
        
        Args:
            provider: LLM provider ('openai' or 'anthropic')
            api_key: API key for the provider
            model: Model to use
            cache_dir: Directory for caching responses
            max_tokens_per_chunk: Maximum tokens per chunk
            overlap_tokens: Tokens to overlap between chunks
        """
        # Initialize provider
        if provider.lower() == "openai":
            self.provider = OpenAIProvider(api_key=api_key, model=model or "gpt-4-1106-preview")
        elif provider.lower() == "anthropic":
            self.provider = AnthropicProvider(api_key=api_key, model=model or "claude-2")
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        # Initialize components
        self.cache = LLMCache(cache_dir=cache_dir)
        self.chunk_manager = LLMChunkManager(max_tokens=max_tokens_per_chunk, overlap=overlap_tokens)
        self.prompt_manager = LLMPromptManager()
    
    async def analyze_code(
        self, 
        code: str, 
        task: str = "decompilation", 
        language: Optional[str] = None,
        use_cache: bool = True,
        **kwargs
    ) -> str:
        """
        Analyze code using the LLM
        
        Args:
            code: Code to analyze
            task: Analysis task ('decompilation', 'structure_analysis', etc.)
            language: Target language for reconstruction/mimicry
            use_cache: Whether to use cached responses
            **kwargs: Additional parameters
            
        Returns:
            Analysis result
        """
        logger.info(f"Analyzing code for task: {task}")
        
        # Check if code is too large for a single prompt
        token_counter = TokenCounter()
        code_tokens = token_counter.count_tokens(code)
        
        if code_tokens > self.chunk_manager.max_tokens // 2:
            logger.info(f"Code is large ({code_tokens} tokens), using chunking")
            return await self._analyze_chunked_code(code, task, language, use_cache, **kwargs)
        
        # For smaller code, use a single prompt
        template_params = {"code": code}
        if language:
            template_params["language"] = language
        
        prompt = self.prompt_manager.get_prompt(task, **template_params)
        
        # Check cache
        if use_cache:
            cached_response = self.cache.get_cached_response(prompt, kwargs)
            if cached_response:
                logger.info("Using cached response")
                return cached_response
        
        # Generate completion
        response = await self.provider.generate_completion(prompt, **kwargs)
        
        # Cache response
        if use_cache and response:
            self.cache.cache_response(prompt, kwargs, response)
        
        return response
    
    async def _analyze_chunked_code(
        self, 
        code: str, 
        task: str, 
        language: Optional[str] = None,
        use_cache: bool = True,
        **kwargs
    ) -> str:
        """
        Analyze code that has been split into chunks
        
        Args:
            code: Code to analyze
            task: Analysis task
            language: Target language
            use_cache: Whether to use cached responses
            **kwargs: Additional parameters
            
        Returns:
            Combined analysis result
        """
        template_params = {}
        if language:
            template_params["language"] = language
        
        # Get chunked prompts
        prompts = self.prompt_manager.get_chunked_prompts(
            task, 
            code, 
            self.chunk_manager, 
            **template_params
        )
        
        logger.info(f"Analyzing code in {len(prompts)} chunks")
        
        # Process each chunk
        chunk_responses = []
        
        for i, prompt in enumerate(prompts):
            logger.info(f"Processing chunk {i+1}/{len(prompts)}")
            
            # Check cache
            if use_cache:
                cached_response = self.cache.get_cached_response(prompt, kwargs)
                if cached_response:
                    logger.info(f"Using cached response for chunk {i+1}")
                    chunk_responses.append(cached_response)
                    continue
            
            # Generate completion
            response = await self.provider.generate_completion(prompt, **kwargs)
            
            # Cache response
            if use_cache and response:
                self.cache.cache_response(prompt, kwargs, response)
            
            chunk_responses.append(response)
        
        # Combine chunk responses
        if task == "decompilation" or task == "structure_analysis":
            return self._combine_analysis_responses(chunk_responses)
        else:
            return "\n\n".join(chunk_responses)
    
    def _combine_analysis_responses(self, responses: List[str]) -> str:
        """
        Combine analysis responses from multiple chunks
        
        Args:
            responses: List of response chunks
            
        Returns:
            Combined response
        """
        if not responses:
            return ""
        
        # For simplicity, let's use the first chunk for high-level information
        # and append code/structure details from other chunks
        combined = responses[0]
        
        # Extract detailed sections from other chunks and append
        for response in responses[1:]:
            # Find code blocks or detailed analysis sections
            import re
            code_blocks = re.findall(r'```.*?```', response, re.DOTALL)
            
            for block in code_blocks:
                if block not in combined:
                    combined += f"\n\n{block}"
        
        return combined
    
    async def generate_specification(
        self, 
        analysis_results: List[str], 
        use_cache: bool = True,
        **kwargs
    ) -> str:
        """
        Generate a specification from analysis results
        
        Args:
            analysis_results: Analysis results
            use_cache: Whether to use cached responses
            **kwargs: Additional parameters
            
        Returns:
            Generated specification
        """
        logger.info("Generating specification from analysis results")
        
        # Combine analysis results
        combined_analysis = "\n\n".join(analysis_results)
        
        # Create messages for chat completion
        messages = [
            {"role": "system", "content": "You are an expert software architect tasked with creating a detailed specification."},
            {"role": "user", "content": f"Based on this software analysis:\n\n{combined_analysis}\n\nCreate a comprehensive specification document."}
        ]
        
        # Check cache
        prompt = json.dumps(messages)
        if use_cache:
            cached_response = self.cache.get_cached_response(prompt, kwargs)
            if cached_response:
                logger.info("Using cached response")
                return cached_response
        
        # Generate chat completion
        response = await self.provider.generate_chat_completion(messages, **kwargs)
        
        # Extract content from response
        if "choices" in response and len(response["choices"]) > 0:
            content = response["choices"][0]["message"]["content"]
            
            # Cache response
            if use_cache and content:
                self.cache.cache_response(prompt, kwargs, content)
            
            return content
        
        return ""
    
    async def mimic_software(
        self, 
        specification: str, 
        target_language: str,
        use_cache: bool = True,
        **kwargs
    ) -> str:
        """
        Generate a mimicked version of software
        
        Args:
            specification: Software specification
            target_language: Target programming language
            use_cache: Whether to use cached responses
            **kwargs: Additional parameters
            
        Returns:
            Generated code
        """
        logger.info(f"Mimicking software in {target_language}")
        
        # Check if specification is too large
        token_counter = TokenCounter()
        spec_tokens = token_counter.count_tokens(specification)
        
        if spec_tokens > self.chunk_manager.max_tokens // 2:
            logger.info(f"Specification is large ({spec_tokens} tokens), using chunking")
            # Split into sections and process each
            return await self._mimic_chunked_specification(specification, target_language, use_cache, **kwargs)
        
        # For smaller specifications, use a single prompt
        prompt = self.prompt_manager.get_prompt(
            "mimicry", 
            specification=specification, 
            language=target_language
        )
        
        # Check cache
        if use_cache:
            cached_response = self.cache.get_cached_response(prompt, kwargs)
            if cached_response:
                logger.info("Using cached response")
                return cached_response
        
        # Generate completion
        response = await self.provider.generate_completion(prompt, **kwargs)
        
        # Cache response
        if use_cache and response:
            self.cache.cache_response(prompt, kwargs, response)
        
        return response
    
    async def _mimic_chunked_specification(
        self, 
        specification: str, 
        target_language: str, 
        use_cache: bool = True,
        **kwargs
    ) -> str:
        """
        Mimic software from a chunked specification
        
        Args:
            specification: Software specification
            target_language: Target programming language
            use_cache: Whether to use cached responses
            **kwargs: Additional parameters
            
        Returns:
            Generated code
        """
        # Split specification into sections
        import re
        sections = re.split(r'#{1,3} ', specification)
        
        # First section is usually empty due to the split
        if sections and not sections[0].strip():
            sections = sections[1:]
        
        # Add the headers back to the sections
        for i in range(len(sections)):
            if i == 0:
                sections[i] = f"# {sections[i]}"
            else:
                sections[i] = f"## {sections[i]}"
        
        logger.info(f"Mimicking software in {len(sections)} sections")
        
        # Generate code for each section
        section_results = []
        
        for i, section in enumerate(sections):
            logger.info(f"Processing section {i+1}/{len(sections)}")
            
            # Create prompt for this section
            section_prompt = f"""
Generate {target_language} code for this section of the software specification:

{section}

Consider this in the context of the overall software being built, which is divided into multiple sections.
Focus on implementing this specific section with clean, maintainable code using modern {target_language} features and best practices.
"""
            
            # Check cache
            if use_cache:
                cached_response = self.cache.get_cached_response(section_prompt, kwargs)
                if cached_response:
                    logger.info(f"Using cached response for section {i+1}")
                    section_results.append(cached_response)
                    continue
            
            # Generate completion
            response = await self.provider.generate_completion(section_prompt, **kwargs)
            
            # Cache response
            if use_cache and response:
                self.cache.cache_response(section_prompt, kwargs, response)
            
            section_results.append(response)
        
        # Combine section results
        combined = "\n\n".join(section_results)
        
        # Generate a final integration prompt
        integration_prompt = f"""
I've generated code for different sections of a {target_language} application based on this specification:

{specification}

Here are the code sections I've created:

{combined}

Now please review the code sections, resolve any inconsistencies, and organize them into a cohesive structure.
Create any additional files needed for a complete application.
Make sure the components can work together properly.
"""
        
        # Check cache
        if use_cache:
            cached_response = self.cache.get_cached_response(integration_prompt, kwargs)
            if cached_response:
                logger.info("Using cached response for integration")
                return cached_response
        
        # Generate completion
        response = await self.provider.generate_completion(integration_prompt, **kwargs)
        
        # Cache response
        if use_cache and response:
            self.cache.cache_response(integration_prompt, kwargs, response)
        
        return response
    
    async def enhance_software(
        self, 
        original_code: str, 
        target_language: str,
        enhancements: List[str] = None,
        use_cache: bool = True,
        **kwargs
    ) -> str:
        """
        Enhance existing software with improvements
        
        Args:
            original_code: Original source code
            target_language: Target programming language (can be the same as original)
            enhancements: List of specific enhancements to apply
            use_cache: Whether to use cached responses
            **kwargs: Additional parameters
            
        Returns:
            Enhanced code
        """
        logger.info(f"Enhancing software in {target_language}")
        
        # Default enhancements if not provided
        if not enhancements:
            enhancements = [
                "Improve architecture using modern design patterns",
                "Add comprehensive error handling and validation",
                "Optimize performance where possible",
                "Enhance security best practices",
                "Improve code readability and documentation"
            ]
        
        enhancement_text = "\n".join([f"- {e}" for e in enhancements])
        
        # Create a prompt for enhancement
        prompt = f"""
You are tasked with enhancing the following software code:

```
{original_code}
```

Create an improved version in {target_language} with these enhancements:
{enhancement_text}

Your enhanced code should maintain the core functionality while implementing all the requested improvements.
Focus on making the code more robust, maintainable, and following modern best practices.
"""
        
        # Check if code is too large
        token_counter = TokenCounter()
        code_tokens = token_counter.count_tokens(original_code)
        
        if code_tokens > self.chunk_manager.max_tokens // 2:
            logger.info(f"Code is large ({code_tokens} tokens), will process in chunks")
            
            # First, analyze the code to understand its structure
            analysis = await self.analyze_code(
                original_code, 
                task="structure_analysis", 
                use_cache=use_cache,
                **kwargs
            )
            
            # Then generate a specification
            specification = await self.generate_specification(
                [analysis], 
                use_cache=use_cache,
                **kwargs
            )
            
            # Finally, mimic the software with enhancements
            return await self.mimic_software(
                specification, 
                target_language,
                use_cache=use_cache,
                **{**kwargs, "temperature": kwargs.get("temperature", 0.7) + 0.1}  # Slightly higher temperature for creativity
            )
        
        # For smaller code, process directly
        
        # Check cache
        if use_cache:
            cached_response = self.cache.get_cached_response(prompt, kwargs)
            if cached_response:
                logger.info("Using cached response")
                return cached_response
        
        # Generate completion
        response = await self.provider.generate_completion(prompt, **kwargs)
        
        # Cache response
        if use_cache and response:
            self.cache.cache_response(prompt, kwargs, response)
        
        return response

async def main():
    """Example usage of the LLM integration module"""
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM Integration for Software Analysis")
    parser.add_argument("--file", "-f", help="Path to source code file")
    parser.add_argument("--task", "-t", help="Task (decompile, analyze, specify, mimic, enhance)", default="analyze")
    parser.add_argument("--language", "-l", help="Target language for mimicry/enhancement", default="python")
    parser.add_argument("--provider", "-p", help="LLM provider (openai, anthropic)", default="openai")
    parser.add_argument("--model", "-m", help="LLM model", default=None)
    parser.add_argument("--no-cache", help="Disable caching", action="store_true")
    
    args = parser.parse_args()
    
    if not args.file and args.task != "help":
        print("Please specify a file with --file")
        return 1
    
    # Initialize LLM integration
    llm = LLMIntegration(
        provider=args.provider,
        model=args.model
    )
    
    # Show help if requested
    if args.task == "help":
        print("Available tasks:")
        print("  decompile  - Decompile and clean up source code")
        print("  analyze    - Analyze code structure")
        print("  specify    - Generate a specification from code")
        print("  mimic      - Create a mimicked version in a target language")
        print("  enhance    - Enhance code with improvements")
        return 0
    
    # Read file
    if args.file:
        try:
            with open(args.file, 'r') as f:
                code = f.read()
        except Exception as e:
            print(f"Error reading file: {str(e)}")
            return 1
    
    # Process based on task
    try:
        if args.task == "decompile":
            result = await llm.analyze_code(
                code, 
                task="decompilation", 
                use_cache=not args.no_cache
            )
            print(result)
            
        elif args.task == "analyze":
            result = await llm.analyze_code(
                code, 
                task="structure_analysis", 
                use_cache=not args.no_cache
            )
            print(result)
            
        elif args.task == "specify":
            analysis = await llm.analyze_code(
                code, 
                task="structure_analysis", 
                use_cache=not args.no_cache
            )
            specification = await llm.generate_specification(
                [analysis], 
                use_cache=not args.no_cache
            )
            print(specification)
            
        elif args.task == "mimic":
            analysis = await llm.analyze_code(
                code, 
                task="structure_analysis", 
                use_cache=not args.no_cache
            )
            specification = await llm.generate_specification(
                [analysis], 
                use_cache=not args.no_cache
            )
            mimicked = await llm.mimic_software(
                specification, 
                args.language,
                use_cache=not args.no_cache
            )
            print(mimicked)
            
        elif args.task == "enhance":
            enhanced = await llm.enhance_software(
                code, 
                args.language,
                use_cache=not args.no_cache
            )
            print(enhanced)
            
        else:
            print(f"Unknown task: {args.task}")
            return 1
    
    except Exception as e:
        print(f"Error processing task: {str(e)}")
        logger.error(f"Error processing task: {str(e)}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())