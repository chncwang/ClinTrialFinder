import logging
import os
import pickle
import time
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple

import requests

from base.pricing import AITokenPricing

logger = logging.getLogger(__name__)


class PerplexityClient:
    """Client for interacting with the Perplexity AI API."""

    BASE_URL = "https://api.perplexity.ai/chat/completions"
    CACHE_FILE = "perplexity_cache.pkl"
    MAX_CACHE_SIZE = 1000  # Set maximum cache size
    EXPIRATION_DURATION = timedelta(
        days=1
    ).total_seconds()  # Set expiration duration to one day

    def __init__(self, api_key: str):
        """Initialize the Perplexity client.

        Args:
            api_key (str): The Perplexity API key
        """
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self.cache = self.load_cache()

    def load_cache(self) -> Dict:
        """Load cache from a file."""
        if os.path.exists(self.CACHE_FILE):
            with open(self.CACHE_FILE, "rb") as f:
                return pickle.load(f)
        return {}

    def save_cache(self):
        """Save cache to a file."""
        with open(self.CACHE_FILE, "wb") as f:
            pickle.dump(self.cache, f)

    def is_expired(self, timestamp: float) -> bool:
        """Check if a cache entry is expired."""
        return (time.time() - timestamp) > self.EXPIRATION_DURATION

    def get_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "sonar-pro",
        max_tokens: int = 1000,
        temperature: float = 0.2,
        top_p: float = 0.9,
    ) -> Tuple[Optional[str], Optional[List[str]], float]:
        """Get a completion from the Perplexity API.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model to use (default: sonar-pro)
            max_tokens: Maximum tokens in response (default: 1000)
            temperature: Temperature parameter (default: 0.2)
            top_p: Top p parameter (default: 0.9)

        Returns:
            Tuple of (completion_text, citations, cost) where:
                - completion_text is the response text if successful, None if failed
                - citations is a list of citation URLs if available, None if not
                - cost is the estimated cost in USD
        """
        # Create a cache key based on the input parameters
        cache_key = (
            tuple((msg["role"], msg["content"]) for msg in messages),
            model,
            max_tokens,
            temperature,
            top_p,
        )

        # Check if the result is already in the cache and not expired
        if cache_key in self.cache:
            completion_text, citations, timestamp = self.cache[cache_key]
            if not self.is_expired(timestamp):
                logger.debug("Cache hit for get_completion")
                return completion_text, citations, 0.0
            else:
                logger.debug("Cache entry expired for get_completion")
                del self.cache[cache_key]

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "return_images": False,
            "return_related_questions": False,
            "stream": False,
        }

        try:
            response = requests.post(self.BASE_URL, json=payload, headers=self.headers)
            response.raise_for_status()
            result = response.json()
            logger.debug(f"get_completion: Number of choices: {len(result['choices'])}")
            completion_text = result["choices"][0]["message"]["content"]

            # Extract citations if available
            citations = result.get("citations")

            # Calculate cost based on input and output tokens
            prompt_text = " ".join(msg["content"] for msg in messages)
            cost = AITokenPricing.calculate_cost(
                prompt_text, completion_text, model=model
            )

            # Store the result in the cache with the current timestamp
            self.cache[cache_key] = (completion_text, citations, time.time())

            # Ensure the cache does not exceed the maximum size
            if len(self.cache) > self.MAX_CACHE_SIZE:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]

            self.save_cache()  # Save the updated cache to the file

            return completion_text, citations, cost
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling Perplexity AI API: {e}")
            return None, None, 0.0
