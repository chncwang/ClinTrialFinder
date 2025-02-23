import logging
from typing import Any, Dict, List, Optional, Tuple

import requests

from base.pricing import AITokenPricing

logger = logging.getLogger(__name__)


class PerplexityClient:
    """Client for interacting with the Perplexity AI API."""

    BASE_URL = "https://api.perplexity.ai/chat/completions"

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

    def get_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "sonar-pro",
        max_tokens: int = 1000,
        temperature: float = 0.2,
        top_p: float = 0.9,
    ) -> Tuple[Optional[str], float]:
        """Get a completion from the Perplexity API.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model to use (default: sonar-pro)
            max_tokens: Maximum tokens in response (default: 1000)
            temperature: Temperature parameter (default: 0.2)
            top_p: Top p parameter (default: 0.9)

        Returns:
            Tuple of (completion_text, cost) where:
                - completion_text is the response text if successful, None if failed
                - cost is the estimated cost in USD
        """
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
            completion_text = result["choices"][0]["message"]["content"]

            # Calculate cost based on input and output tokens
            prompt_text = " ".join(msg["content"] for msg in messages)
            cost = AITokenPricing.calculate_cost(
                prompt_text, completion_text, model=model
            )

            return completion_text, cost
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling Perplexity AI API: {e}")
            return None, 0.0
