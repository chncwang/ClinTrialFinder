import json
import logging
import time
from typing import Any, Dict, Optional, Tuple, Union

from openai import OpenAI

from base.pricing import AITokenPricing
from base.prompt_cache import PromptCache

logger = logging.getLogger(__name__)


class GPTClient:
    """Base class for making GPT API calls with retry logic and caching."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        cache_size: int = 100000,
        temperature: float = 0.1,
        max_retries: int = 3,
    ):
        """
        Initialize the GPT client.

        Args:
            api_key: OpenAI API key
            model: GPT model to use
            cache_size: Maximum number of cached responses to keep
            temperature: Default temperature for GPT calls
            max_retries: Maximum number of retry attempts for failed calls
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.cache = PromptCache(max_size=cache_size)
        self.default_temperature = temperature
        self.max_retries = max_retries

    def call_gpt(
        self,
        prompt: str,
        system_role: str,
        temperature: Optional[float] = None,
        refresh_cache: bool = False,
        response_format: Optional[Dict[str, str]] = None,
    ) -> Tuple[str, float]:
        """
        Make a GPT API call with caching.

        Args:
            prompt: The user prompt to send
            system_role: The system role message
            temperature: Temperature setting (uses default if None)
            refresh_cache: Whether to bypass and refresh the cache
            response_format: Optional response format specification

        Returns:
            Tuple of (response_content, cost)
        """
        temp = temperature if temperature is not None else self.default_temperature

        # Check cache first, unless refresh_cache is True
        if not refresh_cache:
            cached_result = self.cache.get(prompt, temp)
            if cached_result is not None:
                logger.debug("Using cached result")
                return cached_result, 0.0

        try:
            messages = [
                {"role": "system", "content": system_role},
                {"role": "user", "content": prompt},
            ]

            completion_kwargs: Dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": temp,
            }
            if response_format:
                completion_kwargs["response_format"] = response_format

            response = self.client.chat.completions.create(**completion_kwargs)

            result = response.choices[0].message.content
            self.cache.set(prompt, temp, result)

            cost = AITokenPricing.calculate_cost(prompt + system_role, result)
            return result, cost

        except Exception as e:
            logger.error(f"Error in GPT call: {str(e)}")
            raise

    def call_with_retry(
        self,
        prompt: str,
        system_role: str,
        temperature: Optional[float] = None,
        response_format: Optional[Dict[str, str]] = None,
        validate_json: bool = False,
    ) -> Tuple[Union[str, Dict], float]:
        """
        Make a GPT API call with retry logic.

        Args:
            prompt: The user prompt to send
            system_role: The system role message
            temperature: Temperature setting (uses default if None)
            response_format: Optional response format specification
            validate_json: Whether to validate and parse JSON response

        Returns:
            Tuple of (response_content or parsed_json, cost)
        """
        total_cost = 0.0

        for attempt in range(self.max_retries):
            try:
                response, cost = self.call_gpt(
                    prompt,
                    system_role,
                    temperature=temperature,
                    refresh_cache=(attempt > 0),
                    response_format=response_format,
                )
                total_cost += cost

                if validate_json:
                    parsed = json.loads(response)
                    return parsed, total_cost
                return response, total_cost

            except json.JSONDecodeError if validate_json else Exception as e:
                if attempt == self.max_retries - 1:
                    logger.warning(
                        f"Failed after {self.max_retries} attempts: {str(e)}"
                    )
                    raise
                time.sleep(2**attempt)  # Exponential backoff

        raise RuntimeError(f"Failed after {self.max_retries} attempts")
