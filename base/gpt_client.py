import hashlib
import json
import logging
import threading
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
        cache_size: int = 100000,
        temperature: float = 0.1,
        max_retries: int = 3,
        cache_dir: str = ".cache",
    ):
        """
        Initialize the GPT client.

        Args:
            api_key: OpenAI API key
            cache_size: Maximum number of cached responses to keep
            temperature: Default temperature for GPT calls
            max_retries: Maximum number of retry attempts for failed calls
            cache_dir: Directory to store cache files
        """
        self.client = OpenAI(api_key=api_key)
        self.cache = PromptCache(cache_dir=cache_dir, max_size=cache_size)
        self.default_temperature = temperature
        self.max_retries = max_retries
        self._cache_lock = threading.RLock()  # Add a reentrant lock for thread safety
        self.cache_hits = 0
        self.cache_misses = 0
        self.api_calls = 0
        logger.info(
            f"GPTClient initialized with cache size {cache_size} in directory {cache_dir}"
        )

    def call_gpt(
        self,
        prompt: str,
        system_role: str,
        model: str = "gpt-4.1-mini",
        temperature: Optional[float] = None,
        refresh_cache: bool = False,
        response_format: Optional[Dict[str, str]] = None,
    ) -> Tuple[str, float]:
        """
        Make a GPT API call with caching.

        Args:
            prompt: The user prompt to send
            system_role: The system role message
            model: GPT model to use
            temperature: Temperature setting (uses default if None)
            refresh_cache: Whether to bypass and refresh the cache
            response_format: Optional response format specification

        Returns:
            Tuple of (response_content, cost)
        """
        start_time = time.time()
        temp = temperature if temperature is not None else self.default_temperature
        cache_check_time = 0.0
        cache_set_time = 0.0

        # Construct a cache key from relevant parameters
        cache_string = f"{model}:{system_role}:{prompt}:{temp}"
        cache_key = hashlib.sha256(cache_string.encode("utf-8")).hexdigest()

        prompt_preview = prompt[:50] + "..." if len(prompt) > 50 else prompt
        logger.debug(f"Call GPT for prompt: {prompt_preview}")

        # Check cache first, unless refresh_cache is True
        if not refresh_cache:
            cache_check_start = time.time()
            with self._cache_lock:  # Use lock when accessing the cache
                cached_result = self.cache.get(cache_key)
                if cached_result is not None:
                    self.cache_hits += 1
                    cache_check_time = time.time() - cache_check_start
                    total_time = time.time() - start_time
                    hit_rate = (
                        self.cache_hits / (self.cache_hits + self.cache_misses) * 100
                        if (self.cache_hits + self.cache_misses) > 0
                        else 0
                    )
                    logger.info(
                        f"Cache HIT ({hit_rate:.1f}% hit rate) in {cache_check_time:.3f}s, total time {total_time:.3f}s"
                    )
                    return cached_result, 0.0
                else:
                    logger.debug(f"Cache MISS in {cache_check_time:.3f}s")

            self.cache_misses += 1
            cache_check_time = time.time() - cache_check_start
            logger.debug(f"Cache check took {cache_check_time:.3f}s")

        try:
            api_start_time = time.time()
            messages = [
                {"role": "system", "content": system_role},
                {"role": "user", "content": prompt},
            ]

            completion_kwargs: Dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temp,
            }
            if response_format:
                completion_kwargs["response_format"] = response_format

            logger.info(f"Making API call to {model} (refresh_cache={refresh_cache})")
            self.api_calls += 1
            response = self.client.chat.completions.create(**completion_kwargs)
            api_time = time.time() - api_start_time
            logger.info(f"API call completed in {api_time:.3f}s")

            result = response.choices[0].message.content

            cache_set_start = time.time()
            with self._cache_lock:  # Use lock when modifying the cache
                self.cache.set(cache_key, result)
            cache_set_time = time.time() - cache_set_start
            logger.debug(f"Cache set took {cache_set_time:.3f}s")

            cost = AITokenPricing.calculate_cost(prompt + system_role, result, model)
            total_time = time.time() - start_time
            logger.info(
                f"Total GPT call took {total_time:.3f}s (API: {api_time:.3f}s, Cache operations: {cache_check_time + cache_set_time:.3f}s)"
            )

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
        model: str = "gpt-4.1-mini",
    ) -> Tuple[Union[str, Dict], float]:
        """
        Make a GPT API call with retry logic.

        Args:
            prompt: The user prompt to send
            system_role: The system role message
            temperature: Temperature setting (uses default if None)
            response_format: Optional response format specification
            validate_json: Whether to validate and parse JSON response
            model: GPT model to use
        Returns:
            Tuple of (response_content or parsed_json, cost)
        """
        total_cost = 0.0
        start_time = time.time()
        prompt_preview = prompt[:50] + "..." if len(prompt) > 50 else prompt
        logger.info(f"Call with retry for prompt: {prompt_preview}")

        for attempt in range(self.max_retries):
            try:
                logger.info(f"Attempt {attempt+1}/{self.max_retries}")
                attempt_start = time.time()
                response, cost = self.call_gpt(
                    prompt,
                    system_role,
                    model=model,
                    temperature=temperature,
                    refresh_cache=(attempt > 0),
                    response_format=response_format,
                )
                attempt_time = time.time() - attempt_start
                total_cost += cost

                if validate_json:
                    try:
                        json_start = time.time()
                        parsed = json.loads(response)
                        json_time = time.time() - json_start
                        logger.debug(f"JSON parsing took {json_time:.3f}s")
                        total_time = time.time() - start_time
                        logger.info(
                            f"Call with retry succeeded in {total_time:.3f}s (attempt {attempt+1}/{self.max_retries})"
                        )
                        return parsed, total_cost
                    except json.JSONDecodeError as e:
                        logger.warning(
                            f"JSON decode error on attempt {attempt+1}: {str(e)}"
                        )
                        logger.debug(f"Invalid JSON response: {response[:200]}...")
                        raise

                total_time = time.time() - start_time
                logger.info(
                    f"Call with retry succeeded in {total_time:.3f}s (attempt {attempt+1}/{self.max_retries})"
                )
                return response, total_cost

            except json.JSONDecodeError if validate_json else Exception as e:
                if attempt == self.max_retries - 1:
                    logger.warning(
                        f"Failed after {self.max_retries} attempts: {str(e)}"
                    )
                    raise
                backoff_time = 2**attempt
                logger.info(
                    f"Attempt {attempt+1} failed, retrying after {backoff_time}s backoff: {str(e)}"
                )
                time.sleep(backoff_time)  # Exponential backoff

        total_time = time.time() - start_time
        logger.error(
            f"Call with retry failed after {total_time:.3f}s and {self.max_retries} attempts"
        )
        raise RuntimeError(f"Failed after {self.max_retries} attempts")
