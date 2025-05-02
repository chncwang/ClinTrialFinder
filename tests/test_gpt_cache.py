#!/usr/bin/env python3
import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Add parent directory to Python path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from base.gpt_client import GPTClient

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("test_gpt_cache.log"),
    ],
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Test GPT cache functionality")
    parser.add_argument(
        "--openai-api-key",
        help="OpenAI API key (alternatively, set OPENAI_API_KEY environment variable)",
    )
    parser.add_argument(
        "--cache-dir",
        default=".cache",
        help="Directory to store cache files",
    )
    parser.add_argument(
        "--cache-size",
        type=int,
        default=10000,
        help="Maximum number of cached responses to keep",
    )
    args = parser.parse_args()

    # Get API key from argument or environment
    api_key = args.openai_api_key or os.getenv("OPENAI_API_KEY")

    if not api_key:
        logger.error(
            "OpenAI API key must be provided via --openai-api-key or OPENAI_API_KEY environment variable"
        )
        sys.exit(1)

    # Initialize GPT client
    gpt_client = GPTClient(
        api_key=api_key,
        cache_size=args.cache_size,
        cache_dir=args.cache_dir,
    )

    # Run tests
    test_basic_caching(gpt_client)
    test_repeated_calls(gpt_client)
    

def test_basic_caching(gpt_client):
    """Test that the cache works for identical prompts."""
    logger.info("=== Testing basic caching ===")
    
    prompt = "What is 2+2?"
    system_role = "You are a helpful assistant."
    
    logger.info("First call (should miss cache):")
    start_time = time.time()
    result1, cost1 = gpt_client.call_gpt(prompt, system_role)
    time1 = time.time() - start_time
    
    logger.info(f"Result: {result1[:50]}... (cost: ${cost1:.6f})")
    logger.info(f"Time taken: {time1:.3f}s")
    
    logger.info("\nSecond call with same prompt (should hit cache):")
    start_time = time.time()
    result2, cost2 = gpt_client.call_gpt(prompt, system_role)
    time2 = time.time() - start_time
    
    logger.info(f"Result: {result2[:50]}... (cost: ${cost2:.6f})")
    logger.info(f"Time taken: {time2:.3f}s")
    logger.info(f"Cache speedup: {time1/time2:.1f}x faster")
    
    if result1 == result2 and cost2 == 0.0:
        logger.info("✅ Cache working correctly!")
    else:
        logger.error("❌ Cache not working correctly!")


def test_repeated_calls(gpt_client):
    """Test that cache performs well with repeated calls."""
    logger.info("\n=== Testing repeated calls ===")
    
    system_role = "You are a helpful assistant."
    
    # First, make 5 unique calls to populate cache
    unique_prompts = [
        f"What is {i} + {i+1}?" for i in range(5)
    ]
    
    logger.info("Populating cache with 5 unique prompts:")
    for prompt in unique_prompts:
        result, cost = gpt_client.call_gpt(prompt, system_role)
        logger.info(f"  Prompt: {prompt}, Cost: ${cost:.6f}")
    
    # Now test with a mix of cache hits and misses
    test_prompts = [
        # These should hit cache
        "What is 0 + 1?",
        "What is 1 + 2?",
        # This should miss cache
        "What is 10 + 11?", 
        # This should hit cache
        "What is 2 + 3?",
    ]
    
    logger.info("\nTesting with mix of cache hits and misses:")
    total_cost = 0
    for prompt in test_prompts:
        start_time = time.time()
        result, cost = gpt_client.call_gpt(prompt, system_role)
        elapsed = time.time() - start_time
        total_cost += cost
        cache_status = "MISS" if cost > 0 else "HIT"
        logger.info(f"  Prompt: {prompt}, Cache: {cache_status}, Time: {elapsed:.3f}s, Cost: ${cost:.6f}")
    
    logger.info(f"\nTotal cost for test: ${total_cost:.6f}")
    logger.info(f"Cache hit rate: {gpt_client.cache_hits}/{gpt_client.cache_hits + gpt_client.cache_misses} ({gpt_client.cache_hits/(gpt_client.cache_hits + gpt_client.cache_misses)*100:.1f}%)")


if __name__ == "__main__":
    main()