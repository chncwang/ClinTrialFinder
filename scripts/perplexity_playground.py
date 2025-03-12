#!/usr/bin/env python3
import argparse
import logging
import os
import sys
from pathlib import Path

# Add parent directory to Python path to import modules
sys.path.append(str(Path(__file__).parent.parent))
from base.perplexity import PerplexityClient

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(handler)


def main():
    parser = argparse.ArgumentParser(description="Perplexity API Playground")
    parser.add_argument(
        "--perplexity-api-key",
        help="Perplexity API key (alternatively, set PERPLEXITY_API_KEY environment variable)",
    )
    parser.add_argument(
        "--system-prompt",
        help="System prompt for the conversation",
        default="You are a helpful AI assistant.",
    )
    parser.add_argument(
        "--prompt",
        help="Direct prompt to send to Perplexity (optional)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1000,
        help="Maximum number of tokens in the response",
    )

    args = parser.parse_args()

    # Get API key from argument or environment
    api_key = args.perplexity_api_key or os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        logger.error(
            "Perplexity API key must be provided via --perplexity-api-key or PERPLEXITY_API_KEY environment variable"
        )
        sys.exit(1)

    # Get the prompt from command line
    if not args.prompt:
        logger.error("Must specify --prompt")
        sys.exit(1)

    user_prompt = args.prompt

    # Initialize Perplexity client
    perplexity_client = PerplexityClient(api_key)

    # Prepare messages
    messages = [
        {"role": "system", "content": args.system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # Call the API
    logger.info("Sending request to Perplexity API...")
    completion, citations, cost = perplexity_client.get_completion(
        messages, max_tokens=args.max_tokens
    )

    if completion is None:
        logger.error("Failed to get response from Perplexity")
        sys.exit(1)

    # Print results
    logger.info("\n=== Response ===")
    logger.info(completion)

    if citations:
        logger.info("\n=== Citations ===")
        for i, citation in enumerate(citations, 1):
            logger.info(f"{i}. {citation}")

    logger.info(f"\nCost: ${cost:.6f}")


if __name__ == "__main__":
    main()
