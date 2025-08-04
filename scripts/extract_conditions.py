import argparse
import os
from datetime import datetime
from loguru import logger

from base.disease_expert import extract_conditions_from_record
from base.gpt_client import GPTClient

# Configure loguru with file output
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logger.add(f"extract_conditions_{timestamp}.log", format="{time:YYYY-MM-DD HH:mm:ss} - {level} - {message}")


def process_clinical_record(
    clinical_record_file: str,
    openai_api_key: str | None = None,
    refresh_cache: bool = False,
) -> None:
    """Process the clinical record file and extract clinical history in chronological order."""
    try:
        # Validate API key
        if not openai_api_key:
            raise ValueError("OpenAI API key is required. Please provide it via --openai-api-key argument or OPENAI_API_KEY environment variable.")
        
        # Initialize GPT client
        gpt_client = GPTClient(api_key=openai_api_key)

        # Extract clinical history
        history = extract_conditions_from_record(
            clinical_record_file, gpt_client, refresh_cache
        )

        # Log history
        logger.info(f"Extracted {len(history)} clinical history items:")
        for item in history:
            logger.info(f"- {item}")

    except Exception as e:
        logger.error(f"Error processing clinical record: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract clinical history from a clinical record file."
    )
    parser.add_argument(
        "clinical_record_file", help="Path to the input clinical record file"
    )
    parser.add_argument(
        "--openai-api-key",
        help="OpenAI API key",
        default=os.getenv("OPENAI_API_KEY"),
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Refresh the cache of GPT responses",
    )
    args = parser.parse_args()

    process_clinical_record(
        clinical_record_file=args.clinical_record_file,
        openai_api_key=args.openai_api_key,
        refresh_cache=args.refresh_cache,
    )
