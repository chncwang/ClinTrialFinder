import argparse
import logging
import os
from datetime import datetime

from base.disease_expert import extract_conditions_from_record
from base.gpt_client import GPTClient

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create stream handler
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
)
logger.addHandler(stream_handler)

# Create file handler with timestamp in the file name
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
file_handler = logging.FileHandler(f"extract_conditions_{timestamp}.log")
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
)
logger.addHandler(file_handler)


def process_clinical_record(
    clinical_record_file: str,
    openai_api_key: str | None = None,
) -> None:
    """Process the clinical record file and extract clinical history in chronological order."""
    try:
        # Initialize GPT client
        gpt_client = GPTClient(api_key=openai_api_key)

        # Extract clinical history
        history = extract_conditions_from_record(clinical_record_file, gpt_client)

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
    args = parser.parse_args()

    process_clinical_record(
        clinical_record_file=args.clinical_record_file,
        openai_api_key=args.openai_api_key,
    )
