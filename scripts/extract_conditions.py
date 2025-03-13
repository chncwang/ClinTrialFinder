import argparse
import logging
import os
from datetime import datetime

from base.disease_expert import extract_conditions_from_record
from base.gpt_client import GPTClient
from base.utils import read_input_file

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
    """Process the clinical record file and extract conditions."""
    try:
        # Read the clinical record
        clinical_record = read_input_file(clinical_record_file)
        logger.info(f"Read clinical record from {clinical_record_file}")

        # Initialize GPT client
        gpt_client = GPTClient(api_key=openai_api_key)

        # Extract conditions
        conditions, cost = extract_conditions_from_record(
            clinical_record=clinical_record,
            gpt_client=gpt_client,
        )

        # Log results
        logger.info(f"Extracted {len(conditions)} conditions:")
        for condition in conditions:
            logger.info(f"- {condition}")
        logger.info(f"Total API cost: ${cost:.6f}")

    except Exception as e:
        logger.error(f"Error processing clinical record: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract conditions from a clinical record file."
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
