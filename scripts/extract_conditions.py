import argparse
import logging
import os
from datetime import datetime

from base.gpt_client import GPTClient
from base.utils import parse_json_response, read_input_file

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
        # Read the clinical record
        clinical_record = read_input_file(clinical_record_file)
        logger.info(f"Read clinical record from {clinical_record_file}")

        # Get current time for relative time calculations
        current_time = datetime.now()
        current_time_str = current_time.strftime("%Y-%m-%d")

        # Initialize GPT client
        gpt_client = GPTClient(api_key=openai_api_key)

        # Extract clinical history in chronological order
        prompt = (
            "Extract the patient's clinical history from the following clinical record in chronological order. "
            "Return the results as a JSON array of strings, where each string represents an event, condition, or treatment. "
            f"Today's date is {current_time_str}. EVERY condition, event, and treatment MUST include a relative time reference "
            "(e.g., '9 months ago', '2 weeks ago', 'currently', etc.). Do not use absolute dates.\n\n"
            "Include:\n"
            "- Basic patient information (current age, gender)\n"
            "- Medical conditions with when they occurred/were diagnosed\n"
            "- Treatments and procedures with start and end times\n"
            "- Response to treatments with timing\n"
            "- Current clinical status with 'currently' or 'present'\n\n"
            "Example format:\n"
            "[\n"
            '  "Currently 35 years old",\n'
            '  "Male",\n'
            '  "Diagnosed with Type 2 Diabetes 2 years ago",\n'
            '  "Started chemotherapy 6 months ago",\n'
            '  "Developed metastasis 3 months ago",\n'
            '  "Had partial response to immunotherapy from 4 months ago until 2 months ago",\n'
            '  "Currently receiving drug Z for the past 2 weeks",\n'
            '  "Currently has ECOG PS of 1"\n'
            "]\n\n"
            "Rules:\n"
            "1. Every medical condition must include when it was diagnosed/occurred\n"
            "2. Every treatment must include when it started and ended (or 'currently' if ongoing)\n"
            "3. Every status must include 'currently' if it's a present condition\n"
            "4. Use the most precise time reference possible (e.g., '2 weeks ago' instead of 'recently')\n\n"
            f"Clinical Record:\n{clinical_record}\n\n"
            "Return only the JSON array, no additional text or explanation."
        )

        completion, cost = gpt_client.call_gpt(
            prompt=prompt,
            system_role="You are a medical expert specialized in extracting clinical history from medical records.",
            temperature=0.1,
            model="gpt-4o",
        )

        if completion:
            history, total_cost = parse_json_response(
                completion, expected_type=list, gpt_client=gpt_client, cost=cost
            )

            # Log history
            logger.info(f"Extracted {len(history)} clinical history items:")
            for item in history:
                logger.info(f"- {item}")

            logger.info(f"Total API cost: ${total_cost:.6f}")

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
