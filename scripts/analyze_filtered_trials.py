import argparse
import json
import logging
from datetime import datetime

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create stream handler
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
)
logger.addHandler(stream_handler)


def log_trial(trial):
    """Log important information about a single clinical trial."""
    nct_id = trial["identification"]["nct_id"]
    title = trial["identification"]["brief_title"]
    status = trial["status"]["overall_status"]
    enrollment = trial["design"]["enrollment"]

    logger.info(f"Processing trial {nct_id}")
    logger.info(f"Title: {title}")
    logger.info(f"Status: {status}")
    logger.info(f"Enrollment: {enrollment}")
    logger.info("-" * 80)  # Separator line


def process_trials_file(filename):
    """Read and process the trials JSON file."""
    try:
        with open(filename, "r") as f:
            trials = json.load(f)

        logger.info(f"Starting to process {len(trials)} trials from {filename}")

        for trial in trials:
            log_trial(trial)

        logger.info(f"Finished processing all trials")

    except FileNotFoundError:
        logger.error(f"File not found: {filename}")
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from {filename}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process clinical trials from a JSON file."
    )
    parser.add_argument(
        "input_file", help="Path to the input JSON file containing trial data"
    )
    args = parser.parse_args()

    process_trials_file(args.input_file)
