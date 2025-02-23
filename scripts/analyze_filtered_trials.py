import argparse
import json
import logging
import os
from datetime import datetime

from base.gpt_client import GPTClient
from base.perplexity import PerplexityClient
from base.trial_analyzer import analyze_drugs_and_get_recommendation

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


def process_trials_file(filename, gpt_client, perplexity_client):
    """Read and process the trials JSON file."""
    try:
        with open(filename, "r") as f:
            trials = json.load(f)

        logger.info(f"Starting to process {len(trials)} trials from {filename}")

        for trial in trials:
            log_trial(trial)
            # Analyze the trial
            recommendation, reason, total_cost = analyze_drugs_and_get_recommendation(
                novel_drugs=trial.get("novel_drugs", []),
                clinical_record=trial.get("clinical_record", ""),
                trial=trial,
                perplexity_client=perplexity_client,
                gpt_client=gpt_client,
            )
            logger.info(f"Recommendation: {recommendation}")
            logger.info(f"Reason: {reason}")
            logger.info(f"Total Cost: ${total_cost:.6f}")

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
        "trials_json_file", help="Path to the input JSON file containing trial data"
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
        "--perplexity-api-key",
        help="Perplexity API key",
        default=os.getenv("PERPLEXITY_API_KEY"),
    )
    args = parser.parse_args()

    # Initialize clients
    gpt_client = GPTClient(api_key=args.openai_api_key)
    perplexity_client = PerplexityClient(api_key=args.perplexity_api_key)

    process_trials_file(args.trials_json_file, gpt_client, perplexity_client)
    # You can add a function to process the clinical record file if needed
    # process_clinical_record_file(args.clinical_record_file)
