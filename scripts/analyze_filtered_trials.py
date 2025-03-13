import argparse
import json
import logging
import os
from datetime import datetime

from base.clinical_trial import ClinicalTrial
from base.gpt_client import GPTClient
from base.perplexity import PerplexityClient
from base.trial_analyzer import (
    RecommendationLevel,
    analyze_drugs_and_get_recommendation,
)
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
file_handler = logging.FileHandler(f"analyze_filtered_trials_{timestamp}.log")
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
)
logger.addHandler(file_handler)

# Set up logger for base.trial_analyzer
trial_analyzer_logger = logging.getLogger("base.trial_analyzer")
trial_analyzer_logger.setLevel(logging.INFO)

# Add handlers to the trial_analyzer_logger if needed
trial_analyzer_logger.addHandler(stream_handler)
trial_analyzer_logger.addHandler(file_handler)


def process_trials_file(filename, gpt_client, perplexity_client, clinical_record):
    """Read and process the trials JSON file."""
    try:
        with open(filename, "r") as f:
            trials = json.load(f)

        logger.info(f"Starting to process {len(trials)} trials from {filename}")

        # Prepare a list to store updated trial data
        updated_trials = []

        for trial_dict in trials:
            trial = ClinicalTrial(trial_dict)
            logger.info(f"Processing trial: {trial}")
            # Analyze the trial
            recommendation: RecommendationLevel
            reason: str
            drug_analyses: dict[str, str]
            total_cost: float
            recommendation, reason, drug_analyses, total_cost = (
                analyze_drugs_and_get_recommendation(
                    clinical_record,
                    trial=trial,
                    perplexity_client=perplexity_client,
                    gpt_client=gpt_client,
                )
            )
            logger.info(f"Recommendation: {recommendation}")
            logger.info(f"Reason: {reason}")
            logger.info(f"Drug Analyses: {drug_analyses}")
            logger.info(f"Total Cost: ${total_cost:.6f}")

            # Add recommendation, reason, and drug analysis to the trial data
            trial_dict["recommendation_level"] = str(recommendation)
            trial_dict["reason"] = reason
            trial_dict["drug_analysis"] = drug_analyses
            updated_trials.append(trial_dict)

        # Write the analyzed trials to a new JSON file
        output_filename = f"analyzed_{os.path.basename(filename)}"
        with open(output_filename, "w") as f:
            json.dump(updated_trials, f, indent=4)

        logger.info(
            f"Finished processing all trials. Output written to {output_filename}"
        )

    except FileNotFoundError:
        logger.error(f"File not found: {filename}")
        raise
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from {filename}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise


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

    clinical_record = read_input_file(args.clinical_record_file)
    process_trials_file(
        args.trials_json_file, gpt_client, perplexity_client, clinical_record
    )
