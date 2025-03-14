import argparse
import json
import logging
import os
from datetime import datetime

from base.clinical_trial import ClinicalTrial
from base.gpt_client import GPTClient
from base.perplexity import PerplexityClient
from base.trial_expert import RecommendationLevel, analyze_drugs_and_get_recommendation
from base.utils import read_input_file

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up file handler
file_handler = logging.FileHandler("analyze_filtered_trials.log")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)

# Set up stream handler for console output
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(logging.Formatter("%(message)s"))

# Set up logger for base.trial_expert
trial_expert_logger = logging.getLogger("base.trial_expert")
trial_expert_logger.setLevel(logging.INFO)

# Add handlers to the trial_expert_logger if needed
trial_expert_logger.addHandler(stream_handler)
trial_expert_logger.addHandler(file_handler)


def process_trials_file(
    filename, gpt_client, perplexity_client, clinical_record, output_filename=None
):
    """Read and process the trials JSON file."""
    try:
        with open(filename, "r") as f:
            trials = json.load(f)

        logger.info(f"Starting to process {len(trials)} trials from {filename}")

        # Prepare a list to store updated trial data
        updated_trials = []

        for index, trial_dict in enumerate(trials, start=1):
            trial = ClinicalTrial(trial_dict)
            logger.info(f"Processing trial {index}/{len(trials)}: {trial}")
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
        if output_filename is None:
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
    parser.add_argument(
        "--output",
        help="Output JSON file path (default: analyzed_[input_filename])",
    )
    args = parser.parse_args()

    # Initialize clients
    gpt_client = GPTClient(api_key=args.openai_api_key)
    perplexity_client = PerplexityClient(api_key=args.perplexity_api_key)

    clinical_record = read_input_file(args.clinical_record_file)
    process_trials_file(
        args.trials_json_file,
        gpt_client,
        perplexity_client,
        clinical_record,
        args.output,
    )
