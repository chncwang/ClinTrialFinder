import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

from base.clinical_trial import ClinicalTrial
from base.disease_expert import extract_conditions_from_record
from base.gpt_client import GPTClient
from base.trial_analyzer import GPTTrialFilter

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_trials(trials_file: str) -> List[Dict[str, Any]]:
    """Load trials from a JSON file."""
    with open(trials_file, "r") as f:
        return json.load(f)


def filter_trials_by_conditions(
    trials: List[Dict[str, Any]], conditions: List[str], gpt_filter: GPTTrialFilter
) -> List[Dict[str, Any]]:
    """Filter trials based on extracted conditions using GPT evaluation."""
    filtered_trials = []

    # Log the conditions we're looking for
    logger.info("Looking for conditions:")
    for condition in conditions:
        logger.info(f"- {condition}")

    for trial_dict in trials:
        # Create ClinicalTrial object directly
        trial = ClinicalTrial(trial_dict)

        # Evaluate trial using GPTTrialFilter
        is_eligible, probability, failure_reason = gpt_filter.evaluate_trial(
            trial, conditions
        )

        if is_eligible:
            logger.info(
                f"Found matching trial: {trial.identification.nct_id} "
                f"(probability: {probability:.2f})"
            )
            # Add probability to the trial data
            trial_dict["suitability_probability"] = probability
            filtered_trials.append(trial_dict)
        else:
            logger.debug(
                f"Trial {trial.identification.nct_id} excluded: "
                f"{failure_reason.message if failure_reason else 'Unknown reason'}"
            )

    # Sort filtered trials by suitability probability
    filtered_trials.sort(
        key=lambda x: x.get("suitability_probability", 0), reverse=True
    )
    return filtered_trials


def main():
    parser = argparse.ArgumentParser(
        description="Filter trials based on clinical record conditions"
    )
    parser.add_argument("clinical_record", help="Path to the clinical record file")
    parser.add_argument("trials_file", help="Path to the trials JSON file")
    parser.add_argument(
        "--output",
        "-o",
        help="Output file path for filtered trials",
        default="filtered_trials.json",
    )
    parser.add_argument(
        "--api-key",
        help="OpenAI API key. If not provided, will try to get from OPENAI_API_KEY environment variable",
    )
    parser.add_argument(
        "--cache-size",
        type=int,
        default=100000,
        help="Size of the GPT response cache",
    )

    args = parser.parse_args()

    # Get API key from command line or environment
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OpenAI API key not found. Please provide it via --api-key argument or OPENAI_API_KEY environment variable"
        )

    # Initialize GPT client and filter
    gpt_client = GPTClient(api_key=api_key)
    gpt_filter = GPTTrialFilter(api_key=api_key, cache_size=args.cache_size)

    # Extract conditions from clinical record
    logger.info(f"Reading clinical record from {args.clinical_record}")
    history_items = extract_conditions_from_record(args.clinical_record, gpt_client)
    logger.info(f"Extracted {len(history_items)} history items")

    # Load trials
    logger.info(f"Loading trials from {args.trials_file}")
    trials = load_trials(args.trials_file)
    logger.info(f"Loaded {len(trials)} trials")

    # Filter trials
    logger.info("Filtering trials based on conditions")
    filtered_trials = filter_trials_by_conditions(trials, history_items, gpt_filter)
    logger.info(f"Found {len(filtered_trials)} matching trials")

    # Save filtered trials
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(filtered_trials, f, indent=2)
    logger.info(f"Saved filtered trials to {output_path}")


if __name__ == "__main__":
    main()
