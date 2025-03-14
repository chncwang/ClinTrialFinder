#!/usr/bin/env python3
import argparse
import datetime
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add parent directory to Python path to import base module
sys.path.append(str(Path(__file__).parent.parent))
from base.clinical_trial import ClinicalTrial, ClinicalTrialsParser
from base.gpt_client import GPTClient
from base.trial_expert import GPTTrialFilter, TrialFailureReason

# Configure logging
log_file = f"filter_trials_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class EligibilityCriteriaError(Exception):
    """Custom exception for eligibility criteria format errors."""

    pass


@dataclass
class CriterionEvaluation:
    """Represents the evaluation result of a clinical trial criterion."""

    criterion: str
    reason: str
    eligibility: float


@dataclass
class TrialFailureReason:
    """Represents why a trial was deemed ineligible."""

    type: str  # "title" or "inclusion_criterion"
    message: str  # General failure message for title failures
    # Fields specific to inclusion criterion failures
    failed_condition: Optional[str] = None
    failed_criterion: Optional[str] = None
    failure_details: Optional[str] = None


def load_json_file(file_path: str) -> List[dict]:
    """Load and parse a JSON file."""
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"load_json_file: File '{file_path}' not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        logger.error(f"load_json_file: File '{file_path}' is not a valid JSON file.")
        sys.exit(1)


def save_json_file(data: List[Dict], output_path: str):
    """Save filtered trials to a JSON file."""
    try:
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"save_json_file: Results saved to {output_path}")
    except Exception as e:
        logger.error(f"save_json_file: Error saving results: {str(e)}")
        sys.exit(1)


def save_excluded_trials(excluded_trials: List[Dict], output_path: str):
    """Save excluded trials with their failure reasons to a JSON file."""
    excluded_path = output_path.replace(".json", "_excluded.json")
    try:
        with open(excluded_path, "w") as f:
            json.dump(excluded_trials, f, indent=2)
        logger.info(f"save_excluded_trials: Excluded trials saved to {excluded_path}")
    except Exception as e:
        logger.error(f"save_excluded_trials: Error saving excluded trials: {str(e)}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Filter clinical trials using GPT-4 based on custom conditions"
    )
    parser.add_argument(
        "json_file", help="Path to the JSON file containing clinical trials data"
    )
    parser.add_argument(
        "conditions",
        nargs="*",  # Changed from '+' to '*' to make it optional
        help="Optional conditions to filter trials (e.g., 'doesn\\'t have a measurable site')",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=f"filtered_trials_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--api-key",
        help="OpenAI API key (alternatively, set OPENAI_API_KEY environment variable)",
    )
    parser.add_argument(
        "--cache-size",
        type=int,
        default=100000,
        help="Maximum number of cached responses to keep",
    )
    # Add new arguments from parse_trials.py
    parser.add_argument(
        "--recruiting",
        action="store_true",
        help="Filter for only recruiting trials",
    )
    parser.add_argument(
        "--phase",
        help="Filter for trials of a specific phase (accepts both Arabic (1-4) and Roman numerals (I-IV))",
    )
    parser.add_argument(
        "--exclude-study-type",
        help="Exclude trials of a specific study type (e.g., 'Observational')",
    )
    parser.add_argument(
        "--recommendation-level",
        choices=["Strongly Recommended", "Recommended", "Neutral", "Not Recommended"],
        help="Filter trials by recommendation level (from previous analysis)",
    )

    args = parser.parse_args()

    # Get API key from arguments or environment
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if (
        not api_key and args.conditions
    ):  # Only require API key if conditions are provided
        logger.error(
            "OpenAI API key must be provided via --api-key or OPENAI_API_KEY environment variable when using conditions"
        )
        sys.exit(1)

    # Initialize GPT filter only if needed
    gpt_filter = None
    if args.conditions:
        gpt_filter = GPTTrialFilter(api_key, cache_size=args.cache_size)

    json_data = load_json_file(args.json_file)
    trials_parser = ClinicalTrialsParser(json_data)
    trials = trials_parser.trials

    logger.info(f"main: Loaded {len(trials)} trials from input file")

    # Apply filters
    if args.exclude_study_type:
        trials = trials_parser.get_trials_excluding_study_type(args.exclude_study_type)
        logger.info(
            f"main: Excluded study type '{args.exclude_study_type}': {len(trials)} trials remaining"
        )

    if args.recruiting:
        trials = trials_parser.get_recruiting_trials()
        logger.info(f"main: Filtered for recruiting trials: {len(trials)} found")

    if args.phase:
        phase_filtered = trials_parser.get_trials_by_phase(int(args.phase))
        trials = [t for t in trials if t in phase_filtered]
        logger.info(
            f"main: Filtered for phase {args.phase} trials: {len(trials)} found"
        )

    if args.recommendation_level:
        recommendation_filtered = trials_parser.get_trials_by_recommendation_level(
            args.recommendation_level
        )
        trials = [t for t in trials if t in recommendation_filtered]
        logger.info(
            f"main: Filtered for recommendation level '{args.recommendation_level}': {len(trials)} found"
        )

    filtered_trials = []
    excluded_trials = []
    total_trials = len(trials)
    total_cost = 0.0
    eligible_count = 0

    if args.conditions:
        logger.info(
            f"Processing {total_trials} trials with conditions: {args.conditions}"
        )

        for i, trial in enumerate(trials, 1):
            logger.info(
                f"Processing trial {i}/{total_trials}: {trial.identification.nct_id}"
            )

            # Now we unpack three values: eligibility, cost, and failure reason
            is_eligible, cost, failure_reason = gpt_filter.evaluate_trial(
                trial, args.conditions
            )
            total_cost += cost

            if is_eligible:
                # If the trial is eligible, we keep the entire record
                trial_dict = trial.to_dict()
                filtered_trials.append(trial_dict)
                eligible_count += 1
            else:
                # If the trial is ineligible, store failure details
                excluded_info = {
                    "nct_id": trial.identification.nct_id,
                    "brief_title": trial.identification.brief_title,
                    "eligibility_criteria": trial.eligibility.criteria,
                    "failure_type": failure_reason.type,
                    "failure_message": failure_reason.message,
                }

                # Add additional fields for inclusion criterion failures
                if failure_reason.type == "inclusion_criterion":
                    excluded_info.update(
                        {
                            "failed_condition": failure_reason.failed_condition,
                            "failed_criterion": failure_reason.failed_criterion,
                            "failure_details": failure_reason.failure_details,
                        }
                    )

                # Add analyzed trial fields if they exist
                if (
                    hasattr(trial, "recommendation_level")
                    and trial.recommendation_level
                ):
                    excluded_info["recommendation_level"] = trial.recommendation_level
                if hasattr(trial, "analysis_reason") and trial.analysis_reason:
                    excluded_info["analysis_reason"] = trial.analysis_reason

                excluded_trials.append(excluded_info)

            logger.info(
                f"main: Eligible trials so far: {eligible_count}/{i} processed, total cost: ${total_cost:.2f}"
            )
    else:
        # If no conditions provided, all trials that made it through the filters are eligible
        logger.info(
            "No conditions provided - keeping all trials that passed other filters"
        )
        filtered_trials = [trial.to_dict() for trial in trials]
        eligible_count = len(filtered_trials)

    # Save the passing (filtered) trials
    save_json_file(filtered_trials, args.output)

    # Save the excluded trials only if we did condition filtering
    if args.conditions:
        excluded_path = args.output.replace(".json", "_excluded.json")
        try:
            with open(excluded_path, "w") as f:
                json.dump(excluded_trials, f, indent=2)
            logger.info(f"Excluded trials saved to {excluded_path}")
        except Exception as e:
            logger.error(f"Error saving excluded trials: {str(e)}")
            sys.exit(1)

    logger.info(
        f"main: Final results: {eligible_count}/{total_trials} trials were eligible"
    )
    logger.info(f"main: Filtered trials saved to {args.output}")
    if args.conditions:
        logger.info(
            f"main: Excluded trials saved to {args.output.replace('.json', '_excluded.json')}"
        )
        logger.info(f"main: Total API cost: ${total_cost:.2f}")


if __name__ == "__main__":
    main()
