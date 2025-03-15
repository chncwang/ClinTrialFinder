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
from base.trial_expert import (
    GPTTrialFilter,
    TrialFailureReason,
    process_trials_with_conditions,
)
from base.utils import load_json_list_file, save_json_list_file

# Configure logging
log_file = f"filter_trials_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Update logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout),  # Explicitly use stdout
    ],
    force=True,  # Force reconfiguration of the root logger
)

# Get the root logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Get the script's logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
    parser.add_argument(
        "--has-novel-drug-analysis",
        action="store_true",
        help="Filter for trials that have non-empty novel drug analysis results",
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

    json_data = load_json_list_file(args.json_file)
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

    if args.has_novel_drug_analysis:
        novel_drug_filtered = trials_parser.get_trials_with_novel_drug_analysis()
        trials = [t for t in trials if t in novel_drug_filtered]
        logger.info(
            f"main: Filtered for trials with novel drug analysis: {len(trials)} found"
        )

    # Process trials with conditions and save results
    total_cost, eligible_count = process_trials_with_conditions(
        trials, args.conditions, args.output, gpt_filter
    )

    if args.conditions:
        logger.info(f"main: Total API cost: ${total_cost:.2f}")


if __name__ == "__main__":
    main()
