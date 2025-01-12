#!/usr/bin/env python3
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List

# Add parent directory to Python path to import base module
sys.path.append(str(Path(__file__).parent.parent))
from base.clinical_trial import ClinicalTrialsParser

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(handler)


def load_json_file(file_path: str) -> List[dict]:
    """Load and parse a JSON file."""
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"load_json_file: Error: File '{file_path}' not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        logger.error(
            f"load_json_file: Error: File '{file_path}' is not a valid JSON file."
        )
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Parse and analyze clinical trials data."
    )
    parser.add_argument(
        "json_file", help="Path to the JSON file containing clinical trials data"
    )
    parser.add_argument("--nct-id", help="Get trial by NCT ID")
    parser.add_argument(
        "--recruiting", action="store_true", help="List all recruiting trials"
    )
    parser.add_argument(
        "--phase",
        help="List all trials for a specific phase (accepts both Arabic (1-4) and Roman numerals (I-IV))",
    )

    args = parser.parse_args()

    # Load and parse the JSON file
    json_data = load_json_file(args.json_file)
    trials_parser = ClinicalTrialsParser(json_data)

    # Process based on arguments
    trials = trials_parser.trials

    # Apply filters based on arguments
    if args.recruiting:
        trials = trials_parser.get_recruiting_trials()
        logger.info(f"main: Filtering for recruiting trials: {len(trials)} found")

    if args.phase:
        # Convert Arabic numerals to Roman numerals if needed
        phase_map = {"1": "I", "2": "II", "3": "III", "4": "IV"}
        phase = phase_map.get(args.phase, args.phase.upper())

        phase_filtered = trials_parser.get_trials_by_phase(phase)
        trials = [t for t in trials if t in phase_filtered]
        logger.info(f"main: Filtering for phase {phase} trials: {len(trials)} found")

    # Display results
    if args.nct_id:
        trial = trials_parser.get_trial_by_nct_id(args.nct_id)
        if trial:
            logger.info("main: " + json.dumps(trial.to_dict(), indent=2))
        else:
            logger.info(f"main: No trial found with NCT ID: {args.nct_id}")
    elif args.recruiting or args.phase:
        logger.info("main: Results:")
        for trial in trials:
            logger.info(
                f"- {trial.identification.nct_id}: {trial.identification.brief_title}"
            )
    else:
        logger.info(f"main: Total number of trials: {len(trials_parser.trials)}")
        logger.info("main: Use --help to see available options")


if __name__ == "__main__":
    main()
