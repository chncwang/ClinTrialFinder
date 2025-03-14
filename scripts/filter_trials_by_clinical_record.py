import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from base.disease_expert import extract_conditions_from_record
from base.trial_analyzer import TrialAnalyzer

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
    trials: List[Dict[str, Any]], conditions: List[str]
) -> List[Dict[str, Any]]:
    """Filter trials based on extracted conditions."""
    filtered_trials = []
    for trial in trials:
        # Check if trial's conditions match any of the extracted conditions
        trial_conditions = trial.get("conditions", [])
        if any(
            condition.lower() in trial_condition.lower()
            for condition in conditions
            for trial_condition in trial_conditions
        ):
            filtered_trials.append(trial)
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

    args = parser.parse_args()

    # Extract conditions from clinical record
    logger.info(f"Reading clinical record from {args.clinical_record}")
    conditions = extract_conditions_from_record(args.clinical_record)
    logger.info(f"Extracted {len(conditions)} conditions from clinical record")

    # Load trials
    logger.info(f"Loading trials from {args.trials_file}")
    trials = load_trials(args.trials_file)
    logger.info(f"Loaded {len(trials)} trials")

    # Filter trials
    logger.info("Filtering trials based on conditions")
    filtered_trials = filter_trials_by_conditions(trials, conditions)
    logger.info(f"Found {len(filtered_trials)} matching trials")

    # Save filtered trials
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(filtered_trials, f, indent=2)
    logger.info(f"Saved filtered trials to {output_path}")


if __name__ == "__main__":
    main()
