#!/usr/bin/env python3
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

# Global logger instance
logger = None


def setup_logging():
    """Set up logging configuration with timestamp in filename."""
    global logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"trial_ranking_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)
    return logger


def read_trials_file(file_path):
    """Read and parse the trials JSON file."""
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON file: {e}")
        raise
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise


def log_trial_details(trial):
    """Log relevant details for each trial."""
    logger.info("=" * 80)
    identification = trial.get("identification", {})
    logger.info(f"Trial ID: {identification.get('nct_id', 'N/A')}")
    logger.info(f"Title: {identification.get('brief_title', 'N/A')}")
    logger.info(f"Drug Analysis: {trial.get('drug_analysis', {})}")
    logger.info(f"Recommendation Level: {trial.get('recommendation_level', 'N/A')}")
    logger.info(f"Analysis Reason: {trial.get('reason', 'N/A')}")
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Rank clinical trials based on various criteria."
    )
    parser.add_argument(
        "input_file", help="Path to the input JSON file containing analyzed trials"
    )
    args = parser.parse_args()

    setup_logging()
    logger.info(f"Starting trial ranking process for file: {args.input_file}")

    try:
        trials = read_trials_file(args.input_file)
        logger.info(f"Successfully loaded {len(trials)} trials from {args.input_file}")

        for trial in trials:
            log_trial_details(trial)

        # Log summary after processing all trials
        logger.info("\n" + "=" * 80)
        logger.info(f"TRIAL SUMMARY")
        logger.info(f"Total number of trials processed: {len(trials)}")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise


if __name__ == "__main__":
    main()
