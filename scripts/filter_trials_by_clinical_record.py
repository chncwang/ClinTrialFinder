import argparse
import datetime
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List
import logging

logger = logging.getLogger(__name__)

from base.clinical_trial import ClinicalTrialsParser
from base.disease_expert import extract_conditions_from_record
from base.gpt_client import GPTClient
from base.trial_expert import GPTTrialFilter, process_trials_with_conditions

# Configure standard logging with timestamp in filename
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# Create logs directory if it doesn't exist
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)
log_file = logs_dir / f"filter_trials_by_clinical_record_{timestamp}.log"

# Configure standard logging with both file and console output
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ],
    force=True
)

# Set console handler to INFO level
for handler in logging.getLogger().handlers:
    if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
        handler.setLevel(logging.INFO)


def load_trials(trials_file: str) -> List[Dict[str, Any]]:
    """Load trials from a JSON file."""
    with open(trials_file, "r") as f:
        return json.load(f)


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
        default=f"filtered_trials_{timestamp}.json",
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
    parser.add_argument(
        "--max-trials",
        type=int,
        default=None,
        help="Maximum number of trials to process. Default is no limit",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Refresh the cache of GPT responses",
    )

    args = parser.parse_args()

    logger.info("Starting trial filtering process")
    logger.info(f"Input clinical record: {args.clinical_record}")
    logger.info(f"Input trials file: {args.trials_file}")
    logger.info(f"Output file: {args.output}")

    # Get API key from command line or environment
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OpenAI API key not found")
        raise ValueError(
            "OpenAI API key not found. Please provide it via --api-key argument or OPENAI_API_KEY environment variable"
        )

    try:
        # Initialize GPT client and filter
        logger.info("Initializing GPT client and filter")
        gpt_client = GPTClient(api_key=api_key)
        gpt_filter = GPTTrialFilter(api_key=api_key, cache_size=args.cache_size)

        # Extract conditions from clinical record
        logger.info(f"Reading clinical record from {args.clinical_record}")
        logger.info("Extracting conditions from clinical record")
        history_items = extract_conditions_from_record(
            args.clinical_record, gpt_client, refresh_cache=args.refresh_cache
        )
        logger.info(f"Extracted {len(history_items)} conditions:")
        for item in history_items:
            logger.info(f"  - {item}")

        # Load trials
        logger.info(f"Loading trials from {args.trials_file}")
        trials = load_trials(args.trials_file)
        if args.max_trials:
            trials = trials[: args.max_trials]
            logger.info(f"Limited to processing {args.max_trials} trials")
        trials_parser = ClinicalTrialsParser(trials)
        logger.info(f"Loaded {len(trials)} trials")

        # Process trials with conditions
        logger.info("Starting trial filtering process")
        total_cost, eligible_count = process_trials_with_conditions(
            trials_parser.trials,
            history_items,
            args.output,
            gpt_filter,
            refresh_cache=args.refresh_cache,
        )

        # Log final results
        logger.info("=" * 50)
        logger.info("Filtering process completed")
        logger.info(f"Total trials processed: {len(trials)}")
        logger.info(f"Eligible trials found: {eligible_count}")
        logger.info(f"Total API cost: ${total_cost:.2f}")
        logger.info(f"Results saved to: {args.output}")
        logger.info(
            f"Excluded trials saved to: {args.output.replace('.json', '_excluded.json')}"
        )
        logger.info(f"Log file: {log_file}")
        logger.info("=" * 50)

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
