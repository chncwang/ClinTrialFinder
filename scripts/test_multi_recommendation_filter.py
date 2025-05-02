#!/usr/bin/env python3
import logging
import subprocess
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def run_test(input_file):
    """Run tests for multi-recommendation filtering"""

    logger.info("Testing single recommendation level filter (Recommended)")
    cmd_single = [
        "python",
        "scripts/filter_trials.py",
        input_file,
        "--recommendation-level",
        "Recommended",
        "-o",
        "single_recommended_test.json",
    ]
    subprocess.run(cmd_single, check=True)

    logger.info("Testing single recommendation level filter (Strongly Recommended)")
    cmd_single_strongly = [
        "python",
        "scripts/filter_trials.py",
        input_file,
        "--recommendation-level",
        "Strongly Recommended",
        "-o",
        "single_strongly_recommended_test.json",
    ]
    subprocess.run(cmd_single_strongly, check=True)

    logger.info(
        "Testing multiple recommendation levels filter (Recommended + Strongly Recommended)"
    )
    cmd_multiple = [
        "python",
        "scripts/filter_trials.py",
        input_file,
        "--recommendation-level",
        "Recommended",
        "Strongly Recommended",
        "-o",
        "multi_recommended_test.json",
    ]
    subprocess.run(cmd_multiple, check=True)

    logger.info("All tests completed successfully!")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        logger.error(
            "Usage: python test_multi_recommendation_filter.py <input_json_file>"
        )
        sys.exit(1)

    input_file = sys.argv[1]
    if not Path(input_file).exists():
        logger.error(f"Input file {input_file} does not exist")
        sys.exit(1)

    run_test(input_file)
