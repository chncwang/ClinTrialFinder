#!/usr/bin/env python3
import argparse
import json
import logging
import random
from datetime import datetime
from typing import List, Any

from base.clinical_trial import ClinicalTrial
from base.gpt_client import GPTClient
from base.trial_expert import compare_trials
from base.disease_expert import extract_disease_from_record

# Global logger instance
logger = None


def get_logger() -> logging.Logger:
    """Get the logger instance, initializing it if necessary."""
    global logger
    if logger is None:
        setup_logging()
    assert logger is not None  # Type checker hint
    return logger


def setup_logging():
    """Set up logging configuration with timestamp in filename."""
    global logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"trial_ranking_{timestamp}.log"

    # Create formatters for file and console
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_formatter = logging.Formatter("%(levelname)s: %(message)s")

    # Set up file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(file_formatter)

    # Set up console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)

    # Configure root logger to capture all logs
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers = []  # Clear existing handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # Get logger for this module
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.propagate = True  # Ensure logs propagate to root logger

    return logger


def read_trials_file(file_path: str):
    """Read and parse the trials JSON file."""
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        get_logger().error(f"Error parsing JSON file: {e}")
        raise
    except FileNotFoundError:
        get_logger().error(f"File not found: {file_path}")
        raise


def log_trial_details(trial: ClinicalTrial):
    """Log relevant details for each trial."""
    get_logger().info("=" * 80)
    get_logger().info(f"log_trial_details: Trial ID: {trial.identification.nct_id}")
    get_logger().info(f"log_trial_details: Title: {trial.identification.brief_title}")
    get_logger().info(f"log_trial_details: Drug Analysis: {trial.drug_analysis}")
    get_logger().info(f"log_trial_details: Recommendation Level: {trial.recommendation_level}")
    get_logger().info(f"log_trial_details: Analysis Reason: {trial.analysis_reason}")
    get_logger().info("=" * 80)


def partition(
    trials: List[ClinicalTrial],
    clinical_record: str,
    disease: str,
    low: int,
    high: int,
    gpt_client: GPTClient,
) -> tuple[int, float]:
    """
    Partition trials using the last element as pivot.
    Returns the position of the pivot after partitioning and the total cost.
    Args:
        trials: List of trials to rank
        clinical_record: Patient's clinical record
        disease: Patient's disease
        low: Low index
        high: High index
        gpt_client: GPT client for comparisons
    """
    pivot = trials[high]
    i = low - 1
    total_cost = 0.0

    for j in range(low, high):
        # Compare current trial with pivot
        better_trial, _, cost = compare_trials(
            clinical_record,
            disease,
            trials[j],
            pivot,
            gpt_client,
        )
        total_cost += cost

        # Get brief titles for logging
        trial1_title = trials[j].identification.brief_title
        trial2_title = pivot.identification.brief_title
        better_title = better_trial.identification.brief_title

        get_logger().info(
            f"partition: Comparison: '{trial1_title}' vs '{trial2_title}'"
            f"\npartition: Better trial: '{better_title}'"
            f"\npartition: Cost: ${cost:.6f}, Total: ${total_cost:.6f}"
        )

        # If current trial is better than pivot
        if better_trial.identification.nct_id == trials[j].identification.nct_id:
            i += 1
            trials[i], trials[j] = trials[j], trials[i]

    # Place pivot in correct position
    trials[i + 1], trials[high] = trials[high], trials[i + 1]
    return i + 1, total_cost


def quicksort(
    trials: List[ClinicalTrial],
    clinical_record: str,
    disease: str,
    low: int,
    high: int,
    gpt_client: GPTClient,
) -> float:
    """
    Sort trials using quicksort algorithm.
    Returns the total cost of API calls.
    """
    total_cost = 0.0
    if low < high:
        # Find pivot position
        pivot_pos, partition_cost = partition(
            trials, clinical_record, disease, low, high, gpt_client
        )
        total_cost += partition_cost

        # Recursively sort sub-arrays
        total_cost += quicksort(trials, clinical_record, disease, low, pivot_pos - 1, gpt_client)
        total_cost += quicksort(
            trials, clinical_record, disease, pivot_pos + 1, high, gpt_client
        )

    return total_cost


def rank_trials(
    trials: List[Any],
    clinical_record: str,
    disease: str,
    gpt_client: GPTClient,
    seed: int = 42,  # Default seed for reproducibility
) -> tuple[List[ClinicalTrial], float]:
    """
    Rank trials using quicksort with compare_trials for comparison.
    Returns sorted list of trials from best to worst and the total cost.

    Args:
        trials: List of trials to rank
        clinical_record: Patient's clinical record
        gpt_client: GPT client for comparisons
        seed: Random seed for deterministic shuffling (default: 42)
    """
    if not trials:
        return [], 0.0

    # Convert trial dictionaries to ClinicalTrial objects if needed
    trial_objects = [
        trial if isinstance(trial, ClinicalTrial) else ClinicalTrial(trial)
        for trial in trials
    ]

    # Log first 10 trial titles before shuffle
    get_logger().info("rank_trials: First 10 trial titles before shuffle:")
    for i in range(min(10, len(trial_objects))):
        get_logger().info(f"rank_trials: {i+1}. {trial_objects[i].identification.brief_title}")

    # Randomize trials before sorting
    random.seed(seed)
    get_logger().info(f"rank_trials: Randomized {len(trial_objects)} trials before sorting (seed: {seed})")
    get_logger().info(
        f"rank_trials: Initial pivotal trial title (last position): {trial_objects[-1].identification.brief_title}"
    )
    random.shuffle(trial_objects)
    get_logger().info(
        f"rank_trials: Pivotal trial title after randomization (last position): {trial_objects[-1].identification.brief_title}"
    )

    # Log first 10 trial titles after shuffle
    get_logger().info("rank_trials: First 10 trial titles after shuffle:")
    for i in range(min(10, len(trial_objects))):
        get_logger().info(f"rank_trials: {i+1}. {trial_objects[i].identification.brief_title}")

    # Sort trials using quicksort
    total_cost = quicksort(
        trial_objects,
        clinical_record,
        disease,
        0,
        len(trial_objects) - 1,
        gpt_client,
    )
    get_logger().info(f"rank_trials: Total cost of ranking: ${total_cost:.6f}")

    return trial_objects, total_cost


def main():
    parser = argparse.ArgumentParser(
        description="Rank clinical trials based on various criteria."
    )
    parser.add_argument(
        "input_file", help="Path to the input JSON file containing analyzed trials"
    )
    parser.add_argument(
        "clinical_record_file", help="Path to the file containing the clinical record"
    )
    parser.add_argument(
        "--openai-api-key",
        help="OpenAI API key",
        default=None,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic shuffling (default: 42)",
    )
    parser.add_argument(
        "--csv-output",
        action="store_true",
        help="Also output results in CSV format",
    )

    args = parser.parse_args()

    setup_logging()
    get_logger().info(f"main: Starting trial ranking process for file: {args.input_file}")

    try:
        # Read trials and clinical record
        trials = read_trials_file(args.input_file)
        with open(args.clinical_record_file, "r") as f:
            clinical_record = f.read().strip()

        get_logger().info(f"main: Successfully loaded {len(trials)} trials from {args.input_file}")
        get_logger().info(
            f"main: Successfully loaded clinical record from {args.clinical_record_file}"
        )

        # Initialize GPT client
        gpt_client = GPTClient(api_key=args.openai_api_key)

        # Extract disease from clinical record
        disease, _ = extract_disease_from_record(clinical_record, gpt_client)
        if disease is None:
            disease = "Unknown disease"
        get_logger().info(f"main: Disease: {disease}")

        # Rank trials with specified seed
        ranked_trials, total_cost = rank_trials(
            trials, clinical_record, disease, gpt_client, seed=args.seed
        )

        # Log ranked trials
        get_logger().info("\nmain: Ranked Trials (from best to worst):")
        for i, trial in enumerate(ranked_trials, 1):
            get_logger().info(f"\nmain: Rank {i}:")
            log_trial_details(trial)

        # Save ranked trials to file
        output_file = f"ranked_trials_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, "w") as f:
            json.dump([trial.to_dict() for trial in ranked_trials], f, indent=2)
        get_logger().info(f"\nmain: Ranked trials saved to: {output_file}")

        # Generate CSV output if requested
        csv_output_file = None
        if args.csv_output:
            try:
                from scripts.json_to_csv_converter import convert_json_to_csv
                csv_output_file = convert_json_to_csv(output_file)
                get_logger().info(f"main: CSV output saved to: {csv_output_file}")
            except ImportError:
                get_logger().warning("main: CSV conversion module not available. Skipping CSV output.")
            except Exception as e:
                get_logger().error(f"main: Error generating CSV output: {e}")

        # Log summary
        get_logger().info("\n" + "=" * 80)
        get_logger().info(f"main: TRIAL RANKING SUMMARY")
        get_logger().info(f"main: Total number of trials ranked: {len(ranked_trials)}")
        get_logger().info(f"main: Total cost of ranking: ${total_cost:.6f}")
        get_logger().info(f"main: Output file: {output_file}")
        if args.csv_output and csv_output_file:
            get_logger().info(f"main: CSV output: {csv_output_file}")
        get_logger().info("main: " + "=" * 80)

    except Exception as e:
        get_logger().error(f"main: An error occurred: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
