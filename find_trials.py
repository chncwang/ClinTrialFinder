#!/usr/bin/env python3
"""
One-click pipeline: Clinical record to recommended trials

This script provides a streamlined command-line interface for finding and ranking
clinical trials based on a patient's clinical record. It mirrors the functionality
of the ClinTrialFinder mail service.

Usage:
    python find_trials.py --clinical-record patient_record.txt --output results.csv

Pipeline stages:
1. Extract disease and conditions from clinical record
2. Download relevant trials from ClinicalTrials.gov
3. Filter trials by eligibility criteria
4. Analyze trials with drug investigation
5. Rank trials by recommendation level
6. Output results in specified format
"""

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from base.clinical_trial import ClinicalTrial, ClinicalTrialsParser
from base.disease_expert import extract_conditions_from_content, extract_disease_from_record
from base.gpt_client import GPTClient
from base.perplexity import PerplexityClient
from base.trial_expert import GPTTrialFilter, process_trials_with_conditions
from base.utils import trials_to_csv
from scripts.analyze_filtered_trials import process_trials_file
from scripts.download_trials import download_trials
from scripts.rank_trials import rank_trials

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def read_clinical_record(clinical_record_input: str) -> str:
    """
    Read clinical record from file or use as inline text.

    Args:
        clinical_record_input: Path to file or inline text

    Returns:
        Clinical record content
    """
    # Check if it's a file path
    if os.path.isfile(clinical_record_input):
        logger.info(f"Reading clinical record from file: {clinical_record_input}")
        with open(clinical_record_input, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        logger.info("Using inline clinical record text")
        return clinical_record_input


def download_active_trials_for_disease(disease_name: str, output_file: str) -> bool:
    """
    Download active clinical trials for a given disease.

    Args:
        disease_name: The disease or condition to search for
        output_file: Path to save the downloaded trials

    Returns:
        True if download succeeded, False otherwise
    """
    if not disease_name or disease_name in ("Invalid", "Error", "Unknown"):
        logger.error(f"Invalid disease name: {disease_name}")
        return False

    try:
        logger.info(f"Downloading active trials for disease: {disease_name}")

        # Create args object for download_trials function
        args = argparse.Namespace()
        args.condition = disease_name
        args.exclude_completed = True
        args.output_file = output_file
        args.specific_trial = None
        args.log_level = "INFO"
        args.include_broader = False
        args.openai_api_key = None
        args.location_facility = None
        args.location_state = None
        args.location_country = None
        args.location_city = None

        # Call download_trials function
        success, result_file = download_trials(args)

        if success and result_file and os.path.exists(result_file) and os.path.getsize(result_file) > 0:
            logger.info(f"Successfully downloaded trials to {result_file}")
            return True
        else:
            logger.error(f"Failed to download trials for {disease_name}")
            return False

    except Exception as e:
        logger.error(f"Error downloading trials: {str(e)}")
        return False


def filter_analyze_and_rank_trials(
    trials_file: str,
    conditions: List[str],
    clinical_record: str,
    gpt_client: GPTClient,
    perplexity_client: PerplexityClient,
    output_file: str,
    max_results: Optional[int] = None
) -> tuple[int, float]:
    """
    Filter, analyze, and rank clinical trials based on conditions.

    Args:
        trials_file: Path to JSON file containing trials
        conditions: List of clinical conditions
        clinical_record: Full clinical record text
        gpt_client: Initialized GPT client
        perplexity_client: Initialized Perplexity client
        output_file: Path to save results
        max_results: Maximum number of results to return (None = all)

    Returns:
        Tuple of (number of recommended trials, total cost)
    """
    try:
        # Create temporary directory for intermediate files
        temp_dir = tempfile.mkdtemp()
        filtered_file = os.path.join(temp_dir, "filtered_trials.json")
        analyzed_file = os.path.join(temp_dir, "analyzed_trials.json")

        # Load trials from file
        with open(trials_file, "r") as f:
            trials_data = json.load(f)

        trials_parser = ClinicalTrialsParser(trials_data)
        trials = trials_parser.trials

        logger.info(f"Loaded {len(trials)} trials")

        # Create GPT filter instance
        gpt_filter = GPTTrialFilter(gpt_client)

        # Stage 1: Filter trials by eligibility criteria
        logger.info("Stage 1/4: Filtering trials by eligibility criteria...")
        filter_cost, eligible_count = process_trials_with_conditions(
            trials=trials,
            conditions=conditions,
            output_path=filtered_file,
            gpt_filter=gpt_filter,
        )
        logger.info(f"Found {eligible_count} eligible trials (cost: ${filter_cost:.2f})")

        if not os.path.exists(filtered_file) or os.path.getsize(filtered_file) == 0:
            logger.warning("No eligible trials found")
            return 0, filter_cost

        # Stage 2: Analyze eligible trials (drug investigation)
        logger.info("Stage 2/4: Analyzing trials with drug investigation...")
        process_trials_file(
            filtered_file,
            gpt_client,
            perplexity_client,
            clinical_record,
            analyzed_file
        )

        # Load analyzed trials
        with open(analyzed_file, "r") as f:
            analyzed_trials = json.load(f)

        # Stage 3: Filter for recommended trials only
        logger.info("Stage 3/4: Filtering for recommended trials...")
        recommended_trials = [
            trial for trial in analyzed_trials
            if trial.get("recommendation_level") in [
                "RecommendationLevel.RECOMMENDED",
                "RecommendationLevel.STRONGLY_RECOMMENDED"
            ]
        ]

        logger.info(f"Found {len(recommended_trials)} recommended trials out of {len(analyzed_trials)} analyzed")

        if not recommended_trials:
            logger.warning("No recommended trials found")
            return 0, filter_cost

        # Convert to ClinicalTrial objects
        trial_objects = [ClinicalTrial(trial) for trial in recommended_trials]

        # Extract disease for ranking
        disease, _ = extract_disease_from_record(clinical_record, gpt_client)
        logger.info(f"Extracted disease for ranking: {disease}")

        # Stage 4: Rank trials
        logger.info("Stage 4/4: Ranking trials...")
        ranked_trials, ranking_cost = rank_trials(
            trial_objects,
            clinical_record,
            disease or "Unknown disease",
            gpt_client
        )

        total_cost = filter_cost + ranking_cost
        logger.info(f"Ranking completed (cost: ${ranking_cost:.2f})")
        logger.info(f"Total cost: ${total_cost:.2f}")

        # Apply max_results limit if specified
        if max_results:
            ranked_trials = ranked_trials[:max_results]
            logger.info(f"Limited results to top {max_results} trials")

        # Convert to dictionaries for serialization
        ranked_trials_dict = [trial.to_dict() for trial in ranked_trials]

        # Determine output format from file extension
        output_ext = os.path.splitext(output_file)[1].lower()

        if output_ext == '.json':
            # Save as JSON
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(ranked_trials_dict, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(ranked_trials)} trials to {output_file} (JSON)")
        elif output_ext == '.csv':
            # Save as CSV
            trials_to_csv(ranked_trials_dict, output_file)
            logger.info(f"Saved {len(ranked_trials)} trials to {output_file} (CSV)")
        else:
            # Default to JSON
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(ranked_trials_dict, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(ranked_trials)} trials to {output_file} (JSON)")

        return len(ranked_trials), total_cost

    except Exception as e:
        logger.error(f"Error processing trials: {str(e)}")
        raise


def main():
    """Main entry point for the pipeline."""
    parser = argparse.ArgumentParser(
        description="Find and rank clinical trials based on patient clinical record",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with file input
  python find_trials.py --clinical-record patient.txt --output results.csv

  # With inline text
  python find_trials.py --clinical-record "Patient with stage 3 lung cancer..." --output results.json

  # Limit results
  python find_trials.py --clinical-record patient.txt --output results.csv --max-results 10

  # Verbose logging
  python find_trials.py --clinical-record patient.txt --output results.csv --verbose
        """
    )

    # Required arguments
    parser.add_argument(
        '--clinical-record',
        required=True,
        help='Path to clinical record file or inline clinical record text'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Output file path (.json or .csv)'
    )

    # Optional arguments
    parser.add_argument(
        '--max-results',
        type=int,
        help='Maximum number of results to return (default: all recommended trials)'
    )
    parser.add_argument(
        '--openai-api-key',
        help='OpenAI API key (default: from OPENAI_API_KEY environment variable)'
    )
    parser.add_argument(
        '--perplexity-api-key',
        help='Perplexity API key (default: from PERPLEXITY_API_KEY environment variable)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")

    # Get API keys
    openai_api_key = args.openai_api_key or os.environ.get('OPENAI_API_KEY')
    perplexity_api_key = args.perplexity_api_key or os.environ.get('PERPLEXITY_API_KEY')

    if not openai_api_key:
        logger.error("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or use --openai-api-key")
        sys.exit(1)

    if not perplexity_api_key:
        logger.error("Perplexity API key not provided. Set PERPLEXITY_API_KEY environment variable or use --perplexity-api-key")
        sys.exit(1)

    # Initialize clients
    logger.info("Initializing API clients...")
    gpt_client = GPTClient(api_key=openai_api_key)
    perplexity_client = PerplexityClient(api_key=perplexity_api_key)

    # Start timer
    start_time = time.time()

    try:
        # Read clinical record
        clinical_record = read_clinical_record(args.clinical_record)
        logger.info(f"Clinical record length: {len(clinical_record)} characters")

        # Extract disease and conditions
        logger.info("Extracting disease and conditions from clinical record...")
        disease, _ = extract_disease_from_record(clinical_record, gpt_client)
        conditions = extract_conditions_from_content(clinical_record, gpt_client)

        logger.info(f"Identified disease: {disease}")
        logger.info(f"Identified conditions: {conditions}")

        if not disease or disease in ("Invalid", "Error", "Unknown"):
            logger.error("Unable to identify disease from clinical record")
            sys.exit(1)

        # Download trials
        temp_dir = tempfile.mkdtemp()
        trials_file = os.path.join(temp_dir, "downloaded_trials.json")

        logger.info("Downloading trials from ClinicalTrials.gov...")
        success = download_active_trials_for_disease(disease, trials_file)

        if not success:
            logger.error("Failed to download trials")
            sys.exit(1)

        # Filter, analyze, and rank trials
        num_results, total_cost = filter_analyze_and_rank_trials(
            trials_file,
            conditions,
            clinical_record,
            gpt_client,
            perplexity_client,
            args.output,
            args.max_results
        )

        # Calculate elapsed time
        elapsed_time = time.time() - start_time

        # Print summary
        print("\n" + "="*60)
        print("PIPELINE COMPLETE")
        print("="*60)
        print(f"Disease identified: {disease}")
        print(f"Conditions: {len(conditions)}")
        print(f"Recommended trials: {num_results}")
        print(f"Output file: {args.output}")
        print(f"Total API cost: ${total_cost:.2f}")
        print(f"Processing time: {elapsed_time:.1f} seconds")
        print("="*60)

        if num_results == 0:
            print("\nNo recommended trials found for the given clinical record.")
            sys.exit(0)

        logger.info("Pipeline completed successfully")

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
