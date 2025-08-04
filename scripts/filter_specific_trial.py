#!/usr/bin/env python3
"""
Script to filter a specific clinical trial by NCT ID against a clinical record.

This script evaluates whether a specific clinical trial (identified by NCT ID) 
is suitable for a patient based on their clinical record. It provides detailed
logging of the filtering process and results.
"""

import argparse
import datetime
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from base.clinical_trial import ClinicalTrialsParser
from base.disease_expert import extract_conditions_from_record
from base.gpt_client import GPTClient
from base.trial_expert import GPTTrialFilter

# Global logger for this module
logger = None


def setup_logging(nct_id: str) -> str:
    """Setup logging with timestamp and NCT ID in filename."""
    global logger
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    log_file = logs_dir / f"filter_trial_{nct_id}_{timestamp}.log"

    # Create a dedicated logger for this script
    logger = logging.getLogger("filter_specific_trial")
    logger.handlers.clear()  # Clear any existing handlers
    logger.setLevel(logging.INFO)
    logger.propagate = False  # Don't propagate to root logger

    # Create single formatter to use for both handlers
    formatter = logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    # Create and configure file handler
    file_handler = logging.FileHandler(str(log_file))
    file_handler.setFormatter(formatter)
    
    # Create and configure console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Add handlers to our dedicated logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Set specific loggers to WARNING level to reduce noise
    logging.getLogger("base.gpt_client").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("base.prompt_cache").setLevel(logging.INFO)

    return str(log_file)


def load_trials(trials_file: str) -> List[Dict[str, Any]]:
    """Load trials from a JSON file."""
    with open(trials_file, "r") as f:
        return json.load(f)


def fetch_trial_data(nct_id: str) -> List[Dict[str, Any]]:
    """Fetch clinical trial data directly from ClinicalTrials.gov."""
    import tempfile
    
    # Create a temporary file to store the spider output
    with tempfile.NamedTemporaryFile(
        mode="w+", suffix=".json", delete=False
    ) as tmp_file:
        temp_output = tmp_file.name

    try:
        # Import scrapy components here to avoid circular imports
        from scrapy.crawler import CrawlerProcess
        from scrapy.utils.project import get_project_settings
        from clinical_trial_crawler.clinical_trial_crawler.spiders.clinical_trials_spider import (
            ClinicalTrialsSpider,
        )

        # Configure and run the spider with minimal settings
        settings = get_project_settings()
        settings.set("DOWNLOAD_DELAY", 1)
        settings.set(
            "USER_AGENT",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
        )

        process = CrawlerProcess(settings)
        process.crawl(
            ClinicalTrialsSpider, specific_trial=nct_id, output_file=temp_output
        )
        process.start()

        # Read the results
        with open(temp_output, "r") as f:
            content = f.read().strip()
            if content:
                return json.loads(content)
            else:
                return []

    except Exception as e:
        if logger:
            logger.error(f"Failed to fetch trial data: {e}")
        return []
    finally:
        try:
            os.unlink(temp_output)
        except Exception as e:
            if logger:
                logger.warning(f"Failed to delete temporary file {temp_output}: {e}")


def filter_specific_trial(
    clinical_record: str,
    nct_id: str,
    trials_file: Optional[str] = None,
    api_key: Optional[str] = None,
    cache_size: int = 100000,
    refresh_cache: bool = False,
) -> Dict[str, Any]:
    """
    Filter a specific clinical trial by NCT ID against a clinical record.
    
    Args:
        clinical_record: Path to the clinical record file
        nct_id: NCT ID of the trial to filter
        trials_file: Optional path to JSON file containing trial data
        api_key: OpenAI API key
        cache_size: Size of the GPT response cache
        refresh_cache: Whether to refresh the cache
        
    Returns:
        Dictionary containing the filtering results
    """
    
    if logger is None:
        raise RuntimeError("Logger not initialized. Call setup_logging() first.")
    
    logger.info("=" * 60)
    logger.info(f"filter_specific_trial: Starting filtering process for trial {nct_id}")
    logger.info("=" * 60)
    
    # Initialize GPT client and filter
    logger.info("filter_specific_trial: Initializing GPT client and filter")
    if not api_key:
        raise ValueError("API key is required")
    gpt_client = GPTClient(api_key=api_key)
    gpt_filter = GPTTrialFilter(api_key=api_key, cache_size=cache_size)
    
    # Extract conditions from clinical record
    logger.info(f"filter_specific_trial: Reading clinical record from {clinical_record}")
    logger.info("filter_specific_trial: Extracting conditions from clinical record")
    history_items = extract_conditions_from_record(
        clinical_record, gpt_client, refresh_cache=refresh_cache
    )
    logger.info(f"filter_specific_trial: Extracted {len(history_items)} conditions:")
    for item in history_items:
        logger.info(f"filter_specific_trial:  - {item}")
    
    # Load or fetch trial data
    if trials_file and os.path.exists(trials_file):
        logger.info(f"filter_specific_trial: Loading trial from file: {trials_file}")
        trials_data = load_trials(trials_file)
        trials_parser = ClinicalTrialsParser(trials_data)
        trial = trials_parser.get_trial_by_nct_id(nct_id)
        if not trial:
            raise ValueError(f"Trial {nct_id} not found in {trials_file}")
    else:
        logger.info(f"filter_specific_trial: Fetching trial {nct_id} from ClinicalTrials.gov")
        trials_data = fetch_trial_data(nct_id)
        if not trials_data:
            raise ValueError(f"Failed to fetch trial {nct_id} from ClinicalTrials.gov")
        
        trials_parser = ClinicalTrialsParser(trials_data)
        trial = trials_parser.get_trial_by_nct_id(nct_id)
        if not trial:
            raise ValueError(f"Trial {nct_id} not found in fetched data")
    
    logger.info(f"filter_specific_trial: Trial found: {trial.identification.brief_title}")
    logger.info(f"filter_specific_trial: Study type: {trial.design.study_type}")
    logger.info(f"filter_specific_trial: Phases: {trial.design.phases}")
    logger.info(f"filter_specific_trial: Recruiting: {trial.is_recruiting}")
    
    # Evaluate the trial against the conditions
    logger.info("=" * 40)
    logger.info("filter_specific_trial: Evaluating trial eligibility")
    logger.info("=" * 40)
    
    is_eligible, cost, failure_reason = gpt_filter.evaluate_trial(
        trial, history_items, refresh_cache
    )
    
    # Prepare results
    result = {
        "nct_id": nct_id,
        "brief_title": trial.identification.brief_title,
        "official_title": trial.identification.official_title,
        "study_type": trial.design.study_type,
        "phases": trial.design.phases,
        "is_recruiting": trial.is_recruiting,
        "conditions": trial.description.conditions,
        "eligibility_criteria": trial.eligibility.criteria,
        "clinical_record_conditions": history_items,
        "is_eligible": is_eligible,
        "api_cost": cost,
        "failure_reason": {
            "type": failure_reason.type,
            "message": failure_reason.message,
            "failed_condition": failure_reason.failed_condition,
            "failed_criterion": failure_reason.failed_criterion,
            "failure_details": failure_reason.failure_details
        } if not is_eligible and failure_reason is not None else None,
        "evaluation_timestamp": datetime.datetime.now().isoformat(),
    }
    
    # Log results
    logger.info("=" * 60)
    logger.info("filter_specific_trial: FILTERING RESULTS")
    logger.info("=" * 60)
    logger.info(f"filter_specific_trial: NCT ID: {nct_id}")
    logger.info(f"filter_specific_trial: Trial Title: {trial.identification.brief_title}")
    logger.info(f"filter_specific_trial: Eligibility: {'ELIGIBLE' if is_eligible else 'NOT ELIGIBLE'}")
    logger.info(f"filter_specific_trial: API Cost: ${cost:.4f}")
    
    if is_eligible:
        logger.info("filter_specific_trial: ✅ Trial is suitable for the patient based on clinical record")
    else:
        logger.info(f"filter_specific_trial: ❌ Trial is not suitable: {failure_reason}")
    
    logger.info("=" * 60)
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Filter a specific clinical trial by NCT ID against a clinical record"
    )
    parser.add_argument("clinical_record", help="Path to the clinical record file")
    parser.add_argument("nct_id", help="NCT ID of the trial to filter")
    parser.add_argument(
        "--trials-file",
        help="Path to the trials JSON file (optional, will fetch from ClinicalTrials.gov if not provided)",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file path for filtering results",
        default=None,
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
        "--refresh-cache",
        action="store_true",
        help="Refresh the cache of GPT responses",
    )

    args = parser.parse_args()

    # Setup logging
    log_file = setup_logging(args.nct_id)

    if logger is None:
        raise RuntimeError("Logger not initialized. Call setup_logging() first.")

    logger.info("main: Starting specific trial filtering process")
    logger.info(f"main: Input clinical record: {args.clinical_record}")
    logger.info(f"main: Target NCT ID: {args.nct_id}")
    if args.trials_file:
        logger.info(f"main: Input trials file: {args.trials_file}")
    else:
        logger.info("main: Will fetch trial data from ClinicalTrials.gov")

    # Get API key from command line or environment
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("main: OpenAI API key not found")
        raise ValueError(
            "OpenAI API key not found. Please provide it via --api-key argument or OPENAI_API_KEY environment variable"
        )

    try:
        # Perform the filtering
        result = filter_specific_trial(
            clinical_record=args.clinical_record,
            nct_id=args.nct_id,
            trials_file=args.trials_file,
            api_key=api_key,
            cache_size=args.cache_size,
            refresh_cache=args.refresh_cache,
        )

        # Save results if output file specified
        if args.output:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2)
            logger.info(f"main: Results saved to: {args.output}")
        else:
            # Print results to console
            logger.info("\n" + "=" * 60)
            logger.info("main: FILTERING RESULTS")
            logger.info("=" * 60)
            logger.info(json.dumps(result, indent=2))
            logger.info("=" * 60)

        logger.info(f"main: Log file: {log_file}")
        logger.info("main: Filtering process completed successfully")

    except Exception as e:
        logger.error(f"main: An error occurred: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
