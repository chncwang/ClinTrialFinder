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
import os
from typing import Any, Dict, List, Optional

import logging

logger = logging.getLogger(__name__)

from base.clinical_trial import ClinicalTrialsParser
from base.disease_expert import extract_conditions_from_record
from base.gpt_client import GPTClient
from base.trial_expert import GPTTrialFilter
from base.logging_config import setup_logging
from base.utils import load_json_list_file, get_api_key, create_gpt_client


def setup_logging_for_trial(nct_id: str, log_level: str = "INFO") -> str:
    """Setup logging with timestamp and NCT ID in filename."""
    return setup_logging("filter_specific_trial", log_level)


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
        logger.error(f"Failed to fetch trial data: {e}")
        return []
    finally:
        try:
            os.unlink(temp_output)
        except Exception as e:
            logger.warning(f"Failed to delete temporary file {temp_output}: {e}")


def filter_specific_trial(
    clinical_record: str,
    nct_id: str,
    trials_file: Optional[str] = None,
    api_key: Optional[str] = None,
    cache_size: int = 100000,
    refresh_cache: bool = False,
    subgroup_aware: bool = True,
    use_trialgpt_approach: bool = False,
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
        subgroup_aware: Whether to enable subgroup awareness in filtering
        use_trialgpt_approach: Whether to use TrialGPT's two-stage approach (batch matching + R+E aggregation)

    Returns:
        Dictionary containing the filtering results
    """

    logger.info("=" * 60)
    logger.info(f"filter_specific_trial: Starting filtering process for trial {nct_id}")
    logger.info("=" * 60)

    # Initialize GPT client and filter
    logger.info("filter_specific_trial: Initializing GPT client and filter")
    if not api_key:
        raise ValueError("API key is required")
    gpt_client = create_gpt_client(api_key=api_key, cache_size=cache_size)
    gpt_filter = GPTTrialFilter(gpt_client, subgroup_aware=subgroup_aware)

    # Extract conditions from clinical record
    logger.info(f"filter_specific_trial: Reading clinical record from {clinical_record}")
    logger.info("filter_specific_trial: Extracting conditions from clinical record")
    logger.debug(f"filter_specific_trial: refresh_cache: {refresh_cache}")
    history_items = extract_conditions_from_record(
        clinical_record, gpt_client, refresh_cache=refresh_cache
    )
    logger.info(f"filter_specific_trial: Extracted {len(history_items)} conditions:")
    for item in history_items:
        logger.info(f"filter_specific_trial:  - {item}")

    # Load or fetch trial data
    if trials_file and os.path.exists(trials_file):
        logger.info(f"filter_specific_trial: Loading trial from file: {trials_file}")
        trials_data = load_json_list_file(trials_file)
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
        trial, history_items, refresh_cache, use_trialgpt_approach
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
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set the logging level",
    )
    parser.add_argument(
        "--no-subgroup-aware",
        action="store_true",
        help="Disable subgroup awareness in filtering (default: enabled)",
    )
    parser.add_argument(
        "--use-trialgpt-approach",
        action="store_true",
        default=False,
        dest="use_trialgpt_approach",
        help="Use TrialGPT's two-stage approach: (1) Batch criterion matching, (2) R+E aggregation scoring. Evaluates ALL criteria in one prompt (Stage 1) then aggregates with Relevance (R: 0-100) + Eligibility (E: -R to R) scoring (Stage 2).",
    )

    args = parser.parse_args()

    # Determine subgroup awareness setting (default: enabled, can be disabled with --no-subgroup-aware)
    subgroup_aware = not args.no_subgroup_aware

    # Setup logging
    log_file = setup_logging_for_trial(args.nct_id, args.log_level)

    logger.info("main: Starting specific trial filtering process")
    logger.info(f"main: Input clinical record: {args.clinical_record}")
    logger.info(f"main: Target NCT ID: {args.nct_id}")
    logger.info(f"main: Subgroup awareness: {'enabled' if subgroup_aware else 'disabled'}")
    if args.trials_file:
        logger.info(f"main: Input trials file: {args.trials_file}")
    else:
        logger.info("main: Will fetch trial data from ClinicalTrials.gov")

    # Get API key from command line or environment
    api_key = get_api_key(args.api_key)

    try:
        # Perform the filtering
        result = filter_specific_trial(
            clinical_record=args.clinical_record,
            nct_id=args.nct_id,
            trials_file=args.trials_file,
            api_key=api_key,
            cache_size=args.cache_size,
            refresh_cache=args.refresh_cache,
            subgroup_aware=subgroup_aware,
            use_trialgpt_approach=args.use_trialgpt_approach,
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
