import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List
import logging

logger = logging.getLogger(__name__)

# Add parent directory to Python path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from base.clinical_trial import ClinicalTrialsParser
from base.gpt_client import GPTClient
from base.perplexity import PerplexityClient
from base.trial_expert import analyze_drugs_and_get_recommendation
from base.trial_downloader import TrialDownloader
from base.utils import read_input_file, get_api_key, create_gpt_client

# Configure logging using centralized configuration
from base.logging_config import setup_logging
log_filename = setup_logging("analyze_trial", log_level="INFO")

# Log the filename being used
logger.info(f"All logs will be written to: {os.path.abspath(log_filename)}")


def fetch_trial_data(nct_id: str) -> List[Dict[str, Any]]:
    """Fetch clinical trial data directly from ClinicalTrials.gov."""
    try:
        downloader = TrialDownloader()
        trial_data = downloader.fetch_by_nct_id(nct_id)
        if trial_data:
            return [trial_data]
        else:
            logger.error("No data was returned from the API")
            return []
    except Exception as e:
        logger.error(f"Failed to fetch trial data: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and display clinical trial information"
    )
    parser.add_argument("nct_id", help="NCT ID of the trial to analyze")
    parser.add_argument(
        "--input_file", help="Path to the input text file containing clinical record"
    )
    parser.add_argument(
        "--openai-api-key",
        help="OpenAI API key (alternatively, set OPENAI_API_KEY environment variable)",
    )
    parser.add_argument(
        "--perplexity-api-key",
        help="Perplexity API key (alternatively, set PERPLEXITY_API_KEY environment variable)",
    )
    parser.add_argument(
        "--cache-size",
        type=int,
        default=10000,
        help="Maximum number of cached responses to keep",
    )
    args = parser.parse_args()

    # Get API keys from arguments or environment
    api_key = get_api_key(args.openai_api_key)
    perplexity_api_key = get_api_key(args.perplexity_api_key, "PERPLEXITY_API_KEY")

    # Read clinical record from input file if specified
    clinical_record = ""
    if args.input_file:
        clinical_record = read_input_file(args.input_file)

    # Fetch and parse the trial data
    nct_id: str = args.nct_id
    json_data = fetch_trial_data(nct_id)
    trials_parser = ClinicalTrialsParser(json_data)

    # Find the specific trial
    trial = trials_parser.get_trial_by_nct_id(args.nct_id)
    if trial is None:
        logger.error(f"main: Trial with NCT ID {args.nct_id} not found")
        sys.exit(1)

    # Initialize clients
    perplexity_client = PerplexityClient(perplexity_api_key)
    gpt_client = create_gpt_client(api_key=api_key, cache_size=args.cache_size)
    # Replace the drug analysis and recommendation section with:
    try:
        recommendation, reason, _, total_cost = analyze_drugs_and_get_recommendation(
            clinical_record=clinical_record,
            trial=trial,
            perplexity_client=perplexity_client,
            gpt_client=gpt_client,
        )
        logger.info(f"Recommendation Level: {recommendation}")
        logger.info(f"Recommendation Reason: {reason}")
        logger.info(f"Total Cost: ${total_cost:.6f}")
    except ValueError as e:
        logger.error(f"Error parsing recommendation: {e}")
        raise
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        raise


if __name__ == "__main__":
    main()
