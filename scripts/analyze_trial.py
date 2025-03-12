import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

import requests

# Add parent directory to Python path to import modules
sys.path.append(str(Path(__file__).parent.parent))
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings

from base.clinical_trial import ClinicalTrialsParser
from base.disease_expert import extract_disease_from_record
from base.drug_analyzer import analyze_drug_effectiveness
from base.gpt_client import GPTClient
from base.perplexity import PerplexityClient
from base.trial_analyzer import (
    CLINICAL_TRIAL_SYSTEM_PROMPT,
    analyze_drugs_and_get_recommendation,
    build_recommendation_prompt,
    parse_recommendation_response,
)
from base.utils import read_input_file
from clinical_trial_crawler.clinical_trial_crawler.spiders.clinical_trials_spider import (
    ClinicalTrialsSpider,
)

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Set logger level to INFO
logger.propagate = False  # Prevent propagation to parent loggers

# Configure base.pricing logger
pricing_logger = logging.getLogger("base.pricing")
pricing_logger.setLevel(logging.DEBUG)
pricing_logger.propagate = False  # Prevent propagation to parent loggers

# Configure base.perplexity logger
perplexity_logger = logging.getLogger("base.perplexity")
perplexity_logger.setLevel(logging.DEBUG)
perplexity_logger.propagate = False  # Prevent propagation to parent loggers

# Configure base.trial_analyzer logger
trial_analyzer_logger = logging.getLogger("base.trial_analyzer")
trial_analyzer_logger.setLevel(logging.DEBUG)
trial_analyzer_logger.propagate = False  # Prevent propagation to parent loggers

# Configure handler only once
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)  # Standard format with timestamp, logger name, and level

# Add handler to all loggers
logger.addHandler(handler)
pricing_logger.addHandler(handler)
perplexity_logger.addHandler(handler)
trial_analyzer_logger.addHandler(handler)


def fetch_trial_data(nct_id: str) -> list[dict]:
    """Fetch clinical trial data directly from ClinicalTrials.gov."""
    # Create a temporary file to store the spider output
    with tempfile.NamedTemporaryFile(
        mode="w+", suffix=".json", delete=False
    ) as tmp_file:
        temp_output = tmp_file.name
        logger.info(f"Using temporary file: {temp_output}")

    try:
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
        process.start()  # This will block until the crawl is complete

        # Read the results
        try:
            with open(temp_output, "r") as f:
                content = f.read().strip()
                if not content:
                    logger.error("No data was written to the output file")
                    return []
                logger.debug(f"Read content: {content[:200]}...")
                return json.loads(content)
        except FileNotFoundError:
            logger.error(f"Output file not found: {temp_output}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse spider output: {e}")
            logger.error(f"Content of output file: {content[:200]}...")
            return []

    finally:
        # Clean up the temporary file
        try:
            Path(temp_output).unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"Failed to delete temporary file {temp_output}: {e}")


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
    api_key = args.openai_api_key or os.getenv("OPENAI_API_KEY")
    perplexity_api_key = args.perplexity_api_key or os.getenv("PERPLEXITY_API_KEY")

    if not api_key:
        logger.error(
            "OpenAI API key must be provided via --openai-api-key or OPENAI_API_KEY environment variable"
        )
        sys.exit(1)

    if not perplexity_api_key:
        logger.error(
            "Perplexity API key must be provided via --perplexity-api-key or PERPLEXITY_API_KEY environment variable"
        )
        sys.exit(1)

    # Read clinical record from input file if specified
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

    # Initialize Perplexity client
    perplexity_client = PerplexityClient(perplexity_api_key)
    gpt_client = GPTClient(api_key)
    # Replace the drug analysis and recommendation section with:
    try:
        recommendation, reason, total_cost = analyze_drugs_and_get_recommendation(
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
