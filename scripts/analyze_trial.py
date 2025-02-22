#!/usr/bin/env python3
import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

# Add parent directory to Python path to import modules
sys.path.append(str(Path(__file__).parent.parent))
from firecrawl.firecrawl import FirecrawlApp
from openai import OpenAI
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings

from base.clinical_trial import ClinicalTrial, ClinicalTrialsParser
from base.gpt_client import GPTClient
from base.pricing import OpenAITokenPricing
from base.prompt_cache import PromptCache
from clinical_trial_crawler.clinical_trial_crawler.spiders.clinical_trials_spider import (
    ClinicalTrialsSpider,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",  # Simplified format for readability
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Ensure logger is set to INFO level


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


def build_drug_keywords_prompt(trial: ClinicalTrial) -> str:
    """
    Build a prompt to generate drug name keywords for web search based on the trial's brief title.

    Args:
        trial: ClinicalTrial object containing the trial information

    Returns:
        str: A formatted prompt for generating drug name keywords
    """
    prompt = (
        "Please analyze the following clinical trial title and return a JSON list of drug names and compounds:\n\n"
        f"Trial Title: {trial.identification.brief_title}\n\n"
        "Return a JSON array containing:\n"
        "- Drug names explicitly mentioned in the title\n"
        "- Relevant drug classes or therapeutic categories if no specific drugs are named\n\n"
        "Example response:\n"
        '["rituximab", "cyclophosphamide", "doxorubicin", "vincristine"]\n\n'
        "Return only the JSON array with no other text."
    )
    return prompt


def analyze_drug_keywords(
    trial: ClinicalTrial, api_key: str, cache_size: int = 10000
) -> tuple[list[str], float]:
    """
    Analyze trial title to extract drug names and compounds using GPT with caching.

    Args:
        trial: ClinicalTrial object containing trial information
        api_key: OpenAI API key
        cache_size: Maximum number of cached responses to keep

    Returns:
        Tuple of (list of drug keywords, API cost)
    """
    gpt_client = GPTClient(
        api_key=api_key,
        model="gpt-4o-mini",
        cache_size=cache_size,
        temperature=0.1,
    )

    # Build the prompt
    prompt = build_drug_keywords_prompt(trial)

    try:
        # Make API call with retry logic and JSON validation
        result, cost = gpt_client.call_with_retry(
            prompt=prompt,
            system_role="You are a clinical trial analyst focused on identifying drug names and compounds.",
            response_format={"type": "json_object"},
            validate_json=True,
        )
        return result, cost

    except Exception as e:
        logger.error(f"Error in GPT drug keyword analysis: {str(e)}")
        raise


def search_google(api_key: str, query: str) -> list[dict]:
    """
    Search Google using Firecrawl's search endpoint.

    Args:
        api_key: Firecrawl API key
        query: Search query

    Returns:
        List of search results containing URLs, titles, and descriptions
    """
    try:
        # Initialize Firecrawl client
        app = FirecrawlApp(api_key=api_key)

        # Make search request
        search_results = app.search(query=query)

        if not search_results.get("success"):
            logger.error("Search request failed")
            return []

        return search_results.get("data", [])

    except Exception as e:
        logger.error(f"Error performing search: {str(e)}")
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
        "--api-key",
        help="OpenAI API key (alternatively, set OPENAI_API_KEY environment variable)",
    )
    parser.add_argument(
        "--firecrawl-api-key",
        help="Firecrawl API key (alternatively, set FIRECRAWL_API_KEY environment variable)",
    )
    parser.add_argument(
        "--cache-size",
        type=int,
        default=10000,
        help="Maximum number of cached responses to keep",
    )
    args = parser.parse_args()

    # Get API keys from arguments or environment
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    firecrawl_api_key = args.firecrawl_api_key or os.getenv("FIRECRAWL_API_KEY")

    if not api_key:
        logger.error(
            "OpenAI API key must be provided via --api-key or OPENAI_API_KEY environment variable"
        )
        sys.exit(1)

    if not firecrawl_api_key:
        logger.error(
            "Firecrawl API key must be provided via --firecrawl-api-key or FIRECRAWL_API_KEY environment variable"
        )
        sys.exit(1)

    # Read clinical record from input file if specified
    if args.input_file:
        try:
            with open(args.input_file, "r") as file:
                clinical_record = file.read().strip()
                logger.info(f"Read clinical record from {args.input_file}")
                # Process the clinical record as needed
                # For example, you might parse it or log it
                logger.info(f"Clinical Record Content: {clinical_record[:200]}...")
        except FileNotFoundError:
            logger.error(f"Input file not found: {args.input_file}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error reading input file: {e}")
            sys.exit(1)

    # Fetch and parse the trial data
    nct_id: str = args.nct_id
    json_data = fetch_trial_data(nct_id)
    trials_parser = ClinicalTrialsParser(json_data)

    # Find the specific trial
    trial = trials_parser.get_trial_by_nct_id(args.nct_id)
    if trial is None:
        logger.error(f"Trial with NCT ID {args.nct_id} not found")
        sys.exit(1)

    # Get drug keywords with caching
    try:
        drug_keywords, cost = analyze_drug_keywords(
            trial, api_key, cache_size=args.cache_size
        )
        logger.info(f"Identified drug keywords: {drug_keywords}")
        logger.info(f"API cost: ${cost:.4f}")

        # Search for each drug keyword
        for drug in drug_keywords:
            logger.info(f"\nSearching for drug: {drug}")
            search_query = f"{drug} clinical trials research"
            results = search_google(firecrawl_api_key, search_query)

            if results:
                logger.info(f"Found {len(results)} results for {drug}:")
                for idx, result in enumerate(results, 1):
                    logger.info(f"{idx}. {result.get('title', 'No title')}")
                    logger.info(f"   URL: {result.get('url', 'No URL')}")
                    logger.info(
                        f"   Description: {result.get('description', 'No description')}\n"
                    )
            else:
                logger.info(f"No results found for {drug}")

    except Exception as e:
        logger.error(f"Failed to analyze drug keywords: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
