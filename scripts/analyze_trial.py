#!/usr/bin/env python3
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

from base.clinical_trial import ClinicalTrial, ClinicalTrialsParser
from base.disease_extractor import extract_disease_from_record
from base.drug_analyzer import analyze_drug_effectiveness
from base.gpt_client import GPTClient
from base.perplexity import PerplexityClient
from base.pricing import AITokenPricing
from base.prompts import (
    CLINICAL_TRIAL_SYSTEM_PROMPT,
    build_recommendation_prompt,
    parse_recommendation_response,
)
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

# Configure handler only once
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)  # Standard format with timestamp, logger name, and level

# Add handler to all loggers
logger.addHandler(handler)
pricing_logger.addHandler(handler)
perplexity_logger.addHandler(handler)


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


def analyze_drugs_and_get_recommendation(
    novel_drugs: list[str],
    disease: str,
    clinical_record: str,
    trial: ClinicalTrial,
    perplexity_client: PerplexityClient,
    gpt_client: GPTClient,
) -> tuple[str, str, float]:
    """
    Analyze drug effectiveness and generate AI recommendation.

    Returns:
        tuple containing:
            - Recommendation level (Strongly Recommended, Recommended, etc.)
            - Reason for the recommendation
            - Total cost of API calls
    """
    total_cost = 0.0
    drug_analyses: dict[str, str] = {}

    # Analyze each drug's effectiveness
    if novel_drugs and disease:
        for drug in novel_drugs:
            logger.info(f"Analyzing effectiveness of {drug} for {disease}")
            analysis, citations, cost = analyze_drug_effectiveness(
                drug, disease, perplexity_client
            )
            total_cost += cost

            if analysis:
                drug_analyses[drug] = analysis
                logger.info(f"Drug Analysis: {analysis}")
                if citations:
                    logger.info(f"Citations ({len(citations)}):")
                    for i, citation in enumerate(citations, 1):
                        logger.info(f"Citation {i}: {citation}")
                logger.info(f"Cost: ${cost:.6f}")

    # Generate recommendation
    prompt = build_recommendation_prompt(clinical_record, trial, drug_analyses)
    logger.info(f"Recommendation Prompt:\n{prompt}")

    completion, cost = gpt_client.call_gpt(
        prompt=prompt,
        system_role=CLINICAL_TRIAL_SYSTEM_PROMPT,
        temperature=0.2,
    )
    total_cost += cost

    if completion is None:
        raise RuntimeError("Failed to get AI analysis")

    recommendation, reason = parse_recommendation_response(completion)
    logger.info(f"Recommendation: {recommendation}")
    logger.info(f"Reason: {reason}")

    return recommendation, reason, total_cost


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
        try:
            with open(args.input_file, "r") as file:
                clinical_record = file.read().strip()
                logger.info(f"main: Read clinical record from {args.input_file}")
                # Process the clinical record as needed
                # For example, you might parse it or log it
                logger.info(
                    f"main: Clinical Record Content: {clinical_record[:200]}..."
                )
        except FileNotFoundError:
            logger.error(f"main: Input file not found: {args.input_file}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"main: Error reading input file: {e}")
            sys.exit(1)

    # Fetch and parse the trial data
    nct_id: str = args.nct_id
    json_data = fetch_trial_data(nct_id)
    trials_parser = ClinicalTrialsParser(json_data)

    # Find the specific trial
    trial = trials_parser.get_trial_by_nct_id(args.nct_id)
    if trial is None:
        logger.error(f"main: Trial with NCT ID {args.nct_id} not found")
        sys.exit(1)

    # Extract disease from clinical record
    gpt_client = GPTClient(api_key)
    disease, cost = extract_disease_from_record(clinical_record, gpt_client)
    logger.info(f"main: Extracted Disease: {disease}")
    logger.info(f"main: Cost: ${cost:.6f}")

    # Get novel drugs from trial title
    gpt_client = GPTClient(api_key)
    novel_drugs, cost = trial.get_novel_drugs_from_title(gpt_client)
    logger.info(f"main: Novel Drugs: {novel_drugs}")
    logger.info(f"main: Cost: ${cost:.6f}")

    # Initialize Perplexity client
    perplexity_client = PerplexityClient(perplexity_api_key)

    # Replace the drug analysis and recommendation section with:
    try:
        recommendation, reason, total_cost = analyze_drugs_and_get_recommendation(
            novel_drugs=novel_drugs,
            disease=disease,
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
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
