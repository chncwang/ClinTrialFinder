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
from base.perplexity import PerplexityClient
from base.pricing import AITokenPricing
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


CLINICAL_TRIAL_SYSTEM_PROMPT = (
    "<role>You are a clinical research expert with extensive experience in evaluating patient eligibility and treatment outcomes. "
    "Your expertise includes analyzing clinical trials, published research, and making evidence-based recommendations for patient care.</role>\n\n"
    "<task>Your task is to assess if a clinical trial would be beneficial for a patient. "
    "You will analyze published research and clinical evidence on similar drugs and treatments to inform your recommendation.</task>\n\n"
    "<recommendation_levels>The possible recommendation levels are:\n\n"
    "- Strongly Recommended\n"
    "- Recommended\n"
    "- Neutral\n"
    "- Not Recommended</recommendation_levels>\n\n"
)


def build_recommendation_prompt(clinical_record: str, trial_info: ClinicalTrial) -> str:
    """
    Constructs a prompt for an AI to evaluate patient clinical record against trial information
    and return a recommendation level based on potential benefit to the patient.

    Parameters:
    - clinical_record (str): The patient's clinical record to be evaluated
    - trial_info (ClinicalTrial): The clinical trial information to compare against

    Returns:
    - str: A formatted prompt string for the AI.
    """
    trial_info_str = (
        f"NCT ID: {trial_info.identification.nct_id}\n"
        f"URL: {trial_info.identification.url}\n"
        f"Brief Title: {trial_info.identification.brief_title}\n"
        f"Official Title: {trial_info.identification.official_title}\n"
        f"Overall Status: {trial_info.status.overall_status}\n"
        f"Brief Summary: {trial_info.description.brief_summary}\n"
        f"Detailed Description: {trial_info.description.detailed_description}\n"
        f"Study Type: {trial_info.design.study_type}\n"
        f"Phases: {', '.join(map(str, trial_info.design.phases))}\n"
        f"Arms: {json.dumps(trial_info.design.arms, indent=2)}\n"
        f"Lead Sponsor: {trial_info.sponsor.lead_sponsor}\n"
        f"Collaborators: {', '.join(trial_info.sponsor.collaborators)}"
    )

    return (
        f'<clinical_record>\nClinical Record:\n"{clinical_record}"\n</clinical_record>\n\n'
        f'<trial_info>\nTrial Information:\n"{trial_info_str}"\n</trial_info>\n\n'
        "<output_request>\nIf a novel drug is mentioned in the study title, please focus on that drug. Search for and analyze published research on this drug's effectiveness in treating similar diseases or conditions. Based on this evidence, provide your explanation and recommendation level.\n</output_request>"
    )
    return prompt


def extract_disease_from_record(
    clinical_record: str, gpt_client: GPTClient
) -> tuple[str | None, float]:
    """
    Extracts the primary disease or condition from a clinical record using GPT.

    Parameters:
    - clinical_record (str): The patient's clinical record text
    - gpt_client (GPTClient): Initialized GPT client for making API calls

    Returns:
    - tuple[str | None, float]: A tuple containing the extracted disease name (or None if not found)
                               and the cost of the API call
    """
    prompt = (
        "Extract the primary disease or medical condition from the following clinical record. "
        "Return only the disease name without any additional text or explanation.\n\n"
        f"Clinical Record:\n{clinical_record}"
    )

    completion, cost = gpt_client.get_completion(prompt)
    if completion is None:
        logger.error("Failed to extract disease from clinical record")
        return None, cost

    disease_name = completion.strip()
    logger.info(f"Extracted disease name: {disease_name}")
    return disease_name, cost


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

    # Get novel drugs from trial title
    gpt_client = GPTClient(api_key)
    novel_drugs, cost = trial.get_novel_drugs_from_title(gpt_client)
    logger.info(f"main: Novel Drugs: {novel_drugs}")
    logger.info(f"main: Cost: ${cost:.6f}")

    # Initialize Perplexity client
    perplexity_client = PerplexityClient(perplexity_api_key)

    # Build the prompt
    prompt = build_recommendation_prompt(clinical_record, trial)
    logger.info(f"main: Recommendation Prompt:\n{prompt}")

    # Call the Perplexity AI API using the client
    messages = [
        {
            "role": "system",
            "content": CLINICAL_TRIAL_SYSTEM_PROMPT,
        },
        {"role": "user", "content": prompt},
    ]

    completion, citations, cost = perplexity_client.get_completion(
        messages, max_tokens=2000
    )
    if completion is None:
        logger.error("main: Failed to get AI analysis")
        sys.exit(1)

    logger.info("main: Successfully received AI analysis")
    logger.info(f"main: AI Analysis: {completion}")
    if citations:
        logger.info(f"main: Citations ({len(citations)}):")
        for i, citation in enumerate(citations, 1):
            logger.info(f"main: Citation {i}: {citation}")
    logger.info(f"main: Cost: ${cost:.6f}")


if __name__ == "__main__":
    main()
