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
from clinical_trial_crawler.clinical_trial_crawler.spiders.clinical_trials_spider import (
    ClinicalTrialsSpider,
)

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Set logger level to INFO
logger.propagate = False  # Prevent propagation to parent loggers

# Configure handler only once
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(message)s")
)  # Simplified format for readability
logger.addHandler(handler)


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
    "<instructions>Instructions:\n"
    "1. Carefully analyze the patient's clinical record and trial details.\n"
    "2. Search for and review published research on the effectiveness of similar drugs/treatments.\n"
    "3. Consider the potential therapeutic benefit based on both trial info and research findings.\n"
    "4. Evaluate risks vs potential benefits for the patient's condition.\n"
    "5. Factor in alternative treatment options available to the patient.\n"
    "6. Based on your comprehensive analysis, choose the most appropriate recommendation level.\n"
    "7. Provide an explanation that includes relevant research findings on similar treatments.</instructions>"
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
        f'<clinical_record>Clinical Record:\n"{clinical_record}"</clinical_record>\n\n'
        f'<trial_info>Trial Information:\n"{trial_info_str}"</trial_info>\n\n'
        "<output_request>Please search for and analyze published research on similar treatments, then provide your explanation and recommendation level."
    )
    return prompt


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
        "--cache-size",
        type=int,
        default=10000,
        help="Maximum number of cached responses to keep",
    )
    args = parser.parse_args()

    # Get API keys from arguments or environment
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")

    if not api_key:
        logger.error(
            "OpenAI API key must be provided via --api-key or OPENAI_API_KEY environment variable"
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

    # Build the prompt
    prompt = build_recommendation_prompt(clinical_record, trial)
    logger.info(f"main: Recommendation Prompt:\n{prompt}")

    # Call the Perplexity AI API
    url = "https://api.perplexity.ai/chat/completions"

    payload = {
        "model": "sonar",
        "messages": [
            {
                "role": "system",
                "content": CLINICAL_TRIAL_SYSTEM_PROMPT,
            },
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 1000,
        "temperature": 0.2,
        "top_p": 0.9,
        "return_images": False,
        "return_related_questions": False,
        "stream": False,
    }

    headers = {
        "Authorization": "Bearer " + os.getenv("PERPLEXITY_API_KEY"),
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()
        logger.info("main: Successfully received AI analysis")
        logger.info(f"main: AI Analysis: {result['choices'][0]['message']['content']}")
    except requests.exceptions.RequestException as e:
        logger.error(f"main: Error calling Perplexity AI API: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
