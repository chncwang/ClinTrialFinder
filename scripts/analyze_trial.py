#!/usr/bin/env python3
import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

# Add parent directory to Python path to import modules
sys.path.append(str(Path(__file__).parent.parent))
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings

from base.clinical_trial import ClinicalTrial, ClinicalTrialsParser
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


def format_duration(days: Optional[int]) -> str:
    """Format duration in days to a readable string."""
    if days is None:
        return "Unknown duration"

    years = days // 365
    remaining_days = days % 365
    months = remaining_days // 30
    remaining_days = remaining_days % 30

    parts = []
    if years > 0:
        parts.append(f"{years} year{'s' if years != 1 else ''}")
    if months > 0:
        parts.append(f"{months} month{'s' if months != 1 else ''}")
    if remaining_days > 0:
        parts.append(f"{remaining_days} day{'s' if remaining_days != 1 else ''}")

    return ", ".join(parts) if parts else "0 days"


def log_trial_info(trial: ClinicalTrial):
    """Log detailed information about a clinical trial."""
    # Basic Information
    logger.info("\n=== Trial Information ===")
    logger.info(f"NCT ID: {trial.identification.nct_id}")
    logger.info(f"URL: {trial.identification.url}")
    logger.info(f"\nTitle: {trial.identification.brief_title}")
    if trial.identification.official_title:
        logger.info(f"Official Title: {trial.identification.official_title}")
    if trial.identification.acronym:
        logger.info(f"Acronym: {trial.identification.acronym}")

    # Status Information
    logger.debug("\n=== Status ===")
    logger.debug(f"Current Status: {trial.status.overall_status}")
    if trial.status.start_date:
        logger.debug(f"Start Date: {trial.status.start_date.strftime('%Y-%m-%d')}")
    if trial.status.completion_date:
        logger.debug(
            f"Completion Date: {trial.status.completion_date.strftime('%Y-%m-%d')}"
        )
    if trial.study_duration_days is not None:
        logger.debug(f"Study Duration: {format_duration(trial.study_duration_days)}")

    # Description
    logger.debug("\n=== Description ===")
    logger.debug("Brief Summary:")
    logger.debug(trial.description.brief_summary)
    if trial.description.conditions:
        logger.debug("\nConditions:")
        for condition in trial.description.conditions:
            logger.debug(f"- {condition}")
    if trial.description.keywords:
        logger.debug("\nKeywords:")
        for keyword in trial.description.keywords:
            logger.debug(f"- {keyword}")

    # Design
    logger.debug("\n=== Study Design ===")
    logger.debug(f"Study Type: {trial.design.study_type}")
    if trial.design.phases:
        logger.debug(
            f"Phase{'s' if len(trial.design.phases) > 1 else ''}: {', '.join(map(str, trial.design.phases))}"
        )
    logger.debug(f"Enrollment: {trial.design.enrollment} participants")

    if trial.design.arms:
        logger.debug("\nStudy Arms:")
        for arm in trial.design.arms:
            logger.debug(f"\n- Name: {arm.get('name', 'Not specified')}")
            logger.debug(f"  Type: {arm.get('type', 'Not specified')}")
            if arm.get("description"):
                logger.debug(f"  Description: {arm['description']}")
            if arm.get("interventions"):
                logger.debug(f"  Interventions: {', '.join(arm['interventions'])}")

    # Eligibility
    logger.debug("\n=== Eligibility Criteria ===")
    logger.debug(trial.eligibility.criteria)
    logger.debug(f"\nGender: {trial.eligibility.gender}")
    logger.debug(
        f"Age Range: {trial.eligibility.minimum_age} to {trial.eligibility.maximum_age}"
    )
    logger.debug(
        f"Healthy Volunteers: {'Accepted' if trial.eligibility.healthy_volunteers else 'Not accepted'}"
    )

    # Locations
    logger.debug("\n=== Study Locations ===")
    for location in trial.contacts_locations.locations:
        location_parts = []
        if location.facility:
            location_parts.append(location.facility)
        if location.city:
            location_parts.append(location.city)
        if location.state:
            location_parts.append(location.state)
        if location.country:
            location_parts.append(location.country)

        location_str = " - ".join(location_parts)
        status_str = f" ({location.status})" if location.status else ""
        logger.debug(f"- {location_str}{status_str}")

    # Sponsor Information
    logger.debug("\n=== Sponsorship ===")
    logger.debug(f"Lead Sponsor: {trial.sponsor.lead_sponsor}")
    if trial.sponsor.collaborators:
        logger.debug("Collaborators:")
        for collaborator in trial.sponsor.collaborators:
            logger.debug(f"- {collaborator}")


def build_recommendation_prompt(clinical_record: str, trial_info: str) -> str:
    """
    Constructs a prompt for an AI to evaluate patient clinical record against trial information
    and return a recommendation level based on potential benefit to the patient.

    Parameters:
    - clinical_record (str): The patient's clinical record to be evaluated
    - trial_info (str): The clinical trial information to compare against

    Returns:
    - str: A formatted prompt string for the AI.
    """
    prompt = (
        "You are a clinical research expert with extensive experience in evaluating patient eligibility and treatment outcomes. Your expertise includes analyzing clinical trials, published research, and making evidence-based recommendations for patient care. "
        "Your task is to assess if a clinical trial would be beneficial for a patient. "
        "You will analyze published research and clinical evidence on similar drugs and treatments to inform your recommendation. "
        "The possible recommendation levels are:\n\n"
        "- Strongly Recommended\n"
        "- Recommended\n"
        "- Neutral\n"
        "- Not Recommended\n\n"
        "Instructions:\n"
        "1. Carefully analyze the patient's clinical record and trial details.\n"
        "2. Search for and review published research on the effectiveness of similar drugs/treatments.\n"
        "3. Consider the potential therapeutic benefit based on both trial info and research findings.\n"
        "4. Evaluate risks vs potential benefits for the patient's condition.\n"
        "5. Factor in alternative treatment options available to the patient.\n"
        "6. Based on your comprehensive analysis, choose the most appropriate recommendation level.\n"
        "7. Provide an explanation that includes relevant research findings on similar treatments.\n\n"
        f'Clinical Record:\n"{clinical_record}"\n'
        f'Trial Information:\n"{trial_info}"\n'
        "Please search for and analyze published research on similar treatments, then provide:\n"
        "Recommendation Level: "
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
    args = parser.parse_args()

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
    json_data = fetch_trial_data(args.nct_id)
    trials_parser = ClinicalTrialsParser(json_data)

    # Find the specific trial
    trial = trials_parser.get_trial_by_nct_id(args.nct_id)
    if trial is None:
        logger.error(f"Trial with NCT ID {args.nct_id} not found")
        sys.exit(1)

    # Log the trial information
    log_trial_info(trial)


if __name__ == "__main__":
    main()
