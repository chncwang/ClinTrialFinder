#!/usr/bin/env python3
import argparse
import datetime
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from openai import OpenAI

# Add parent directory to Python path to import base module
sys.path.append(str(Path(__file__).parent.parent))
from base.clinical_trial import ClinicalTrial, ClinicalTrialsParser

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class GPTTrialFilter:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    def create_prompt(self, trial: ClinicalTrial, condition: str) -> str:
        """Create a prompt for GPT to evaluate the trial against the condition."""
        return f"""You are evaluating a clinical trial to determine if it meets a specific condition.

Trial Details:
- Title: {trial.identification.brief_title}

Detailed Eligibility Criteria:
{trial.eligibility.criteria}

Condition to Evaluate:
{condition}

Analyze the trial title and eligibility criteria to determine if this trial meets the following condition: {condition}. Consider both inclusion and exclusion criteria carefully.
If you are uncertain whether the trial meets the condition, return eligible as true.

Provide your analysis in JSON format with these fields:
- "reason": detailed explanation of your decision, citing specific criteria
- "eligible": boolean indicating if trial meets the condition (true if uncertain)

Example response:
{{"reason": "[specific reasons]", "eligible": true}}"""

    def evaluate_trial(self, trial: ClinicalTrial, condition: str) -> Tuple[bool, str]:
        """Evaluate a single trial against the given condition using GPT."""
        try:
            prompt = self.create_prompt(trial, condition)

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a clinical trial analyst. Analyze the trial data and provide clear yes/no decisions with explanations. Your response must be valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
            )

            # Parse the response
            result = json.loads(response.choices[0].message.content)
            return result["eligible"], result["reason"]

        except Exception as e:
            logger.error(
                f"Error evaluating trial {trial.identification.nct_id}: {str(e)}"
            )
            return False, f"Error during evaluation: {str(e)}"


def load_json_file(file_path: str) -> List[dict]:
    """Load and parse a JSON file."""
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"File '{file_path}' not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        logger.error(f"File '{file_path}' is not a valid JSON file.")
        sys.exit(1)


def save_json_file(data: List[Dict], output_path: str):
    """Save filtered trials to a JSON file."""
    try:
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Results saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Filter clinical trials using GPT-4 based on custom conditions"
    )
    parser.add_argument(
        "json_file", help="Path to the JSON file containing clinical trials data"
    )
    parser.add_argument(
        "condition",
        help="Condition to filter trials (e.g., 'doesn\\'t have a measurable site')",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=f"filtered_trials_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--api-key",
        help="OpenAI API key (alternatively, set OPENAI_API_KEY environment variable)",
    )

    args = parser.parse_args()

    # Get API key from arguments or environment
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error(
            "OpenAI API key must be provided via --api-key or OPENAI_API_KEY environment variable"
        )
        sys.exit(1)

    # Initialize GPT filter
    gpt_filter = GPTTrialFilter(api_key)

    # Load and parse trials
    json_data = load_json_file(args.json_file)
    trials_parser = ClinicalTrialsParser(json_data)

    # Filter trials using GPT
    filtered_trials = []
    total_trials = len(trials_parser.trials)

    logger.info(f"Processing {total_trials} trials...")

    for i, trial in enumerate(trials_parser.trials, 1):
        logger.info(
            f"Processing trial {i}/{total_trials}: {trial.identification.nct_id}"
        )
        eligible, reason = gpt_filter.evaluate_trial(trial, args.condition)

        if eligible:
            logger.info(
                f"Trial {trial.identification.nct_id} matches condition: {reason}"
            )
            filtered_trials.append(trial.to_dict())
        else:
            logger.debug(
                f"Trial {trial.identification.nct_id} does not match condition: {reason}"
            )

    # Save results
    save_json_file(filtered_trials, args.output)
    logger.info(f"Found {len(filtered_trials)} matching trials out of {total_trials}")


if __name__ == "__main__":
    main()
