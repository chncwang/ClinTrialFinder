#!/usr/bin/env python3
import argparse
import datetime
import hashlib
import json
import logging
import os
import pickle
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

from openai import OpenAI

# Add parent directory to Python path to import base module
sys.path.append(str(Path(__file__).parent.parent))
from base.clinical_trial import ClinicalTrial, ClinicalTrialsParser

# Configure logging
log_file = f"filter_trials_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PromptCache:
    def __init__(self, cache_dir: str = ".cache", max_size: int = 1000):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_size = max_size
        self.cache_index = OrderedDict()
        self._load_cache_index()

    def _get_cache_key(self, prompt: str) -> str:
        """Generate a unique cache key for a prompt."""
        return hashlib.md5(prompt.encode()).hexdigest()

    def _load_cache_index(self):
        """Load the cache index from disk."""
        index_path = self.cache_dir / "cache_index.pkl"
        if index_path.exists():
            with open(index_path, "rb") as f:
                self.cache_index = pickle.load(f)

    def _save_cache_index(self):
        """Save the cache index to disk."""
        with open(self.cache_dir / "cache_index.pkl", "wb") as f:
            pickle.dump(self.cache_index, f)

    def get(self, prompt: str) -> dict | None:
        """Get cached result for a prompt."""
        cache_key = self._get_cache_key(prompt)
        if cache_key in self.cache_index:
            cache_file = self.cache_dir / f"{cache_key}.json"
            if cache_file.exists():
                with open(cache_file, "r") as f:
                    return json.load(f)
        return None

    def set(self, prompt: str, result: dict):
        """Cache result for a prompt."""
        cache_key = self._get_cache_key(prompt)

        # Enforce cache size limit
        while len(self.cache_index) >= self.max_size:
            # Remove oldest entry
            oldest_key, _ = self.cache_index.popitem(last=False)
            (self.cache_dir / f"{oldest_key}.json").unlink(missing_ok=True)

        # Save new entry
        with open(self.cache_dir / f"{cache_key}.json", "w") as f:
            json.dump(result, f)

        self.cache_index[cache_key] = True
        self._save_cache_index()


class EligibilityCriteriaError(Exception):
    """Custom exception for eligibility criteria format errors."""

    pass


class GPTTrialFilter:
    def __init__(self, api_key: str, cache_size: int = 1000):
        self.client = OpenAI(api_key=api_key)
        self.cache = PromptCache(max_size=cache_size)

    def extract_inclusion_criteria(self, criteria: str) -> str:
        """
        Extract inclusion criteria from the full eligibility criteria text.
        """
        try:
            parts = criteria.split("Exclusion Criteria:")
            if len(parts) < 2:
                raise EligibilityCriteriaError(
                    "Could not find 'Exclusion Criteria:' section"
                )

            inclusion_parts = parts[0].split("Inclusion Criteria:")
            if len(inclusion_parts) < 2:
                raise EligibilityCriteriaError(
                    "Could not find 'Inclusion Criteria:' section"
                )

            inclusion_criteria = inclusion_parts[1].strip()
            if not inclusion_criteria:
                raise EligibilityCriteriaError("Inclusion criteria section is empty")

            return inclusion_criteria

        except Exception as e:
            raise EligibilityCriteriaError(
                f"Error parsing eligibility criteria: {str(e)}"
            )

    def create_prompt(self, trial: ClinicalTrial, conditions: str | list[str]) -> str:
        """Create a prompt for GPT to evaluate the trial against one or more conditions."""
        conditions_list = conditions if isinstance(conditions, list) else [conditions]
        conditions_text = "\n".join(f"- {condition}" for condition in conditions_list)

        try:
            inclusion_criteria = self.extract_inclusion_criteria(
                trial.eligibility.criteria
            )
        except EligibilityCriteriaError as e:
            logger.error(
                f"Error processing trial {trial.identification.nct_id}: {str(e)}"
            )
            raise

        return f"""You are evaluating a clinical trial to determine if it meets specific conditions.

Trial Details:
- Title: {trial.identification.brief_title}

Inclusion Criteria:
{inclusion_criteria}

Patient Conditions to Evaluate:
{conditions_text}

Please analyze if this trial is suitable for the patient with the listed conditions by carefully evaluating:
1. The trial title - does it align with the conditions being evaluated?
2. The inclusion criteria - what characteristics the patient should have?

Respond with a JSON object containing:
- "reason": An explanation of why the trial is or is not suitable
- "eligible": true if the patient could potentially qualify, false if either would be disqualifying

Example response:
{{"reason": "[specific reasons]", "eligible": true}}"""

    def evaluate_trial(
        self, trial: ClinicalTrial, condition: str
    ) -> Tuple[bool, str, float]:
        """Evaluate a single trial against the given condition using GPT."""
        try:
            prompt = self.create_prompt(trial, condition)

            # Check cache first
            cached_result = self.cache.get(prompt)
            if cached_result is not None:
                logger.debug("GPTTrialFilter.evaluate_trial: Using cached result")
                return cached_result["eligible"], cached_result["reason"], 0.0

            input_content = "You are a clinical trial analyst focused on inclusion criteria. Analyze the trial data and provide clear yes/no decisions with explanations. Your response must be valid JSON."
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": input_content,
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
            )

            # Parse the response
            result_content = response.choices[0].message.content
            result = json.loads(result_content)

            # Estimate the cost of the request
            input_string_length = len(prompt) + len(input_content)
            estimated_input_tokens = input_string_length / 4
            estimated_output_tokens = len(result_content) / 4
            estimated_cost = (
                estimated_input_tokens * 0.15e-6 + estimated_output_tokens * 0.6e-6
            )
            logger.debug(
                f"GPTTrialFilter.evaluate_trial: Estimated cost: ${estimated_cost:.6f}"
            )

            # Cache the result
            self.cache.set(prompt, result)

            return result["eligible"], result["reason"], estimated_cost

        except EligibilityCriteriaError as e:
            logger.error(f"Eligibility criteria format error: {str(e)}")
            return False, f"Error: Invalid eligibility criteria format - {str(e)}", 0.0
        except Exception as e:
            logger.error(
                f"GPTTrialFilter.evaluate_trial: Error evaluating trial {trial.identification.nct_id}: {str(e)}"
            )
            return False, f"Error during evaluation: {str(e)}", 0.0


def load_json_file(file_path: str) -> List[dict]:
    """Load and parse a JSON file."""
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"load_json_file: File '{file_path}' not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        logger.error(f"load_json_file: File '{file_path}' is not a valid JSON file.")
        sys.exit(1)


def save_json_file(data: List[Dict], output_path: str):
    """Save filtered trials to a JSON file."""
    try:
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"save_json_file: Results saved to {output_path}")
    except Exception as e:
        logger.error(f"save_json_file: Error saving results: {str(e)}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Filter clinical trials using GPT-4 based on custom conditions"
    )
    parser.add_argument(
        "json_file", help="Path to the JSON file containing clinical trials data"
    )
    parser.add_argument(
        "conditions",
        nargs="+",
        help="One or more conditions to filter trials (e.g., 'doesn\\'t have a measurable site')",
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
    parser.add_argument(
        "--cache-size",
        type=int,
        default=1000,
        help="Maximum number of cached responses to keep",
    )
    # Add new arguments from parse_trials.py
    parser.add_argument(
        "--recruiting",
        action="store_true",
        help="Filter for only recruiting trials",
    )
    parser.add_argument(
        "--phase",
        help="Filter for trials of a specific phase (accepts both Arabic (1-4) and Roman numerals (I-IV))",
    )

    args = parser.parse_args()

    # Get API key from arguments or environment
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error(
            "main: OpenAI API key must be provided via --api-key or OPENAI_API_KEY environment variable"
        )
        sys.exit(1)

    # Initialize GPT filter with cache size
    gpt_filter = GPTTrialFilter(api_key, cache_size=args.cache_size)

    # Load and parse trials
    json_data = load_json_file(args.json_file)
    trials_parser = ClinicalTrialsParser(json_data)
    trials = trials_parser.trials

    # Apply recruiting filter if specified
    if args.recruiting:
        trials = trials_parser.get_recruiting_trials()
        logger.info(f"main: Filtered for recruiting trials: {len(trials)} found")

    # Apply phase filter if specified
    if args.phase:
        phase_filtered = trials_parser.get_trials_by_phase(int(args.phase))
        trials = [t for t in trials if t in phase_filtered]
        logger.info(
            f"main: Filtered for phase {args.phase} trials: {len(trials)} found"
        )

    # Filter trials using GPT
    filtered_trials = []
    total_trials = len(trials)

    logger.info(f"main: Processing {total_trials} trials...")
    logger.info(f"main: Evaluating conditions: {args.conditions}")

    total_cost = 0.0
    for i, trial in enumerate(trials, 1):
        logger.info(
            f"main: Processing trial {i}/{total_trials}: {trial.identification.nct_id}"
        )
        eligible, reason, cost = gpt_filter.evaluate_trial(trial, args.conditions)
        total_cost += cost
        logger.info(f"main: total cost: ${total_cost:.6f}")

        if eligible:
            logger.info(
                f"main: eligible: {eligible}, title: {trial.identification.brief_title},\n reason: {reason}"
            )
            filtered_trials.append(trial.to_dict())
        else:
            logger.info(
                f"main: eligible: {eligible}, title: {trial.identification.brief_title},\n reason: {reason}"
            )
        logger.debug(f"main: trial: {json.dumps(trial.to_dict(), indent=2)}")

    # Save results
    save_json_file(filtered_trials, args.output)
    logger.info(
        f"main: Found {len(filtered_trials)} matching trials out of {total_trials}"
    )


if __name__ == "__main__":
    main()
