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
    def __init__(self, cache_dir: str = ".cache", max_size: int = 10000):
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
    def __init__(self, api_key: str, cache_size: int = 10000):
        self.client = OpenAI(api_key=api_key)
        self.cache = PromptCache(max_size=cache_size)

    def _call_gpt(self, prompt: str, system_role: str) -> Tuple[str, float]:
        """Common method for making GPT API calls."""
        # Check cache first
        cached_result = self.cache.get(prompt)
        if cached_result is not None:
            logger.debug("GPTTrialFilter._call_gpt: Using cached result")
            return cached_result, 0.0

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_role},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
            )

            result = response.choices[0].message.content
            # Move caching responsibility to the cache class
            self.cache.set(prompt, result)

            # Calculate cost
            input_string_length = len(prompt) + len(system_role)
            estimated_input_tokens = input_string_length * 0.25
            estimated_output_tokens = len(result) * 0.25
            cost = estimated_input_tokens * 0.15e-6 + estimated_output_tokens * 0.6e-6
            return result, cost

        except Exception as e:
            logger.error(f"GPTTrialFilter._call_gpt: Error in GPT evaluation: {str(e)}")
            return False, f"Error during evaluation: {str(e)}"

    def _parse_gpt_response(self, response_content: str) -> dict:
        """Parse GPT response content into JSON, with error handling."""
        try:
            return json.loads(response_content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse GPT response: {response_content}")
            raise

    def evaluate_title(
        self, trial: ClinicalTrial, conditions: str | list[str]
    ) -> Tuple[str, str, float]:
        conditions_list = conditions if isinstance(conditions, list) else [conditions]
        conditions_text = "\n".join(f"- {condition}" for condition in conditions_list)

        prompt = f"""You are filtering a clinical trials based on patient conditions and trial title.

Trial Details:
- Title: {trial.identification.brief_title}

Patient Conditions to Evaluate:
{conditions_text}

Please determine if the trial is potentially suitable for the patient conditions.

Return a JSON object containing:
- "reason": An explanation of why the title is or is not suitable
- "answer": "suitable" if the trial is potentially suitable, "unsuitable" if it is not, "uncertain" if unsure

Example response:
{{"reason": "[specific reasons]", "answer": "suitable"}}"""

        response_content, cost = self._call_gpt(
            prompt, "You are a clinical trial analyst focused on evaluating titles."
        )
        result = self._parse_gpt_response(response_content)
        return result["answer"], result["reason"], cost

    def evaluate_inclusion_criterion(
        self, criterion: str, condition: str
    ) -> Tuple[str, str, float]:
        prompt = f"""You are evaluating a clinical trial inclusion criterion against patient conditions.

Inclusion Criterion:
{criterion}

Patient Condition to Evaluate:
{condition}

Please determine if this inclusion criterion aligns with the condition provided.

Return a JSON object containing:
- "reason": An explanation of why the inclusion criterion is or is not suitable
- "answer": "suitable" if it meets the condition or is unrelated, "unsuitable" if it does not meet the condition, "uncertain" if unsure

Example response:
{{"reason": "[specific reasons]", "answer": "suitable"}}"""

        response_content, cost = self._call_gpt(
            prompt,
            "You are a clinical trial analyst focused on evaluating inclusion criteria.",
        )
        result = self._parse_gpt_response(response_content)
        return result["answer"], result["reason"], cost

    def split_inclusion_criteria(self, criteria: str) -> List[str]:
        """Split the inclusion criteria into individual statements."""
        return [
            criterion.strip()
            for criterion in criteria.split("\n")
            if criterion.strip() and "inclusion criteria" not in criterion.lower()
        ]

    def evaluate_trial(
        self, trial: ClinicalTrial, conditions: str | list[str]
    ) -> Tuple[bool, float]:
        """Evaluate a single trial against the given conditions using GPT."""
        # Evaluate the title first
        title_eligible, title_reason, total_cost = self.evaluate_title(
            trial, conditions
        )
        logger.info(
            f"evaluate_trial: title: {trial.identification.brief_title} eligible: {title_eligible}\n reason: {title_reason}"
        )
        if title_eligible == "unsuitable":
            return False, total_cost

        # Split and evaluate inclusion criteria
        inclusion_criteria = self.split_inclusion_criteria(trial.eligibility.criteria)

        all_criteria_eligible = True
        for criterion in inclusion_criteria:
            criterion_eligible, criterion_reason, cost = (
                self.evaluate_inclusion_criterion(criterion, conditions)
            )
            logger.info(
                f"evaluate_trial: criterion: {criterion} eligible: {criterion_eligible}\n reason: {criterion_reason}"
            )
            total_cost += cost
            if criterion_eligible == "unsuitable":
                all_criteria_eligible = False
                break

        return all_criteria_eligible, total_cost


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
        default=10000,
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
    parser.add_argument(
        "--exclude-study-type",
        help="Exclude trials of a specific study type (e.g., 'Observational')",
    )

    args = parser.parse_args()

    # Get API key from arguments or environment
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error(
            "OpenAI API key must be provided via --api-key or OPENAI_API_KEY environment variable"
        )
        sys.exit(1)

    # Initialize GPT filter and load trials
    gpt_filter = GPTTrialFilter(api_key, cache_size=args.cache_size)
    json_data = load_json_file(args.json_file)
    trials_parser = ClinicalTrialsParser(json_data)
    trials = trials_parser.trials

    # Apply study type exclusion if specified
    if args.exclude_study_type:
        trials = trials_parser.get_trials_excluding_study_type(args.exclude_study_type)
        logger.info(
            f"main: Excluded study type '{args.exclude_study_type}': {len(trials)} trials remaining"
        )

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

    filtered_trials = []
    total_trials = len(trials)
    total_cost = 0.0

    logger.info(f"Processing {total_trials} trials with conditions: {args.conditions}")

    for i, trial in enumerate(trials, 1):
        logger.info(
            f"Processing trial {i}/{total_trials}: {trial.identification.nct_id}"
        )
        eligible, cost = gpt_filter.evaluate_trial(trial, args.conditions)
        total_cost += cost
        logger.info(f"Total cost: ${total_cost:.6f}")

        if eligible:
            filtered_trials.append(trial.to_dict())
        logger.info(
            f"main: eligible: {eligible}, title: {trial.identification.brief_title}, url: {trial.identification.url}"
        )

    # Save results
    save_json_file(filtered_trials, args.output)
    logger.info(f"Found {len(filtered_trials)} matching trials out of {total_trials}")


if __name__ == "__main__":
    main()
