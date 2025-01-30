#!/usr/bin/env python3
import argparse
import datetime
import hashlib
import json
import logging
import os
import pickle
import re
import sys
import time
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

    def _get_cache_key(self, prompt: str, temperature: float) -> str:
        """Generate a unique cache key for a prompt and temperature."""
        cache_input = f"{prompt}_{temperature}"
        return hashlib.md5(cache_input.encode()).hexdigest()

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

    def get(self, prompt: str, temperature: float) -> dict | None:
        """Get cached result for a prompt and temperature."""
        cache_key = self._get_cache_key(prompt, temperature)
        if cache_key in self.cache_index:
            cache_file = self.cache_dir / f"{cache_key}.json"
            if cache_file.exists():
                with open(cache_file, "r") as f:
                    return json.load(f)
        return None

    def set(self, prompt: str, temperature: float, result: dict):
        """Cache result for a prompt and temperature."""
        cache_key = self._get_cache_key(prompt, temperature)

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

    def _call_gpt(
        self,
        prompt: str,
        system_role: str,
        temperature: float = 0.1,
        refresh_cache: bool = False,
    ) -> Tuple[str, float]:
        """Common method for making GPT API calls."""
        # Check cache first, unless refresh_cache is True
        if not refresh_cache:
            cached_result = self.cache.get(prompt, temperature)
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
                temperature=temperature,
            )

            result = response.choices[0].message.content
            # Move caching responsibility to the cache class
            self.cache.set(prompt, temperature, result)

            # Calculate cost
            input_string_length = len(prompt) + len(system_role)
            estimated_input_tokens = input_string_length * 0.25
            estimated_output_tokens = len(result) * 0.25
            cost = estimated_input_tokens * 0.15e-6 + estimated_output_tokens * 0.6e-6
            return result, cost

        except Exception as e:
            logger.error(f"GPTTrialFilter._call_gpt: Error in GPT evaluation: {str(e)}")
            raise

    def _call_gpt_with_retry(
        self, prompt: str, system_role: str, max_retries: int = 3
    ) -> Tuple[str, float]:
        for attempt in range(max_retries):
            try:
                # Pass refresh_cache=True on retry attempts
                response, cost = self._call_gpt(
                    prompt,
                    system_role,
                    temperature=0.1,
                    refresh_cache=(attempt > 0),  # Refresh cache on retry attempts
                )
                # Validate JSON before returning
                json.loads(response)
                return response, cost
            except json.JSONDecodeError:
                if attempt == max_retries - 1:
                    logger.warning(
                        f"GPTTrialFilter._call_gpt_with_retry: Failed after {max_retries} attempts"
                    )
                    raise
                time.sleep(2**attempt)  # Exponential backoff

    def _parse_gpt_response(self, response_content: str) -> dict:
        """Parse GPT response content into JSON, with error handling."""
        try:
            return json.loads(response_content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse GPT response: {response_content}")
            raise

    def _parse_gpt_response_with_fallback(self, response_content: str) -> dict:
        try:
            return json.loads(response_content)
        except json.JSONDecodeError:
            # Try to extract probability and reason using regex
            prob_match = re.search(
                r'suitability_probability":\s*(0?\.\d+)', response_content
            )
            reason_match = re.search(r'reason":\s*"([^"]+)"', response_content)

            if prob_match:
                return {
                    "suitability_probability": float(prob_match.group(1)),
                    "reason": (
                        reason_match.group(1)
                        if reason_match
                        else "Parsing error - no reason available"
                    ),
                }

            # If all parsing fails, return a conservative default
            logger.warning(
                f"Failed to parse GPT response, using default values: {response_content}"
            )
            return {
                "suitability_probability": 0.5,  # Conservative middle value
                "reason": "Failed to parse GPT response",
            }

    def _validate_gpt_response(self, parsed_response: dict) -> dict:
        """Validate the GPT response has required fields and valid values.

        Args:
            parsed_response: Dictionary containing the parsed GPT response

        Returns:
            The validated response dictionary

        Raises:
            ValueError: If response is missing required fields or has invalid values
        """
        required_fields = {"reason", "suitability_probability"}
        if not all(field in parsed_response for field in required_fields):
            raise ValueError(
                f"Missing required fields in GPT response: {parsed_response}"
            )

        prob = parsed_response["suitability_probability"]
        if not isinstance(prob, (int, float)) or not 0 <= prob <= 1:
            raise ValueError(f"Invalid probability value: {prob}")

        return parsed_response

    def _build_criterion_prompt(
        self, criterion: str, condition: str, title: str
    ) -> str:
        """Build the prompt for evaluating an inclusion criterion."""
        return f"""You are evaluating a clinical trial inclusion criterion against patient conditions.

Study Title:
{title}

Inclusion Criterion:
{criterion}

Patient Condition to Evaluate:
{condition}

Please determine if this inclusion criterion aligns with the condition provided, considering the context from the study title.
If the condition does not provide information related to the criterion, consider it as fully compatible (probability 1.0).
If the inclusion criterion represents a willingness to participate (e.g. "willing to undergo procedure X"), consider it as suitable.

IMPORTANT: You must respond with a complete, properly formatted JSON object containing exactly these fields:
{{"reason": "your explanation here", "suitability_probability": 0.0-1.0}}

Do not include any other text outside the JSON object.

Example response 1:
{{"reason": "[specific reasons]", "suitability_probability": 0.8}}

Example response 2:
{{"reason": "The patient condition does not mention any information related to the inclusion criterion, so it is fully compatible.", "suitability_probability": 1.0}}"""

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
            prompt,
            "You are a clinical trial analyst focused on evaluating titles.",
            temperature=0.1,
        )
        result = self._parse_gpt_response(response_content)
        return result["answer"], result["reason"], cost

    def evaluate_inclusion_criterion(
        self, criterion: str, condition: str, title: str
    ) -> Tuple[float, str, float]:
        try:
            # Try with retries first
            response_content, cost = self._call_gpt_with_retry(
                self._build_criterion_prompt(criterion, condition, title),
                "You are a clinical trial analyst focused on evaluating inclusion criteria.",
            )
            result = self._parse_gpt_response_with_fallback(response_content)
            validated_result = self._validate_gpt_response(result)
            logger.info(
                f"GPTTrialFilter.evaluate_inclusion_criterion: Evaluated criterion: {criterion} for condition: {condition} with title: {title}"
                + json.dumps(
                    {
                        "criterion": criterion,
                        "condition": condition,
                        "eligibility": validated_result["suitability_probability"],
                        "reason": validated_result["reason"],
                        "cost": cost,
                    },
                    indent=2,
                )
            )
            return (
                validated_result["suitability_probability"],
                validated_result["reason"],
                cost,
            )
        except Exception as e:
            logger.error(f"Failed to evaluate criterion: {str(e)}")
            # Last resort fallback
            return 0.5, f"Evaluation failed: {str(e)}", 0.0

    def split_inclusion_criteria(self, criteria: str) -> List[str]:
        """Split the inclusion criteria into individual statements using GPT."""
        prompt = f"""You are analyzing clinical trial inclusion criteria text.

Inclusion Criteria Text:
{criteria}

Split this text into individual inclusion criterion statements following these rules:

1. **Disease-Specific Consolidation**:
   - Combine ALL disease type requirements into a single criterion using "OR" logic, even if:
     - They appear as separate bullet points
     - They have different biomarker/therapy requirements
     - They have mixed prior therapy rules (naïve vs. treated)
   - Structure as: "Patients must have one of the following: (a) [Disease A] with [requirements]; OR (b) [Disease B] with [different requirements]..."
   - Preserve ALL disease-specific details (biomarker thresholds, prior therapy sequences, progression requirements)

2. **Multi-Level Requirements**:
   - For complex disease requirements, use nested logic:
     '''
     Patients must have:
     a) [Disease 1] with:
        - [Requirement 1]
        - [Requirement 2]
     OR
     b) [Disease 2] with:
        - [Requirement 3]
        - [Requirement 4]
     '''

3. **Separate Non-Disease Criteria**:
   - Keep non-disease requirements (organ function, performance status) as distinct criteria
   - Each requirement will be treated as a separate criterion connected by AND logic

4. **Logical Structure**:
   - Use clear alphanumeric labeling (a), b), c)) for complex criteria
   - Employ AND/OR relationships within disease subgroups as needed
   - Maintain all temporal sequence requirements (e.g., "prior X then Y")
   - All separate criteria are implicitly connected with AND logic

Return ONLY valid JSON with criteria list.

Example response:
{{"criteria": [
    "Patient must have one of the following: (a) Triple-negative breast cancer with PD-L1 CPS ≥10 and progression after prior therapy; OR (b) Advanced endometrial cancer with MSI-H/dMMR and platinum-therapy failure",
    "Adequate organ function per laboratory parameters"
]}}"""

        response_content, cost = self._call_gpt(
            prompt,
            "You are a clinical trial analyst focused on parsing inclusion criteria.",
            temperature=0.0,
        )
        try:
            result = self._parse_gpt_response(response_content)
        except json.JSONDecodeError as e:
            logger.error(
                f"GPTTrialFilter.split_inclusion_criteria: Failed to parse GPT response. Response was:\n{response_content}\nPrompt was:\n{prompt}"
            )
            raise
        logger.info(
            f"GPTTrialFilter.split_inclusion_criteria: original criteria: {criteria}"
        )
        logger.info(f"GPTTrialFilter.split_inclusion_criteria: result: {result}")
        return result["criteria"]

    def _extract_inclusion_criteria(self, criteria: str) -> str:
        """Extract and validate inclusion criteria from the full criteria text.

        Args:
            criteria: Full criteria text containing both inclusion and exclusion criteria

        Returns:
            str: The inclusion criteria section

        Raises:
            EligibilityCriteriaError: If the criteria format is invalid
        """
        if not criteria or "Inclusion Criteria" not in criteria:
            logger.warning(
                f"GPTTrialFilter._extract_inclusion_criteria: Missing 'Inclusion Criteria' section in criteria: {criteria}"
            )
            return criteria

        # Split by "Inclusion Criteria" and take everything after it
        inclusion_text = criteria.split("Inclusion Criteria")[1].strip()

        # If there's an "Exclusion Criteria" section, only take the text before it
        if "Exclusion Criteria" in inclusion_text:
            inclusion_text = inclusion_text.split("Exclusion Criteria")[0].strip()
        else:
            logger.warning(
                f"GPTTrialFilter._extract_inclusion_criteria: Missing 'Exclusion Criteria' section in criteria text: {criteria}"
            )

        return inclusion_text

    def _is_or_criterion(self, criterion: str) -> bool:
        """Check if a criterion contains top-level OR logic using GPT."""
        prompt = f"""Analyze this clinical trial inclusion criterion for top-level OR logic:

Criterion: {criterion}

Does this criterion contain multiple alternative options connected by OR at the top level (not nested within subgroups)? Respond ONLY with JSON:
{{"is_or_criterion": true/false}}"""

        response_content, _ = self._call_gpt(
            prompt,
            "You are a clinical trial analyst specializing in logical structure analysis.",
            temperature=0.0,
        )
        result = self._parse_gpt_response(response_content)
        return result.get("is_or_criterion", False)

    def _split_or_branches(self, criterion: str) -> List[str]:
        """Split a criterion with top-level OR logic into individual branches."""
        prompt = f"""Split this clinical trial inclusion criterion into separate OR branches:

Original Criterion:
{criterion}

Rules:
1. Split only at TOP-LEVEL OR connections
2. Maintain nested AND/OR structures within branches
3. Preserve all original requirements in each branch

Return ONLY JSON with a "branches" list containing the split criteria:
{{"branches": ["branch 1 text", "branch 2 text", ...]}}"""

        response_content, _ = self._call_gpt(
            prompt,
            "You are a clinical trial analyst specializing in logical structure analysis.",
            temperature=0.0,
        )
        result = self._parse_gpt_response(response_content)
        return result.get("branches", [criterion])

    def _evaluate_branch(
        self, branch: str, conditions: List[str], trial_title: str
    ) -> Tuple[float, Dict]:
        """Helper method to evaluate a single branch."""
        branch_prob = 1.0
        branch_violations_current = {condition: [] for condition in conditions}

        for condition in conditions:
            probability, reason, cost = self.evaluate_inclusion_criterion(
                branch, condition, trial_title
            )
            logger.info(
                f"Branch evaluation: {branch}, Condition: {condition}, Probability: {probability}, Cost: {cost}"
            )
            branch_prob *= probability
            if probability <= 0.0:
                branch_violations_current[condition].append(
                    {
                        "branch": branch,
                        "reason": reason,
                        "eligibility": probability,
                        "cost": cost,
                    }
                )

            # Early exit if any condition results in zero probability
            if branch_prob <= 0.0:
                break

        return branch_prob, branch_violations_current

    def evaluate_inclusion_criteria(
        self, inclusion_criteria: List[str], conditions: List[str], trial_title: str
    ) -> Tuple[float, float]:
        """Evaluate a list of inclusion criteria against given conditions."""
        total_cost = 0.0
        overall_probability = 1.0

        for criterion in inclusion_criteria:
            logger.info(
                f"GPTTrialFilter.evaluate_inclusion_criteria: Evaluating criterion: {criterion}"
            )
            criterion_probabilities = []
            criterion_cost = 0.0

            # Handle OR criterion
            if self._is_or_criterion(criterion):
                logger.info(f"OR criterion detected: {criterion}")
                branches = self._split_or_branches(criterion)
                logger.info(
                    f"Split branches: {len(branches)}\n{json.dumps({'num_branches': len(branches), 'branches': branches}, indent=2)}"
                )

                branch_max_prob = 0.0
                branch_cost = 0.0
                branch_results = {condition: [] for condition in conditions}

                for branch in branches:
                    branch_prob, branch_violations_current = self._evaluate_branch(
                        branch, conditions, trial_title
                    )
                    # Sum costs from violations
                    branch_cost += sum(
                        violation["cost"]
                        for condition_violations in branch_violations_current.values()
                        for violation in condition_violations
                    )
                    branch_max_prob = max(branch_max_prob, branch_prob)

                    # Record which conditions met this branch
                    for condition in conditions:
                        # Get the individual condition's probability for this branch
                        condition_violations = branch_violations_current[condition]
                        condition_prob = (
                            1.0
                            if not condition_violations
                            else min(v["eligibility"] for v in condition_violations)
                        )
                        if (
                            condition_prob > 0
                        ):  # Only record if condition met the branch
                            branch_results[condition].append(
                                {"branch": branch, "eligibility": condition_prob}
                            )

                # Log results for each condition
                for condition, met_branches in branch_results.items():
                    if met_branches:  # Condition met at least one branch
                        logger.info(
                            f"Condition met branches in OR criterion: {condition}\n{json.dumps({'condition': condition, 'met_branches': met_branches}, indent=2)}"
                        )
                    else:  # Condition failed all branches
                        logger.info(
                            f"Condition failed all branches: {condition}\n{json.dumps({'condition': condition, 'violations': branch_violations_current[condition], 'criterion': criterion}, indent=2)}"
                        )
                        branch_max_prob = 0.0  # Set probability to 0
                        break  # Exit the branch evaluation loop since this condition can't be satisfied

                total_cost += branch_cost
                criterion_probability = branch_max_prob

                # Early exit if any branch is fully compatible
                if branch_max_prob >= 1.0:
                    logger.info(
                        f"Found fully compatible branch\n{json.dumps({'branch_prob': branch_max_prob, 'early_exit': True}, indent=2)}"
                    )
                    break

            # Handle non-OR criterion
            else:
                logger.info(f"Non-OR criterion detected: {criterion}")
                for condition in conditions:
                    probability, reason, cost = self.evaluate_inclusion_criterion(
                        criterion, condition, trial_title
                    )
                    if abs(probability) < 1e-6:
                        criterion_probabilities = [0.0]
                        criterion_cost += cost
                        break
                    criterion_probabilities.append(probability)
                    criterion_cost += cost

                criterion_probability = 1.0
                for prob in criterion_probabilities:
                    criterion_probability *= prob

                total_cost += criterion_cost

            logger.info(
                f"Criterion evaluation result\n{json.dumps({'criterion': criterion, 'eligibility': criterion_probability}, indent=2)}"
            )

            # Update overall probability
            overall_probability *= criterion_probability

            # Break early if probability is very close to zero
            if abs(overall_probability) < 1e-6:
                overall_probability = 0.0
                break

        return overall_probability, total_cost

    def evaluate_trial(
        self, trial: ClinicalTrial, conditions: list[str]
    ) -> Tuple[bool, float]:
        """Evaluate a trial's eligibility based on title and inclusion criteria.

        Args:
            trial: The clinical trial to evaluate
            conditions: List of conditions to check against

        Returns:
            Tuple of (is_eligible: bool, total_cost: float)
        """
        # Evaluate the title first
        title_eligible, title_reason, title_cost = self.evaluate_title(
            trial, conditions
        )
        title_suitability = 0.0 if title_eligible == "unsuitable" else 1.0

        if abs(title_suitability) < 1e-6:
            logger.info(
                json.dumps(
                    {
                        "message": "evaluate_trial: Trial is ineligible based on title",
                        "trial_id": trial.identification.nct_id,
                        "title": trial.identification.brief_title,
                        "reason": title_reason,
                        "eligibility": 0.0,
                    },
                    indent=2,
                )
            )
            return False, title_cost

        logger.info(
            json.dumps(
                {
                    "message": "evaluate_trial: Trial passed title evaluation",
                    "trial_id": trial.identification.nct_id,
                    "title": trial.identification.brief_title,
                    "reason": title_reason,
                    "eligibility": 1.0,
                },
                indent=2,
            )
        )

        # Extract and validate inclusion criteria
        try:
            inclusion_text = self._extract_inclusion_criteria(
                trial.eligibility.criteria
            )
            inclusion_criteria = self.split_inclusion_criteria(inclusion_text)
            logger.info(
                json.dumps(
                    {
                        "message": "evaluate_trial: Split inclusion criteria",
                        "trial_id": trial.identification.nct_id,
                        "criteria_count": len(inclusion_criteria),
                        "criteria": inclusion_criteria,
                    },
                    indent=2,
                )
            )

        except EligibilityCriteriaError as e:
            logger.error(
                f"evaluate_trial: Invalid criteria format for trial {trial.identification.nct_id}: {str(e)}"
            )
            return False, title_cost

        # Evaluate inclusion criteria
        overall_probability, criteria_cost = self.evaluate_inclusion_criteria(
            inclusion_criteria, conditions, trial.identification.brief_title
        )

        total_cost = title_cost + criteria_cost
        is_eligible = overall_probability > 0.0

        logger.info(
            f"evaluate_trial: Final eligibility: {overall_probability:.4f}, title: {trial.identification.brief_title}, url: {trial.identification.url}"
        )

        return is_eligible, total_cost


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

    # Apply filters
    if args.exclude_study_type:
        trials = trials_parser.get_trials_excluding_study_type(args.exclude_study_type)
        logger.info(
            f"main: Excluded study type '{args.exclude_study_type}': {len(trials)} trials remaining"
        )

    if args.recruiting:
        trials = trials_parser.get_recruiting_trials()
        logger.info(f"main: Filtered for recruiting trials: {len(trials)} found")

    if args.phase:
        phase_filtered = trials_parser.get_trials_by_phase(int(args.phase))
        trials = [t for t in trials if t in phase_filtered]
        logger.info(
            f"main: Filtered for phase {args.phase} trials: {len(trials)} found"
        )

    filtered_trials = []
    total_trials = len(trials)
    total_cost = 0.0
    eligible_count = 0

    logger.info(f"Processing {total_trials} trials with conditions: {args.conditions}")

    for i, trial in enumerate(trials, 1):
        logger.info(
            f"Processing trial {i}/{total_trials}: {trial.identification.nct_id}"
        )

        is_eligible, cost = gpt_filter.evaluate_trial(trial, args.conditions)
        total_cost += cost

        if is_eligible:
            filtered_trials.append(trial.to_dict())
            eligible_count += 1

        logger.info(f"main: Eligible trials so far: {eligible_count}/{i} processed")

    # Save results
    save_json_file(filtered_trials, args.output)
    logger.info(
        f"main: Final results: {eligible_count}/{total_trials} trials were eligible"
    )
    logger.info(f"main: Filtered trials saved to {args.output}")
    logger.info(f"main: Total API cost: ${total_cost:.2f}")


if __name__ == "__main__":
    main()
