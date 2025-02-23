#!/usr/bin/env python3
import argparse
import datetime
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add parent directory to Python path to import base module
sys.path.append(str(Path(__file__).parent.parent))
from base.clinical_trial import ClinicalTrial, ClinicalTrialsParser
from base.gpt_client import GPTClient

# Configure logging
log_file = f"filter_trials_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class EligibilityCriteriaError(Exception):
    """Custom exception for eligibility criteria format errors."""

    pass


@dataclass
class CriterionEvaluation:
    """Represents the evaluation result of a clinical trial criterion."""

    criterion: str
    reason: str
    eligibility: float


@dataclass
class TrialFailureReason:
    """Represents why a trial was deemed ineligible."""

    type: str  # "title" or "inclusion_criterion"
    message: str  # General failure message for title failures
    # Fields specific to inclusion criterion failures
    failed_condition: Optional[str] = None
    failed_criterion: Optional[str] = None
    failure_details: Optional[str] = None


class GPTTrialFilter:
    def __init__(self, api_key: str, cache_size: int = 100000):
        self.gpt_client = GPTClient(
            api_key=api_key,
            model="gpt-4o-mini",
            cache_size=cache_size,
            temperature=0.1,
            max_retries=3,
        )

    def _call_gpt(
        self,
        prompt: str,
        system_role: str,
        temperature: float = 0.1,
        refresh_cache: bool = False,
    ) -> Tuple[str, float]:
        """Common method for making GPT API calls."""
        return self.gpt_client.call_gpt(
            prompt,
            system_role,
            temperature=temperature,
            refresh_cache=refresh_cache,
            response_format={"type": "json_object"},
        )

    def _call_gpt_with_retry(
        self, prompt: str, system_role: str, max_retries: int = 3
    ) -> Tuple[str, float]:
        return self.gpt_client.call_with_retry(
            prompt,
            system_role,
            response_format={"type": "json_object"},
            validate_json=True,
        )

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
        """
        Validate the GPT response has required fields and valid values.

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
        return f"""You are evaluating a clinical trial inclusion criterion against one of the patient's conditions.

Study Title:
{title}

Inclusion Criterion:
{criterion}

Patient Condition to Evaluate:
{condition}

Please determine if this inclusion criterion aligns with this specific condition provided, considering the context from the study title.
Focus only on evaluating this single condition, even though the patient may have other conditions.
If this condition does not provide information related to the criterion, consider it as fully compatible (probability 1.0).
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
    ) -> Tuple[float, str, float]:
        """
        Evaluate if a trial title indicates suitability for given conditions.

        Args:
            trial: The clinical trial to evaluate
            conditions: Patient condition(s) to check against, either as a single string or list of strings

        Returns:
            Tuple of:
                float: Probability of trial suitability (0.0-1.0)
                str: Explanation of the evaluation
                float: Cost of the GPT API call
        """
        conditions_list = conditions if isinstance(conditions, list) else [conditions]
        conditions_text = "\n".join(f"- {condition}" for condition in conditions_list)

        prompt = f"""You are filtering clinical trials based on patient conditions and trial title.

Trial Details:
- Title: {trial.identification.brief_title}

Patient Conditions to Evaluate:
{conditions_text}

Please determine if the trial is potentially suitable for the patient conditions.

Return a JSON object containing:
- "reason": An explanation of why the title is or is not suitable
- "suitability_probability": A float value between 0.0 and 1.0 representing how suitable the trial is:
  - 0.0: Completely unsuitable
  - 0.5: Uncertain
  - 1.0: Completely suitable

Example response:
{{"reason": "[specific reasons]", "suitability_probability": 0.8}}"""

        response_content, cost = self._call_gpt(
            prompt,
            "You are a clinical trial analyst focused on evaluating titles.",
            temperature=0.1,
        )
        result = self._parse_gpt_response(response_content)
        return result["suitability_probability"], result["reason"], cost

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

        try:
            response_content, _ = self._call_gpt(
                prompt,
                "You are a clinical trial analyst specializing in logical structure analysis.",
                temperature=0.0,
            )
            result = self._parse_gpt_response(response_content)
            return result.get("is_or_criterion", False)
        except json.JSONDecodeError:
            # Retry with cache refresh on parse error
            response_content, _ = self._call_gpt(
                prompt,
                "You are a clinical trial analyst specializing in logical structure analysis.",
                temperature=0.0,
                refresh_cache=True,
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
    ) -> Tuple[float, Dict[str, CriterionEvaluation], float]:
        """
        Helper method to evaluate a single branch of an inclusion criterion.

        Args:
            branch: A single branch of an inclusion criterion to evaluate
            conditions: List of medical conditions to check against the branch
            trial_title: Title of the clinical trial for context

        Returns:
            Tuple containing:
                - float: Probability of eligibility (product of all condition probabilities)
                - Dict[str,CriterionEvaluation]: Mapping of conditions to their
                  evaluation results, including reasons
                - float: Cost of the GPT API call
        """
        branch_prob = 1.0
        branch_condition_evaluations = {}
        cost_sum = 0.0

        for condition in conditions:
            probability, reason, cost = self.evaluate_inclusion_criterion(
                branch, condition, trial_title
            )
            cost_sum += cost
            logger.info(
                f"GPTTrialFilter._evaluate_branch: Branch evaluation:\n"
                + json.dumps(
                    {
                        "branch": branch,
                        "condition": condition,
                        "probability": probability,
                        "cost": cost,
                    }
                )
            )
            branch_prob *= probability
            branch_condition_evaluations[condition] = CriterionEvaluation(
                criterion=branch,
                reason=reason,
                eligibility=probability,
            )

            # Early exit if any condition results in zero probability
            if branch_prob <= 0.0:
                break

        return branch_prob, branch_condition_evaluations, cost_sum

    def process_branches(
        self, branches: List[str], conditions: List[str], trial_title: str
    ) -> Tuple[float, Dict[str, List[CriterionEvaluation]], float]:
        """
        Process multiple branches and return the best probability and results.

        Args:
            branches: List of branch criteria to evaluate
            conditions: List of conditions to check against
            trial_title: Title of the clinical trial

        Returns:
            Tuple of (
                max_probability: float,
                condition_results: Dict[str, List[CriterionEvaluation]],
                total_cost: float
            )
        """
        branch_max_prob = 0.0
        branch_cost_sum = 0.0
        branch_results = {condition: [] for condition in conditions}

        for branch in branches:
            branch_prob: float
            branch_condition_evaluations: Dict[
                str, CriterionEvaluation
            ]  # Maps condition (str) to evaluation found for that condition
            branch_prob, branch_condition_evaluations, branch_cost = (
                self._evaluate_branch(branch, conditions, trial_title)
            )
            branch_cost_sum += branch_cost
            branch_max_prob = max(branch_max_prob, branch_prob)

            # Record which conditions met this branch
            for condition in conditions:
                # Get the individual condition's probability for this branch
                if condition in branch_condition_evaluations:
                    condition_evaluation: CriterionEvaluation = (
                        branch_condition_evaluations[condition]
                    )
                    branch_results[condition].append(
                        CriterionEvaluation(
                            criterion=branch,
                            reason=condition_evaluation.reason,
                            eligibility=condition_evaluation.eligibility,
                        )
                    )

            # Early exit if we found a fully compatible branch
            if branch_max_prob >= 1.0:
                logger.info(
                    f"Found fully compatible branch\n{json.dumps({'branch_prob': branch_max_prob, 'early_exit': True}, indent=2)}"
                )
                break

        return branch_max_prob, branch_results, branch_cost_sum

    def _get_or_criterion_failure_reason(
        self, branch_results: Dict[str, List[CriterionEvaluation]], criterion: str
    ) -> Tuple[str, str, str]:
        """
        Find and format the failure reason when all branches of an OR criterion fail.

        Args:
            branch_results: Dictionary mapping conditions to their branch evaluations
            criterion: The original OR criterion being evaluated

        Returns:
            Tuple of (failed_condition, criterion, detailed_reason)
        """
        # Find the first condition that failed all branches
        failed_condition = None
        failed_branch_evaluations = []
        for condition, branch_evaluations in branch_results.items():
            # Check if all branches failed for this condition
            if all(evaluation.eligibility <= 0.0 for evaluation in branch_evaluations):
                # Found a condition that failed all branches
                failed_condition = condition
                failed_branch_evaluations = branch_evaluations
                break

        # Format the detailed failure reason
        branch_reasons = [
            f"Branch {i+1}: {evaluation.reason}"
            for i, evaluation in enumerate(failed_branch_evaluations)
        ]
        detailed_reason = "Failed all OR branches:\n" + "\n".join(branch_reasons)

        logger.info(
            f"Condition failed all branches:\n{json.dumps({'condition': failed_condition, 'criterion': criterion, 'detailed_reason': detailed_reason}, indent=2)}"
        )

        return (failed_condition, criterion, detailed_reason)

    def evaluate_inclusion_criteria(
        self, inclusion_criteria: List[str], conditions: List[str], trial_title: str
    ) -> Tuple[float, Optional[Tuple[str, str, str]], float]:
        """
        Evaluate a list of inclusion criteria against given conditions.

        Args:
            inclusion_criteria: List of inclusion criteria to evaluate
            conditions: List of conditions to check against
            trial_title: Title of the clinical trial

        Returns:
            Tuple of (
                overall_probability: float,
                failure_reason: Optional[Tuple[str, str, str]],  # (condition, criterion, reason) if failed
                total_cost: float
            )
        """
        total_cost = 0.0
        overall_probability = 1.0
        failure_reason = None

        for criterion in inclusion_criteria:
            logger.info(
                f"GPTTrialFilter.evaluate_inclusion_criteria: Evaluating criterion: {criterion}"
            )

            # Handle OR criterion
            if self._is_or_criterion(criterion):
                logger.info(f"OR criterion detected: {criterion}")
                branches = self._split_or_branches(criterion)
                logger.info(
                    f"Split branches: {len(branches)}\n{json.dumps({'num_branches': len(branches), 'branches': branches}, indent=2)}"
                )

                branch_max_prob: float
                branch_results: Dict[str, List[CriterionEvaluation]]
                branch_cost: float
                # Process each branch of the OR criterion, evaluating against all conditions
                # Returns:
                # - branch_max_prob: Maximum probability across all branches (0-1)
                # - branch_results: Dict mapping conditions to list of CriterionEvaluation objects
                #   The evaluations are ordered to match the input branches, but may be incomplete
                #   if early exit occurs when a fully compatible branch is found
                # - branch_cost: Total API cost for evaluating all branches
                branch_max_prob, branch_results, branch_cost = self.process_branches(
                    branches, conditions, trial_title
                )
                total_cost += branch_cost

                # Check if any condition failed all branches
                if branch_max_prob <= 0.0:
                    failure_reason = self._get_or_criterion_failure_reason(
                        branch_results, criterion
                    )
                    overall_probability = 0.0
                    break

            # Handle non-OR criterion
            else:
                logger.info(f"Non-OR criterion detected: {criterion}")
                for condition in conditions:
                    probability, reason, cost = self.evaluate_inclusion_criterion(
                        criterion, condition, trial_title
                    )
                    overall_probability *= probability
                    total_cost += cost

                    if probability <= 0.0:
                        failure_reason = (condition, criterion, reason)
                        break

        if overall_probability <= 0.0 and failure_reason is None:
            raise RuntimeError(
                "Illegal state: overall_probability <= 0.0 but no failure reason was recorded"
            )
        return overall_probability, failure_reason, total_cost

    def evaluate_trial(
        self, trial: ClinicalTrial, conditions: list[str]
    ) -> Tuple[bool, float, Optional[TrialFailureReason]]:
        """
        Evaluate a trial's eligibility based on title and inclusion criteria.

        Args:
            trial: The clinical trial to evaluate
            conditions: List of conditions to check against

        Returns:
            (is_eligible, total_cost, failure_reason)

            - is_eligible: Whether the trial is eligible.
            - total_cost: Estimated GPT API cost for the calls.
            - failure_reason: If ineligible, a TrialFailureReason object describing why it failed.
                            If eligible, None.
        """
        # 1) Evaluate the title first
        title_probability, title_reason, title_cost = self.evaluate_title(
            trial, conditions
        )

        # If the trial fails at the title level
        if title_probability <= 0.0:
            failure = TrialFailureReason(
                type="title", message=f"Title check failed: {title_reason}"
            )
            logger.info(
                f"GPTTrialFilter.evaluate_trial: Title-based ineligibility "
                f"for {trial.identification.nct_id} | {failure.message}"
            )
            return False, title_cost, failure

        # 2) Extract and split the inclusion criteria
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
            failure = TrialFailureReason(
                type="format", message=f"Inclusion criteria format error: {str(e)}"
            )
            logger.error(
                f"evaluate_trial: Invalid criteria format "
                f"for trial {trial.identification.nct_id}: {failure.message}"
            )
            return False, title_cost, failure

        # 3) Evaluate the inclusion criteria
        inclusion_probability, inc_failure_reason, criteria_cost = (
            self.evaluate_inclusion_criteria(
                inclusion_criteria, conditions, trial.identification.brief_title
            )
        )

        total_cost = title_cost + criteria_cost
        overall_probability = title_probability * inclusion_probability

        # If it failed on an inclusion criterion
        if inc_failure_reason is not None:
            (cond_failed, crit_failed, reason) = inc_failure_reason
            failure = TrialFailureReason(
                type="inclusion_criterion",
                message="Failed inclusion criterion evaluation",
                failed_condition=cond_failed,
                failed_criterion=crit_failed,
                failure_details=reason,
            )
            logger.info(
                f"evaluate_trial: Trial {trial.identification.nct_id} failed "
                f"inclusion criteria evaluation"
            )
            return False, total_cost, failure

        # If overall probability is zero or near zero but no explicit inc_failure_reason
        if overall_probability <= 0.0:
            raise RuntimeError(
                "Illegal state: overall_probability <= 0.0 but no failure reason was recorded"
            )

        # Otherwise, the trial is eligible
        return True, total_cost, None


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


def save_excluded_trials(excluded_trials: List[Dict], output_path: str):
    """Save excluded trials with their failure reasons to a JSON file."""
    excluded_path = output_path.replace(".json", "_excluded.json")
    try:
        with open(excluded_path, "w") as f:
            json.dump(excluded_trials, f, indent=2)
        logger.info(f"save_excluded_trials: Excluded trials saved to {excluded_path}")
    except Exception as e:
        logger.error(f"save_excluded_trials: Error saving excluded trials: {str(e)}")
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
        default=100000,
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
    excluded_trials = []
    total_trials = len(trials)
    total_cost = 0.0
    eligible_count = 0

    logger.info(f"Processing {total_trials} trials with conditions: {args.conditions}")

    for i, trial in enumerate(trials, 1):
        logger.info(
            f"Processing trial {i}/{total_trials}: {trial.identification.nct_id}"
        )

        # Now we unpack three values: eligibility, cost, and failure reason
        is_eligible, cost, failure_reason = gpt_filter.evaluate_trial(
            trial, args.conditions
        )
        total_cost += cost

        if is_eligible:
            # If the trial is eligible, we keep the entire record
            trial_dict = trial.to_dict()
            filtered_trials.append(trial_dict)
            eligible_count += 1
        else:
            # If the trial is ineligible, store failure details
            excluded_info = {
                "nct_id": trial.identification.nct_id,
                "brief_title": trial.identification.brief_title,
                "eligibility_criteria": trial.eligibility.criteria,
                "failure_type": failure_reason.type,
                "failure_message": failure_reason.message,
            }

            # Add additional fields for inclusion criterion failures
            if failure_reason.type == "inclusion_criterion":
                excluded_info.update(
                    {
                        "failed_condition": failure_reason.failed_condition,
                        "failed_criterion": failure_reason.failed_criterion,
                        "failure_details": failure_reason.failure_details,
                    }
                )

            excluded_trials.append(excluded_info)

        logger.info(
            f"main: Eligible trials so far: {eligible_count}/{i} processed, total cost: ${total_cost:.2f}"
        )

    # Save the passing (filtered) trials
    save_json_file(filtered_trials, args.output)

    # Save the excluded trials
    excluded_path = args.output.replace(".json", "_excluded.json")
    try:
        with open(excluded_path, "w") as f:
            json.dump(excluded_trials, f, indent=2)
        logger.info(f"Excluded trials saved to {excluded_path}")
    except Exception as e:
        logger.error(f"Error saving excluded trials: {str(e)}")
        sys.exit(1)

    logger.info(
        f"main: Final results: {eligible_count}/{total_trials} trials were eligible"
    )
    logger.info(f"main: Filtered trials saved to {args.output}")
    logger.info(
        f"main: Excluded trials saved to {args.output.replace('.json', '_excluded.json')}"
    )
    logger.info(f"main: Total API cost: ${total_cost:.2f}")


if __name__ == "__main__":
    main()
