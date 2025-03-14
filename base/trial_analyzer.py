import json
import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from base.disease_expert import extract_disease_from_record
from base.drug_analyzer import analyze_drug_effectiveness
from base.gpt_client import GPTClient
from base.utils import parse_json_response

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from base.clinical_trial import ClinicalTrial
    from base.perplexity import PerplexityClient

CLINICAL_TRIAL_SYSTEM_PROMPT = (
    "<role>You are a clinical research expert with extensive experience in evaluating patient eligibility and treatment outcomes. "
    "Your expertise includes analyzing clinical trials, published research, and making evidence-based recommendations for patient care.</role>\n\n"
    "<task>Your task is to assess if a clinical trial would be beneficial for a patient. "
    "You will analyze published research and clinical evidence on similar drugs and treatments to inform your recommendation. "
    "You must provide your assessment in a JSON format with a recommendation level and detailed reasoning.</task>\n\n"
    "<recommendation_levels>The possible recommendation levels are:\n\n"
    "- Strongly Recommended\n"
    "- Recommended\n"
    "- Neutral\n"
    "- Not Recommended</recommendation_levels>\n\n"
    "<output_format>You must respond with a JSON object containing:\n"
    "- reason: A detailed explanation of your recommendation based on the evidence provided\n"
    "- recommendation: One of the recommendation levels listed above</output_format>\n\n"
)


class RecommendationLevel(Enum):
    STRONGLY_RECOMMENDED = "Strongly Recommended"
    RECOMMENDED = "Recommended"
    NEUTRAL = "Neutral"
    NOT_RECOMMENDED = "Not Recommended"

    @classmethod
    def values(cls) -> list[str]:
        return [level.value for level in cls]


def build_recommendation_prompt(
    clinical_record: str,
    trial_info: "ClinicalTrial",
    drug_analyses: dict[str, str] = None,
) -> str:
    """
    Constructs a prompt for an AI to evaluate patient clinical record against trial information
    and return a recommendation level based on potential benefit to the patient.

    Parameters:
    - clinical_record (str): The patient's clinical record to be evaluated
    - trial_info (ClinicalTrial): The clinical trial information to compare against
    - drug_analyses (dict[str, str]): Optional dictionary mapping drug names to their effectiveness analyses

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

    drug_analysis_str = ""
    if drug_analyses:
        drug_analysis_str = "\n\n<drug_analyses>\nDrug Effectiveness Analysis:\n"
        for drug, analysis in drug_analyses.items():
            drug_analysis_str += f"\n{drug}:\n{analysis}\n"
        drug_analysis_str += "</drug_analyses>"

    return (
        f'<clinical_record>\nClinical Record:\n"{clinical_record}"\n</clinical_record>\n\n'
        f'<trial_info>\nTrial Information:\n"{trial_info_str}"\n</trial_info>'
        f"{drug_analysis_str}\n\n"
        "<output_format>\nProvide your response as a JSON object with the following structure:\n"
        "{\n"
        '  "reason": "detailed explanation of your recommendation based on the clinical record, trial information, and drug analyses",\n'
        '  "recommendation": "one of: Strongly Recommended, Recommended, Neutral, Not Recommended"\n'
        "}\n</output_format>\n\n"
        "<output_request>\nBased on the clinical record, trial information, and drug effectiveness analysis (if available), "
        "evaluate if this trial would be beneficial for the patient. Consider both the patient's condition and the evidence "
        "regarding the trial's treatment approach.</output_request>"
    )


def parse_recommendation_response(response: str) -> tuple[RecommendationLevel, str]:
    """
    Parses the JSON response from the AI recommendation.

    Parameters:
    - response (str): The JSON string response from the AI

    Returns:
    - tuple[RecommendationLevel, str]: A tuple containing (recommendation enum, reason)

    Raises:
    - ValueError: If the response is not valid JSON or missing required fields
    """
    try:
        data = json.loads(response)
        recommendation_str = data.get("recommendation")
        reason = data.get("reason")

        if not recommendation_str or not reason:
            raise ValueError(
                "Response missing required 'recommendation' or 'reason' fields"
            )

        if recommendation_str not in RecommendationLevel.values():
            raise ValueError(f"Invalid recommendation level: {recommendation_str}")

        # Convert string to enum
        recommendation = RecommendationLevel(recommendation_str)
        return recommendation, reason

    except json.JSONDecodeError as e:
        logger.error(f"parse_recommendation_response: Invalid JSON response: {e}")
        logger.error(f"parse_recommendation_response: Response content: {response}")
        raise ValueError(f"Invalid JSON response: {e}. Response content: {response}")


def analyze_drugs_and_get_recommendation(
    clinical_record: str,
    trial: "ClinicalTrial",
    perplexity_client: "PerplexityClient",
    gpt_client: "GPTClient",
) -> tuple[RecommendationLevel, str, dict[str, str], float]:
    """
    Analyze drug effectiveness and generate AI recommendation.

    Returns:
        tuple containing:
            - Recommendation level (Strongly Recommended, Recommended, etc.)
            - Reason for the recommendation
            - Dictionary mapping drug names to their effectiveness analyses
            - Total cost of API calls
    """
    total_cost = 0.0
    drug_analyses: dict[str, str] = {}

    # Extract disease from clinical record
    disease, cost = extract_disease_from_record(clinical_record, gpt_client)
    total_cost += cost
    logger.info(f"analyze_drugs_and_get_recommendation: Extracted Disease: {disease}")
    logger.info(f"analyze_drugs_and_get_recommendation: Cost: ${cost:.6f}")

    novel_drugs, cost = trial.get_novel_drugs_from_title(gpt_client)
    logger.debug(f"analyze_drugs_and_get_recommendation: novel_drugs: {novel_drugs}")
    total_cost += cost

    # Analyze each drug's effectiveness
    if novel_drugs and disease:
        for drug in novel_drugs:
            logger.info(
                f"analyze_drugs_and_get_recommendation: Analyzing effectiveness of {drug} for {disease}"
            )
            analysis, citations, cost = analyze_drug_effectiveness(
                drug, disease, perplexity_client
            )
            total_cost += cost

            if analysis:
                drug_analyses[drug] = analysis
                logger.info(
                    f"analyze_drugs_and_get_recommendation: Drug Analysis: {analysis}"
                )
                if citations:
                    logger.info(
                        f"analyze_drugs_and_get_recommendation: Citations ({len(citations)}):"
                    )
                    for i, citation in enumerate(citations, 1):
                        logger.info(
                            f"analyze_drugs_and_get_recommendation: Citation {i}: {citation}"
                        )
                logger.info(f"analyze_drugs_and_get_recommendation: Cost: ${cost:.6f}")

    # Generate recommendation
    prompt = build_recommendation_prompt(clinical_record, trial, drug_analyses)
    logger.info(
        f"analyze_drugs_and_get_recommendation: Recommendation Prompt:\n{prompt}"
    )

    completion: str
    cost: float
    completion, cost = gpt_client.call_gpt(
        prompt=prompt,
        system_role=CLINICAL_TRIAL_SYSTEM_PROMPT,
        model="gpt-4o",
        temperature=0.2,
    )
    total_cost += cost

    if completion is None:
        raise RuntimeError("Failed to get AI analysis")

    # Validate JSON structure
    try:
        data = json.loads(completion)
        # Ensure required fields are present
        if "reason" not in data or "recommendation" not in data:
            raise ValueError(
                "JSON response missing 'reason' or 'recommendation' fields"
            )
    except json.JSONDecodeError:
        logger.warning("Invalid JSON structure detected, attempting to correct it.")
        # Use gpt-4o-mini to convert to valid JSON
        correction_prompt = (
            f"Please convert the following text into a valid JSON object. "
            f"Ensure no additional text or formatting is included:\n{completion}"
        )
        completion, correction_cost = gpt_client.call_gpt(
            prompt=correction_prompt,
            system_role="You are a JSON expert.",
            model="gpt-4o-mini",
            temperature=0.0,
        )
        total_cost += correction_cost
        try:
            json.loads(completion)  # Re-validate the corrected JSON
        except json.JSONDecodeError as e:
            logger.error(f"Failed to correct JSON: {e}")
            logger.error(f"Failed to convert to valid JSON: {completion}")
            raise ValueError("Failed to convert to valid JSON")

    recommendation: RecommendationLevel
    reason: str
    recommendation, reason = parse_recommendation_response(completion)
    logger.info(
        f"analyze_drugs_and_get_recommendation: Recommendation: {recommendation}"
    )
    logger.info(f"analyze_drugs_and_get_recommendation: Reason: {reason}")

    return recommendation, reason, drug_analyses, total_cost


def compare_trials(
    clinical_record: str,
    trial1: "ClinicalTrial",
    trial2: "ClinicalTrial",
    gpt_client: "GPTClient",
) -> tuple["ClinicalTrial", str, float]:
    """
    Compare two clinical trials and determine which is better for a target patient.

    Args:
        clinical_record (str): The patient's clinical record
        trial1 (ClinicalTrial): First trial to compare
        trial2 (ClinicalTrial): Second trial to compare
        gpt_client (GPTClient): Client for GPT-4 analysis

    Returns:
        tuple containing:
            - The better trial (ClinicalTrial)
            - Detailed explanation of the comparison
            - Total cost of API calls
    """
    total_cost = 0.0
    max_retries = 3

    # Build comparison prompt using existing drug analyses
    trial1_info = (
        f"NCT ID: {trial1.identification.nct_id}\n"
        f"Title: {trial1.identification.brief_title}\n"
        f"Recommendation Level: {trial1.recommendation_level}\n"
        f"Reason: {trial1.analysis_reason}\n"
        f"Drug Analyses: {json.dumps(trial1.drug_analysis, indent=2)}"
    )

    trial2_info = (
        f"NCT ID: {trial2.identification.nct_id}\n"
        f"Title: {trial2.identification.brief_title}\n"
        f"Recommendation Level: {trial2.recommendation_level}\n"
        f"Reason: {trial2.analysis_reason}\n"
        f"Drug Analyses: {json.dumps(trial2.drug_analysis, indent=2)}"
    )

    comparison_prompt = (
        f'<clinical_record>\nClinical Record:\n"{clinical_record}"\n</clinical_record>\n\n'
        f"<trial1_info>\n{trial1_info}\n</trial1_info>\n\n"
        f"<trial2_info>\n{trial2_info}\n</trial2_info>\n\n"
        "<output_format>\nProvide your response as a JSON object with the following structure:\n"
        "{\n"
        '  "reason": "detailed explanation of why this trial is better, considering both trials\' recommendations, drug analyses, and other relevant factors",\n'
        '  "better_trial": "nct_id of the better trial"\n'
        "}\n</output_format>\n\n"
        "<output_request>\nBased on the clinical record and both trials' analyses, determine which trial would be better for the patient. "
        "Consider the recommendation levels, drug effectiveness analyses, and any other relevant factors. "
        "Provide a detailed explanation of your decision.</output_request>"
    )

    for attempt in range(max_retries):
        try:
            # Get comparison from GPT-4
            completion, cost = gpt_client.call_gpt(
                prompt=comparison_prompt,
                system_role=CLINICAL_TRIAL_SYSTEM_PROMPT,
                model="gpt-4o",
                temperature=0.2,
                response_format={"type": "json_object"},
                refresh_cache=attempt > 0,  # Refresh cache if retried
            )
            total_cost += cost

            if completion is None:
                raise RuntimeError("Failed to get trial comparison")

            # Parse the comparison response
            try:
                data = json.loads(completion)
                reason = data.get("reason")
                better_trial_id = data.get("better_trial")

                if not better_trial_id or not reason:
                    logger.error(
                        f"Response missing required fields. Response content: {completion}"
                    )
                    if attempt < max_retries - 1:
                        logger.info(
                            f"Retrying comparison (attempt {attempt + 1}/{max_retries})"
                        )
                        continue
                    raise ValueError(
                        "Response missing required 'better_trial' or 'reason' fields"
                    )

                # Determine which trial is better
                better_trial = (
                    trial1
                    if better_trial_id == trial1.identification.nct_id
                    else trial2
                )

                logger.info(
                    f"compare_trials: Better trial: {better_trial.identification.nct_id}"
                )
                logger.info(f"compare_trials: Reason: {reason}")
                logger.info(f"compare_trials: Total cost: ${total_cost:.6f}")

                return better_trial, reason, total_cost

            except json.JSONDecodeError:
                logger.error(
                    f"Invalid JSON response on attempt {attempt + 1}/{max_retries}: {completion}"
                )
                if attempt < max_retries - 1:
                    logger.info(
                        f"Retrying comparison (attempt {attempt + 1}/{max_retries})"
                    )
                    continue
                raise ValueError(f"Invalid JSON response: {completion}")

        except Exception as e:
            logger.error(f"Error on attempt {attempt + 1}/{max_retries}: {str(e)}")
            if attempt < max_retries - 1:
                logger.info(
                    f"Retrying comparison (attempt {attempt + 1}/{max_retries})"
                )
                continue
            raise

    raise RuntimeError(f"Failed to get valid comparison after {max_retries} attempts")


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
            cache_size=cache_size,
            temperature=0.1,
            max_retries=3,
        )
        self.model = "gpt-4o-mini"

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
            model=self.model,
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
        self, trial: "ClinicalTrial", conditions: str | list[str]
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

        max_retries = 3
        total_cost = 0.0

        for attempt in range(max_retries):
            try:
                response_content, cost = self._call_gpt(
                    prompt,
                    "You are a clinical trial analyst focused on evaluating titles.",
                    temperature=0.1,
                    refresh_cache=(attempt > 0),  # Refresh cache on retries
                )
                total_cost += cost
                result = self._parse_gpt_response(response_content)
                return result["suitability_probability"], result["reason"], total_cost
            except json.JSONDecodeError:
                logger.warning(
                    f"GPTTrialFilter.evaluate_title: JSON parsing failed on attempt {attempt + 1}/{max_retries}. Response content: {response_content}"
                )
                if attempt == max_retries - 1:  # Last attempt
                    logger.error(
                        "GPTTrialFilter.evaluate_title: All attempts to parse JSON failed"
                    )
                    return (
                        0.5,
                        "Unable to confidently evaluate title due to parsing errors - treating as uncertain",
                        total_cost,
                    )

    def evaluate_trial(
        self, trial: "ClinicalTrial", conditions: str | list[str]
    ) -> Tuple[bool, float, Optional[TrialFailureReason]]:
        """
        Evaluate if a trial is suitable for given conditions.

        Args:
            trial: The clinical trial to evaluate
            conditions: Patient condition(s) to check against, either as a single string or list of strings

        Returns:
            Tuple of:
                bool: Whether the trial is suitable
                float: Probability of suitability (0.0-1.0)
                Optional[TrialFailureReason]: Reason for failure if trial is not suitable
        """
        # First evaluate the title
        title_prob, title_reason, _ = self.evaluate_title(trial, conditions)

        # If title is clearly unsuitable, return early
        if title_prob < 0.2:
            return (
                False,
                title_prob,
                TrialFailureReason(
                    type="title",
                    message=f"Trial title indicates poor suitability: {title_reason}",
                ),
            )

        # If title is promising, return success
        if title_prob > 0.8:
            return True, title_prob, None

        # For borderline cases, return uncertain but potentially suitable
        return True, title_prob, None
