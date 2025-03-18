import asyncio
import json
import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from base.clinical_trial import ClinicalTrial
from base.disease_expert import extract_disease_from_record
from base.drug_analyzer import analyze_drug_effectiveness
from base.gpt_client import GPTClient
from base.utils import parse_json_response, save_json_list_file

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
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
    trial_info: ClinicalTrial,
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
    trial: ClinicalTrial,
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
    trial1: ClinicalTrial,
    trial2: ClinicalTrial,
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

    def evaluate_inclusion_criterion(
        self, criterion: str, conditions: List[str], title: str
    ) -> Tuple[float, str, float]:
        """
        Evaluate an inclusion criterion against multiple conditions at once.

        Args:
            criterion: The inclusion criterion to evaluate
            conditions: List of conditions to check against the criterion
            title: Title of the clinical trial for context

        Returns:
            Tuple containing:
                - float: Overall probability of eligibility (best match among conditions)
                - str: Reason for the evaluation
                - float: Cost of the GPT API call
        """
        prompt = f"""You are evaluating a clinical trial inclusion criterion against multiple patient conditions.

Study Title:
{title}

Inclusion Criterion:
{criterion}

Patient Conditions to Evaluate:
{json.dumps(conditions, indent=2)}

Please determine if this inclusion criterion aligns with any of the provided conditions, considering the context from the study title.
Choose the BEST MATCHING condition and evaluate the criterion against it.
If none of the conditions provide information related to the criterion, consider it as fully compatible (probability 1.0).
If the inclusion criterion represents a willingness to participate (e.g. "willing to undergo procedure X"), consider it as suitable.

IMPORTANT: You must respond with a complete, properly formatted JSON object containing exactly these fields:
{{"reason": "your explanation here including which condition was most relevant",
  "suitability_probability": 0.0-1.0}}

Do not include any other text outside the JSON object.

Example response 1:
{{"reason": "Condition X is most relevant. [specific reasons]", "suitability_probability": 0.8}}

Example response 2:
{{"reason": "None of the conditions mention information related to the inclusion criterion, so it is fully compatible.", "suitability_probability": 1.0}}"""

        try:
            # Try with retries first
            response_content, cost = self._call_gpt_with_retry(
                prompt,
                "You are a clinical trial analyst focused on evaluating inclusion criteria.",
            )
            logger.info(
                f"GPTTrialFilter.evaluate_inclusion_criterion: Response content: {response_content}"
            )

            # If response_content is already a dict, use it directly
            result = (
                response_content
                if isinstance(response_content, dict)
                else self._parse_gpt_response_with_fallback(response_content)
            )

            validated_result = self._validate_gpt_response(result)
            logger.info(
                f"GPTTrialFilter.evaluate_inclusion_criterion: Evaluated criterion: {criterion} for conditions: {conditions} with title: {title}"
                + json.dumps(
                    {
                        "criterion": criterion,
                        "conditions": conditions,
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
            raise

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
        logger.debug(
            f"GPTTrialFilter.split_inclusion_criteria: original criteria: {criteria}"
        )
        logger.debug(f"GPTTrialFilter.split_inclusion_criteria: result: {result}")
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
                f"GPTTrialFilter._extract_inclusion_criteria: Missing 'Inclusion Criteria' section in criteria. Brief: {criteria[:50]}..."
            )
            return criteria

        # Split by "Inclusion Criteria" and take everything after it
        inclusion_text = criteria.split("Inclusion Criteria")[1].strip()

        # If there's an "Exclusion Criteria" section, only take the text before it
        if "Exclusion Criteria" in inclusion_text:
            inclusion_text = inclusion_text.split("Exclusion Criteria")[0].strip()
        else:
            logger.warning(
                f"GPTTrialFilter._extract_inclusion_criteria: Missing 'Exclusion Criteria' section in criteria text: {criteria[:50]}..."
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

    async def _evaluate_branch_async(
        self, branch: str, conditions: List[str], trial_title: str
    ) -> Tuple[float, Dict[str, CriterionEvaluation], float]:
        """
        Async version of _evaluate_branch method.
        """
        cost_sum = 0.0
        branch_condition_evaluations = {}

        # Get the most relevant conditions
        most_relevant_conditions = self.choose_most_relevant_conditions(
            branch, conditions, trial_title
        )

        # Evaluate the branch against the most relevant conditions
        probability, reason, cost = await self.evaluate_inclusion_criterion_async(
            branch, most_relevant_conditions, trial_title
        )
        cost_sum += cost
        logger.info(
            f"GPTTrialFilter._evaluate_branch_async: Branch evaluation:\n"
            + json.dumps(
                {
                    "branch": branch,
                    "conditions": most_relevant_conditions,
                    "probability": probability,
                    "cost": cost,
                }
            )
        )

        # Store the evaluation result for all relevant conditions
        for condition in most_relevant_conditions:
            branch_condition_evaluations[condition] = CriterionEvaluation(
                criterion=branch,
                reason=reason,
                eligibility=probability,
            )

        # For other conditions, mark them as fully compatible
        for condition in conditions:
            if condition not in most_relevant_conditions:
                branch_condition_evaluations[condition] = CriterionEvaluation(
                    criterion=branch,
                    reason="Not among the most relevant conditions for this criterion - assumed compatible",
                    eligibility=1.0,
                )

        return probability, branch_condition_evaluations, cost_sum

    async def process_branches_async(
        self, branches: List[str], conditions: List[str], trial_title: str
    ) -> Tuple[float, Dict[str, List[CriterionEvaluation]], float]:
        """
        Async version of process_branches that evaluates branches concurrently.
        """
        branch_max_prob = 0.0
        branch_cost_sum = 0.0
        branch_results = {condition: [] for condition in conditions}

        # Create tasks for all branches
        tasks = [
            self._evaluate_branch_async(branch, conditions, trial_title)
            for branch in branches
        ]

        # Execute all tasks concurrently
        branch_evaluations = await asyncio.gather(*tasks)

        # Process results
        for branch, (branch_prob, branch_condition_evaluations, branch_cost) in zip(
            branches, branch_evaluations
        ):
            branch_cost_sum += branch_cost
            branch_max_prob = max(branch_max_prob, branch_prob)

            # Record which conditions met this branch
            for condition in conditions:
                if condition in branch_condition_evaluations:
                    condition_evaluation = branch_condition_evaluations[condition]
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

    def choose_most_relevant_conditions(
        self,
        branch: str,
        conditions: List[str],
        trial_title: str,
        num_conditions: int = 3,
    ) -> List[str]:
        """
        Choose the most relevant conditions from a list of conditions for a given branch.

        Args:
            branch: A single branch of an inclusion criterion to evaluate
            conditions: List of medical conditions to check against the branch
            trial_title: Title of the clinical trial for context
            num_conditions: Number of most relevant conditions to return (default: 1)

        Returns:
            List of the most relevant conditions, ordered by relevance
        """
        if num_conditions >= len(conditions):
            return conditions

        prompt = f"""You are analyzing a clinical trial inclusion criterion branch to determine which patient conditions are most relevant.

Trial Title: {trial_title}

Inclusion Criterion Branch:
{branch}

Patient Conditions:
{json.dumps(conditions, indent=2)}

Task: Choose the {num_conditions} most relevant conditions that should be evaluated against this inclusion criterion branch, ordered by relevance.

Return ONLY a JSON object with this structure:
{{"relevant_conditions": ["condition1", "condition2", ...]}}"""

        response_content, _ = self._call_gpt(
            prompt,
            "You are a clinical trial analyst specializing in patient condition relevance.",
            temperature=0.0,
        )

        try:
            result = self._parse_gpt_response(response_content)
            logger.info(
                f"GPTTrialFilter.choose_most_relevant_condition: Branch: {branch}, Result: {result}"
            )
            relevant_conditions = result.get("relevant_conditions", [conditions[0]])
            # Ensure we don't return more conditions than requested
            return relevant_conditions[:num_conditions]
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(
                f"Failed to parse most relevant conditions response: {e}. Using first {num_conditions} conditions as fallback."
            )
            return conditions[:num_conditions]

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
                # Choose the most relevant condition for this criterion
                most_relevant_conditions: List[str] = (
                    self.choose_most_relevant_conditions(
                        criterion, conditions, trial_title
                    )
                )

                # Only evaluate the most relevant condition
                probability, reason, cost = self.evaluate_inclusion_criterion(
                    criterion, most_relevant_conditions, trial_title
                )
                overall_probability *= probability
                total_cost += cost

                if probability <= 0.0:
                    failure_reason = (most_relevant_conditions[0], criterion, reason)
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


def process_trials_with_conditions(
    trials: List["ClinicalTrial"],
    conditions: List[str],
    output_path: str,
    gpt_filter: Optional["GPTTrialFilter"] = None,
) -> Tuple[float, int]:
    """Process trials with given conditions and save results.

    Args:
        trials: List of clinical trials to process
        conditions: List of conditions to filter trials
        output_path: Path to save output files
        gpt_filter: Optional GPTTrialFilter instance for condition evaluation

    Returns:
        Tuple of (total API cost, number of eligible trials)
    """
    filtered_trials = []
    excluded_trials = []
    total_trials = len(trials)
    total_cost = 0.0
    eligible_count = 0

    if conditions:
        if not gpt_filter:
            raise ValueError(
                "GPTTrialFilter instance required when conditions are provided"
            )

        logger.info(f"Processing {total_trials} trials with conditions: {conditions}")

        for i, trial in enumerate(trials, 1):
            logger.info(
                f"Processing trial {i}/{total_trials}: {trial.identification.nct_id}"
            )

            is_eligible, cost, failure_reason = gpt_filter.evaluate_trial(
                trial, conditions
            )
            total_cost += cost

            if is_eligible:
                trial_dict = trial.to_dict()
                filtered_trials.append(trial_dict)
                eligible_count += 1
            else:
                excluded_info = {
                    "nct_id": trial.identification.nct_id,
                    "brief_title": trial.identification.brief_title,
                    "eligibility_criteria": trial.eligibility.criteria,
                    "failure_type": failure_reason.type,
                    "failure_message": failure_reason.message,
                }

                if failure_reason.type == "inclusion_criterion":
                    excluded_info.update(
                        {
                            "failed_condition": failure_reason.failed_condition,
                            "failed_criterion": failure_reason.failed_criterion,
                            "failure_details": failure_reason.failure_details,
                        }
                    )

                if (
                    hasattr(trial, "recommendation_level")
                    and trial.recommendation_level
                ):
                    excluded_info["recommendation_level"] = trial.recommendation_level
                if hasattr(trial, "analysis_reason") and trial.analysis_reason:
                    excluded_info["analysis_reason"] = trial.analysis_reason

                excluded_trials.append(excluded_info)

            logger.info(
                f"Eligible trials so far: {eligible_count}/{i} processed, total cost: ${total_cost:.2f}"
            )
    else:
        logger.info(
            "No conditions provided - keeping all trials that passed other filters"
        )
        filtered_trials = [trial.to_dict() for trial in trials]
        eligible_count = len(filtered_trials)

    # Save the passing (filtered) trials
    save_json_list_file(filtered_trials, output_path, "filtered trials")

    # Save the excluded trials only if we did condition filtering
    if conditions:
        excluded_path = output_path.replace(".json", "_excluded.json")
        save_json_list_file(excluded_trials, excluded_path, "excluded trials")

    logger.info(f"Final results: {eligible_count}/{total_trials} trials were eligible")
    logger.info(f"Filtered trials saved to {output_path}")
    if conditions:
        logger.info(f"Excluded trials saved to {excluded_path}")

    return total_cost, eligible_count
