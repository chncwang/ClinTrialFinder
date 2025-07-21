import json
import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from base.clinical_trial import ClinicalTrial
from base.disease_expert import extract_disease_from_record, is_oncology_disease
from base.drug_analyzer import analyze_drug_effectiveness
from base.gpt_client import GPTClient
from base.utils import parse_json_response, save_json_list_file

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from base.perplexity import PerplexityClient

CLINICAL_TRIAL_SYSTEM_PROMPT = (
    "You are a clinical research expert with extensive experience in evaluating patient eligibility and treatment outcomes. "
    "Your expertise includes analyzing clinical trials, published research, and making evidence-based recommendations for patient care."
)

TRIAL_COMPARISON_SYSTEM_PROMPT = (
    "You are an experienced oncologist and clinical trial specialist with expertise in personalized medicine and evidence-based treatment selection. "
    "Your role is to act as the patient's advocate, carefully analyzing clinical trial options to determine which would provide the best potential outcome for the specific patient."
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
        "<task>Assess if this clinical trial would be beneficial for the patient. "
        "Analyze published research and clinical evidence on similar drugs and treatments to inform your recommendation. "
        "You must provide your assessment in a JSON format with a recommendation level and concise reasoning (at most 50 tokens).</task>\n\n"
        "<recommendation_levels>The possible recommendation levels are:\n\n"
        "- Strongly Recommended\n"
        "- Recommended\n"
        "- Neutral\n"
        "- Not Recommended</recommendation_levels>\n\n"
        "<output_format>You must respond with a JSON object containing:\n"
        "- reason: A concise explanation (at most 50 tokens) of your recommendation based on the evidence provided\n"
        "- recommendation: One of the recommendation levels listed above</output_format>\n\n"
        "<output_request>Based on the clinical record, trial information, and drug effectiveness analysis (if available), "
        "evaluate if this trial would be beneficial for the patient. Consider both the patient's condition and the evidence "
        "regarding the trial's treatment approach. Your explanation must be concise (at most 50 tokens).</output_request>\n\n"
        f'<clinical_record>\nClinical Record:\n"{clinical_record}"\n</clinical_record>\n\n'
        f'<trial_info>\nTrial Information:\n"{trial_info_str}"\n</trial_info>'
        f"{drug_analysis_str}"
    )


def build_trial_info(trial: ClinicalTrial) -> str:
    """
    Build trial information string for comparison.
    
    Args:
        trial: The clinical trial to build info for
        
    Returns:
        Formatted trial information string
    """
    info_lines = [
        f"NCT ID: {trial.identification.nct_id}",
        f"Title: {trial.identification.brief_title}",
    ]
    info_lines.extend([
        f"Reason: {trial.analysis_reason}",
        f"Drug Analyses: {json.dumps(trial.drug_analysis, indent=2)}"
    ])
    
    # Add arms information if available
    if trial.design.arms:
        arms_info = []
        for i, arm in enumerate(trial.design.arms, 1):
            arm_str = f"Arm {i}: {arm.get('name', 'N/A')} ({arm.get('type', 'N/A')})"
            if arm.get('description'):
                arm_str += f" - {arm['description']}"
            if arm.get('interventions'):
                arm_str += f" - Interventions: {', '.join(arm['interventions'])}"
            arms_info.append(arm_str)
        info_lines.append(f"Arms: {'; '.join(arms_info)}")
    
    return "\n".join(info_lines)


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
        # Attempt to clean the response of any control characters first
        cleaned_response = "".join(
            char for char in response if ord(char) >= 32 or char in "\n\r\t"
        )

        try:
            data = json.loads(cleaned_response)
        except json.JSONDecodeError:
            # Last resort - if JSON parsing still fails, manually extract the fields using regex
            logger.warning(
                "parse_recommendation_response: JSON parsing failed, attempting manual extraction"
            )

            # Extract recommendation field
            rec_match = re.search(r'"recommendation"\s*:\s*"([^"]+)"', cleaned_response)
            recommendation_str = rec_match.group(1) if rec_match else None

            # Extract reason field (this handles multiline text)
            reason_start = cleaned_response.find('"reason"')
            if reason_start != -1:
                # Find the first quote after "reason":
                quote_start = cleaned_response.find('"', reason_start + 8)
                if quote_start != -1:
                    # Find the closing quote for the reason field
                    depth = 0
                    for i in range(quote_start + 1, len(cleaned_response)):
                        if (
                            cleaned_response[i] == '"'
                            and cleaned_response[i - 1] != "\\"
                        ):
                            # Only count this quote as closing if we're at the root level
                            if depth == 0:
                                reason = cleaned_response[quote_start + 1 : i]
                                break
                            depth -= 1
                        elif cleaned_response[i] == "{":
                            depth += 1
                        elif cleaned_response[i] == "}":
                            depth -= 1
                    else:
                        reason = None
                else:
                    reason = None
            else:
                reason = None

            if not recommendation_str or not reason:
                raise ValueError(
                    "Failed to extract required fields from malformed JSON response"
                )

            # Create data dictionary manually
            data = {"recommendation": recommendation_str, "reason": reason}
            logger.info(
                f"parse_recommendation_response: Manually extracted data: {data}"
            )

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
    except Exception as e:
        logger.error(f"parse_recommendation_response: Unexpected error: {e}")
        logger.error(f"parse_recommendation_response: Response content: {response}")
        raise ValueError(f"Error parsing response: {e}. Response content: {response}")


def analyze_drugs_and_get_recommendation(
    clinical_record: str,
    trial: ClinicalTrial,
    perplexity_client: "PerplexityClient",
    gpt_client: "GPTClient",
    refresh_cache: bool = False,
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
        model="gpt-4.1",
        temperature=0.2,
        response_format={"type": "json_object"},
        refresh_cache=refresh_cache,
    )
    total_cost += cost

    if completion is None:
        raise RuntimeError("Failed to get AI analysis")

    # Use the existing parse_json_response utility for robust error handling
    try:
        data, correction_cost = parse_json_response(completion, dict, gpt_client, 0.0)
        total_cost += correction_cost

        # Ensure required fields are present
        if "reason" not in data or "recommendation" not in data:
            raise ValueError(
                "JSON response missing 'reason' or 'recommendation' fields"
            )
    except ValueError as e:
        logger.error(
            f"analyze_drugs_and_get_recommendation: Failed to convert to valid JSON: {completion}"
        )
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
    disease: str,
    trial1: ClinicalTrial,
    trial2: ClinicalTrial,
    gpt_client: "GPTClient",
    refresh_cache: bool = False,
) -> tuple["ClinicalTrial", str, float]:
    """
    Compare two clinical trials and determine which is better for a target patient.

    Args:
        clinical_record (str): The patient's clinical record
        disease (str): The patient's disease
        trial1 (ClinicalTrial): First trial to compare
        trial2 (ClinicalTrial): Second trial to compare
        gpt_client (GPTClient): Client for GPT-4 analysis
        refresh_cache (bool): Whether to refresh the cache

    Returns:
        tuple containing:
            - The better trial (ClinicalTrial)
            - Detailed explanation of the comparison
            - Total cost of API calls
    """
    total_cost = 0.0
    max_retries = 3

    # Build comparison prompt using existing drug analyses
    trial1_info = build_trial_info(trial1)
    trial2_info = build_trial_info(trial2)

    if is_oncology_disease(disease):
        oncology_guidance = (
            "<oncology_guidance>"
            "You should infer whether the patient is heavily pretreated or treatment-naive from the clinical record. "
            "Prioritize Phase 2 trials for heavily pretreated patients, and Phase 3 trials for those with no or minimal prior treatment. "
            "If both trials are the same phase, or the patient's status is unclear, use your best judgment based on the other criteria."
            "</oncology_guidance>\n\n"
        )
    else:
        oncology_guidance = ""

    comparison_prompt = (
        "<task>Compare these two clinical trials and determine which one would be better suited for this specific patient based on their clinical record. "
        "Analyze both trials' characteristics, drug effectiveness data, and other relevant factors to make an informed decision.</task>\n\n"
        "<evaluation_criteria>Consider the following factors when comparing trials:\n\n"
        "- Patient's specific conditions and how they align with trial eligibility\n"
        "- Trial design and methodology (including phase, study type, and treatment arms), with particular attention to the implications for the patient if assigned to the control arm\n"
        "- Drug effectiveness evidence and safety profiles\n"
        "- Potential benefits vs. risks for the specific patient\n"
        "- Trial status and availability</evaluation_criteria>\n\n"
        f"{oncology_guidance}"
        "<output_format>You must respond with a JSON object containing:\n"
        "- reason: A concise explanation (max 50 words) of why one trial is better, or why neither trial is suitable\n"
        "- better_trial: The NCT ID of the better trial, or 'neither' if both trials are equally unsuitable</output_format>\n\n"
        "<decision_guidance>When making your decision:\n"
        "- Prioritize patient safety and potential benefit\n"
        "- Consider the strength of evidence supporting each trial\n"
        "- If both trials are equally poor matches or unsuitable, choose 'neither'\n"
        "- Keep reasoning concise and focused on key factors</decision_guidance>\n\n"
        f'<clinical_record>\nClinical Record:\n"{clinical_record}"\n</clinical_record>\n\n'
        f"<trial1_info>\n{trial1_info}\n</trial1_info>\n\n"
        f"<trial2_info>\n{trial2_info}\n</trial2_info>"
    )
    logger.info(f"compare_trials: Comparison Prompt:\n{comparison_prompt}")

    for attempt in range(max_retries):
        try:
            # Get comparison from GPT-4
            completion, cost = gpt_client.call_gpt(
                prompt=comparison_prompt,
                system_role=TRIAL_COMPARISON_SYSTEM_PROMPT,
                model="gpt-4.1",
                temperature=0.2,
                response_format={"type": "json_object"},
                refresh_cache=refresh_cache or attempt > 0,
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
                        f"compare_trials: Response missing required fields. Response content: {completion}"
                    )
                    if attempt < max_retries - 1:
                        logger.info(
                            f"compare_trials: Retrying comparison (attempt {attempt + 1}/{max_retries})"
                        )
                        continue
                    raise ValueError(
                        "Response missing required 'better_trial' or 'reason' fields"
                    )

                # Determine which trial is better
                if better_trial_id == "neither":
                    # Neither trial is suitable - return trial1 as default but note in reason
                    better_trial = trial1
                else:
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
                    f"compare_trials: Invalid JSON response on attempt {attempt + 1}/{max_retries}: {completion}"
                )
                if attempt < max_retries - 1:
                    logger.info(
                        f"compare_trials: Retrying comparison (attempt {attempt + 1}/{max_retries})"
                    )
                    continue
                raise ValueError(f"Invalid JSON response: {completion}")

        except Exception as e:
            logger.error(
                f"compare_trials: Error on attempt {attempt + 1}/{max_retries}: {str(e)}"
            )
            if attempt < max_retries - 1:
                logger.info(
                    f"compare_trials: Retrying comparison (attempt {attempt + 1}/{max_retries})"
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

    def _call_gpt(
        self,
        prompt: str,
        system_role: str,
        model: str,  # Required model parameter
        temperature: float = 0.1,
        refresh_cache: bool = False,
    ) -> Tuple[str, float]:
        """Common method for making GPT API calls."""
        return self.gpt_client.call_gpt(
            prompt,
            system_role,
            model=model,
            temperature=temperature,
            refresh_cache=refresh_cache,
            response_format={"type": "json_object"},
        )

    def _call_gpt_with_retry(
        self, prompt: str, system_role: str, model: str, max_retries: int = 3
    ) -> Tuple[str, float]:
        return self.gpt_client.call_with_retry(
            prompt,
            system_role,
            model=model,
            response_format={"type": "json_object"},
            validate_json=True,
        )

    def _parse_gpt_response(self, response_content: str) -> dict:
        """Parse GPT response content into JSON, with error handling."""
        try:
            # First try to parse the response directly
            return json.loads(response_content)
        except json.JSONDecodeError as e:
            logger.error(
                f"GPTTrialFilter._parse_gpt_response: Failed to parse GPT response: {response_content}"
            )

            # Try to clean the response by removing any potential markdown formatting
            cleaned_response = response_content.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()

            try:
                # Try parsing the cleaned response
                return json.loads(cleaned_response)
            except json.JSONDecodeError:
                # If still failing, try to extract just the JSON object using regex
                import re

                json_match = re.search(r"\{.*\}", cleaned_response, re.DOTALL)
                if json_match:
                    try:
                        return json.loads(json_match.group(0))
                    except json.JSONDecodeError:
                        pass

                # If all attempts fail, raise the original error
                raise ValueError(f"Failed to parse GPT response as JSON: {str(e)}")

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
                f"GPTTrialFilter._parse_gpt_response_with_fallback: Failed to parse GPT response, using default values: {response_content}"
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
{{"reason": "The patient condition does not mention any information related to the inclusion criterion, so it is fully compatible.", "suitability_probability": 1.0}}

Study Title:
{title}

Inclusion Criterion:
{criterion}

Patient Condition to Evaluate:
{condition}"""

    def evaluate_title(
        self,
        trial: ClinicalTrial,
        conditions: str | list[str],
        refresh_cache: bool = False,
    ) -> Tuple[float, str, float]:
        """
        Evaluate if a trial title indicates suitability for given conditions.
        Uses GPT-4 for accurate evaluation.
        """
        conditions_list = conditions if isinstance(conditions, list) else [conditions]
        conditions_text = "\n".join(f"- {condition}" for condition in conditions_list)

        prompt = f"""You are filtering clinical trials based on patient conditions and trial title.

Please determine if the trial is potentially suitable for the patient conditions.

Return a JSON object containing:
- "reason": An explanation of why the title is or is not suitable
- "suitability_probability": A float value between 0.0 and 1.0 representing how suitable the trial is:
  - 0.0: Completely unsuitable
  - 0.5: Uncertain
  - 1.0: Completely suitable

Example response:
{{"reason": "[specific reasons]", "suitability_probability": 0.8}}

Trial Details:
- Title: {trial.identification.brief_title}

Patient Conditions to Evaluate:
{conditions_text}"""

        max_retries = 3
        total_cost = 0.0

        for attempt in range(max_retries):
            try:
                response_content, cost = self._call_gpt(
                    prompt,
                    "You are a clinical trial analyst focused on evaluating titles.",
                    model="gpt-4.1-mini",  # Use GPT-4 for title evaluation
                    temperature=0.1,
                    refresh_cache=(attempt > 0) or refresh_cache,
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
        Uses GPT-4 for accurate evaluation.
        """
        prompt = f"""You are evaluating a clinical trial inclusion criterion against multiple patient conditions.

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
{{"reason": "None of the conditions mention information related to the inclusion criterion, so it is fully compatible.", "suitability_probability": 1.0}}

Study Title:
{title}

Inclusion Criterion:
{criterion}

Patient Conditions to Evaluate:
{json.dumps(conditions, indent=2)}"""

        try:
            # Try with retries first
            response_content, cost = self._call_gpt_with_retry(
                prompt,
                "You are a clinical trial analyst focused on evaluating inclusion criteria.",
                model="gpt-4.1-mini",  # Use GPT-4 for criterion evaluation
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
            logger.error(
                f"GPTTrialFilter.evaluate_inclusion_criterion: Failed to evaluate criterion: {str(e)}"
            )
            raise

    def _is_or_criterion(self, criterion: str, refresh_cache: bool = False) -> bool:
        """Check if a criterion contains top-level OR logic using GPT."""
        prompt = f"""Analyze this clinical trial inclusion criterion for top-level OR logic:

Does this criterion contain multiple alternative options connected by OR at the top level (not nested within subgroups)? Respond ONLY with JSON:
{{"is_or_criterion": true/false}}

Criterion: {criterion}"""

        try:
            response_content, _ = self._call_gpt(
                prompt,
                "You are a clinical trial analyst specializing in logical structure analysis.",
                model="gpt-4.1-mini",  # Use GPT-4 Mini for simple parsing
                temperature=0.0,
                refresh_cache=refresh_cache,
            )
            result = self._parse_gpt_response(response_content)
            return result.get("is_or_criterion", False)
        except json.JSONDecodeError:
            # Retry with cache refresh on parse error
            response_content, _ = self._call_gpt(
                prompt,
                "You are a clinical trial analyst specializing in logical structure analysis.",
                model="gpt-4.1-mini",  # Use GPT-4 Mini for simple parsing
                temperature=0.0,
                refresh_cache=True,
            )
            result = self._parse_gpt_response(response_content)
            return result.get("is_or_criterion", False)

    def _split_or_branches(
        self, criterion: str, refresh_cache: bool = False
    ) -> List[str]:
        """Split a criterion with top-level OR logic into individual branches."""
        prompt = f"""Split this clinical trial inclusion criterion into separate OR branches:

Rules:
1. Split only at TOP-LEVEL OR connections
2. Maintain nested AND/OR structures within branches
3. Preserve all original requirements in each branch

Example:
Input: "Patient must have diabetes mellitus, hypertension, or chronic kidney disease"
Output: {{
    "branches": [
        "Patient must have diabetes mellitus",
        "Patient must have hypertension",
        "Patient must have chronic kidney disease"
    ]
}}

Return ONLY a JSON object with a "branches" list containing the split criteria:
{{"branches": ["branch 1 text", "branch 2 text", ...]}}

Original Criterion:
{criterion}"""

        response_content, _ = self._call_gpt(
            prompt,
            "You are a clinical trial analyst specializing in logical structure analysis.",
            model="gpt-4.1-mini",
            temperature=0.0,
            refresh_cache=refresh_cache,
        )
        result = self._parse_gpt_response(response_content)
        return result.get("branches", [criterion])

    def _validate_condition_with_gpt(
        self, condition: str, condition_list: List[str], refresh_cache: bool = False
    ) -> Tuple[bool, str, float]:
        """
        Use GPT to validate if a condition is semantically present in a list of conditions.

        Args:
            condition: The candidate condition to validate
            condition_list: The reference list of valid conditions
            refresh_cache: Whether to refresh the GPT cache

        Returns:
            Tuple of (is_valid, matched_condition, cost)
            - is_valid: Whether the condition is semantically present in the list
            - matched_condition: The exact condition from the list that matched, or empty if none
            - cost: The API cost incurred
        """
        # Prepare prompt with the condition and reference list
        prompt = f"""You are validating if a candidate condition is present in a reference list of valid conditions.

Task: Determine if the candidate condition is semantically equivalent to any condition in the reference list.
If it is, respond with the word "YES".
If there is no semantic match, respond only with the word "NO".

IMPORTANT:
- A match should represent the same medical concept/condition
- Minor rewording, abbreviation expansion, or formatting differences can be ignored
- The condition must refer to the same disease, symptom, or health state
- Don't match conditions that are merely related or in the same category but different

Example response 1:
YES

Example response 2:
NO

Candidate condition to validate:
"{condition}"

Reference list of valid conditions:
{json.dumps(condition_list, indent=2)}"""

        response_content, cost = self.gpt_client.call_gpt(
            prompt=prompt,
            system_role="You are a medical terminology expert specializing in semantic matching.",
            model="gpt-4.1-mini",
            temperature=0.0,
            refresh_cache=refresh_cache,
        )

        logger.debug(
            f"GPTTrialFilter._validate_condition_with_gpt: Validation response for '{condition}': {response_content}"
        )

        # Parse the simple YES/NO response
        response_text = response_content.strip()

        if response_text.upper() == "YES":
            # Find the best matching condition in the list
            for valid_condition in condition_list:
                if (
                    valid_condition.lower() in condition.lower()
                    or condition.lower() in valid_condition.lower()
                ):
                    logger.info(
                        f"GPTTrialFilter._validate_condition_with_gpt: Found close match condition: '{condition}' -> '{valid_condition}'"
                    )
                    return True, valid_condition, cost

            # If we get here, no close match was found
            logger.warning(
                f"GPTTrialFilter._validate_condition_with_gpt: GPT returned YES but no matching condition found in the list"
            )
            return False, "", cost
        else:
            logger.info(
                f"GPTTrialFilter._validate_condition_with_gpt: No semantic match found for condition: '{condition}'"
            )
            return False, "", cost

    def choose_most_relevant_conditions(
        self,
        branch: str,
        conditions: List[str],
        trial_title: str,
        num_conditions: int = 5,
        refresh_cache: bool = False,
        need_to_note_list: List[str] = None,
    ) -> List[str]:
        """Choose the most relevant conditions from a list of conditions for a given branch."""
        if num_conditions >= len(conditions):
            return conditions

        # Initialize need-to-note list if none provided
        if need_to_note_list is None:
            need_to_note_list = []

        # Build note for previous failures if any
        validation_note = ""
        if need_to_note_list:
            validation_note = (
                "\n\nIMPORTANT NOTE: Previous responses had the following issues:\n"
            )
            for i, note in enumerate(need_to_note_list, 1):
                validation_note += f"{i}. {note}\n"
            validation_note += "\nYou MUST only select conditions that are exactly present in the provided list. Do not modify, paraphrase, or create new conditions."

        prompt = f"""You are analyzing a clinical trial inclusion criterion branch to determine which patient conditions are most relevant.

Task: Choose the {num_conditions} most relevant conditions that should be evaluated against this inclusion criterion branch, ordered by relevance.

IMPORTANT: You MUST ONLY select conditions that EXACTLY match entries in the provided Patient Conditions list. Do not modify the text of any condition, do not paraphrase, and do not create new conditions.{validation_note}

Return ONLY a JSON object with this structure:
{{"relevant_conditions": ["condition1", "condition2", ...]}}

Example 1:
If the inclusion criterion branch is:
"Patient must have histologically or cytologically confirmed non-small cell lung cancer (NSCLC) with at least one measurable lesion according to RECIST 1.1 criteria."

And the patient conditions include:
1. "65-year-old female"
2. "Type 2 Diabetes Mellitus diagnosed 13 years ago, managed with Metformin"
3. "Non-small cell lung cancer (Stage 3) diagnosed 8 months ago, currently on chemotherapy"
4. "Liver metastasis detected 6 months ago, stable on recent imaging"
5. "Hypertension, well-controlled on Lisinopril"
6. "Chronic kidney disease (Stage 2) with eGFR 75 mL/min"
7. "Myocardial infarction 4 years ago, treated with stent placement"
8. "Osteoarthritis of the right knee, managed with NSAIDs as needed"
9. "Hypothyroidism, on levothyroxine"
10. "COPD with history of 2 exacerbations in the past year, uses albuterol inhaler"

Your response should be:
{{"relevant_conditions": ["Non-small cell lung cancer (Stage 3) diagnosed 8 months ago, currently on chemotherapy", "Liver metastasis detected 6 months ago, stable on recent imaging", "COPD with history of 2 exacerbations in the past year, uses albuterol inhaler"]}}

Example 2:
If the inclusion criterion branch is:
"Patient must NOT have any history of lung cancer or respiratory conditions."

And the patient conditions include the same list as above, your response should be:
{{"relevant_conditions": ["Non-small cell lung cancer (Stage 3) diagnosed 8 months ago, currently on chemotherapy", "COPD with history of 2 exacerbations in the past year, uses albuterol inhaler", "Liver metastasis detected 6 months ago, stable on recent imaging"]}}

Note that even though these conditions would make the patient ineligible, they are still the most relevant to evaluate against this criterion.

Trial Title: {trial_title}

Inclusion Criterion Branch:
{branch}

Patient Conditions:
{json.dumps(conditions, indent=2)}"""

        response_content, cost = self._call_gpt(
            prompt,
            "You are a clinical trial analyst specializing in patient condition relevance.",
            model="gpt-4.1-mini",
            temperature=0.0,
            refresh_cache=refresh_cache,
        )

        total_cost = cost

        try:
            result = self._parse_gpt_response(response_content)
            logger.info(
                f"GPTTrialFilter.choose_most_relevant_conditions: Branch: {branch}, Result: {result}"
            )
            relevant_conditions = result.get("relevant_conditions", [conditions[0]])

            # Validate each condition using GPT
            validated_conditions = []
            invalid_conditions = []

            for condition in relevant_conditions:
                # First check for exact match for efficiency
                if condition in conditions:
                    validated_conditions.append(condition)
                    continue

                # If not an exact match, use GPT to check for semantic match
                is_match, matched_condition, validation_cost = (
                    self._validate_condition_with_gpt(
                        condition, conditions, refresh_cache
                    )
                )
                total_cost += validation_cost

                if is_match and matched_condition:
                    # Use the exact condition text from the reference list
                    validated_conditions.append(matched_condition)
                    logger.info(
                        f"GPTTrialFilter.choose_most_relevant_conditions: Condition semantically matched: '{condition}' -> '{matched_condition}'"
                    )
                else:
                    invalid_conditions.append(condition)
                    logger.warning(
                        f"GPTTrialFilter.choose_most_relevant_conditions: No semantic match found for condition: '{condition}'"
                    )

            # If there are invalid conditions, add them to need-to-note list and retry
            if invalid_conditions:
                error_message = f"Found conditions that are not in the original list: {', '.join(invalid_conditions)}"
                logger.warning(
                    f"GPTTrialFilter.choose_most_relevant_conditions: {error_message}"
                )

                # Add to need-to-note list
                need_to_note_list.append(error_message)

                # Retry with updated need-to-note list
                if (
                    len(need_to_note_list) <= 3
                ):  # Limit retries to prevent infinite loops
                    return self.choose_most_relevant_conditions(
                        branch,
                        conditions,
                        trial_title,
                        num_conditions,
                        True,  # Force refresh cache on retry
                        need_to_note_list,
                    )
                else:
                    logger.error(
                        "GPTTrialFilter.choose_most_relevant_conditions: Too many retries, falling back to first conditions"
                    )
                    return conditions[:num_conditions]

            # Ensure we don't return more conditions than requested
            return validated_conditions[:num_conditions]
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(
                f"GPTTrialFilter.choose_most_relevant_conditions: Failed to parse most relevant conditions response: {e}. Using first {num_conditions} conditions as fallback."
            )
            return conditions[:num_conditions]

    def split_inclusion_criteria(
        self, criteria: str, refresh_cache: bool = False
    ) -> List[str]:
        """Split the inclusion criteria into individual statements using GPT."""
        prompt = f"""You are analyzing clinical trial inclusion criteria text.

Split this text into individual inclusion criterion statements. Preserve logical structure and relationships between criteria.

Return ONLY valid JSON with criteria list.

Example original criteria:
"Inclusion Criteria:
- Patient must have one of the following: a) Triple-negative breast cancer with PD-L1 CPS ≥10 and progression after prior therapy; b) Advanced endometrial cancer with MSI-H/dMMR and platinum-therapy failure
- Adequate organ function per laboratory parameters
- ECOG performance status 0-1
- Life expectancy ≥12 weeks"

Example response:
{{"criteria": [
    "Patient must have one of the following: (a) Triple-negative breast cancer with PD-L1 CPS ≥10 and progression after prior therapy; OR (b) Advanced endometrial cancer with MSI-H/dMMR and platinum-therapy failure",
    "Adequate organ function per laboratory parameters",
    "ECOG performance status 0-1",
    "Life expectancy ≥12 weeks"
]}}

Example original criteria:
"Inclusion Criteria:
* Age:18-75 years, male or female.
* ECOG 0-2
* Histologically or cytologically confirmed de novo metastatic nasopharyngeal carcinoma.(stage IVb, AJCC 8th)
* Complete response or partial response after at least 3 cycles (no more than 6 cycles) of chemotherapy combined with immunotherapy
* Measurable disease based on Response Evaluation Criteria In Solid Tumors (RECIST) 1.1.
* Adequate organ function.
* Patient has given written informed consent."

Example response:
{{"criteria": [
    "Age 18-75 years, male or female",
    "ECOG performance status 0-2",
    "Histologically or cytologically confirmed de novo metastatic nasopharyngeal carcinoma (stage IVb, AJCC 8th)",
    "Complete response or partial response after at least 3 cycles (no more than 6 cycles) of chemotherapy combined with immunotherapy",
    "Measurable disease based on Response Evaluation Criteria In Solid Tumors (RECIST) 1.1",
    "Adequate organ function",
    "Patient has given written informed consent"
]}}

Inclusion Criteria Text:
{criteria}"""

        response_content, cost = self._call_gpt(
            prompt,
            "You are a clinical trial analyst focused on parsing inclusion criteria.",
            model="gpt-4.1-mini",  # Use GPT-4 Mini for parsing
            temperature=0.0,
            refresh_cache=refresh_cache,
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

    def _evaluate_branch(
        self, branch: str, conditions: List[str], trial_title: str
    ) -> Tuple[float, Dict[str, CriterionEvaluation], float]:
        """
        Evaluate a branch against given conditions.
        """
        cost_sum = 0.0
        branch_condition_evaluations = {}

        # Get the most relevant conditions with validation
        need_to_note_list = []
        most_relevant_conditions = self.choose_most_relevant_conditions(
            branch, conditions, trial_title, need_to_note_list=need_to_note_list
        )

        # Evaluate the branch against the most relevant conditions
        probability, reason, cost = self.evaluate_inclusion_criterion(
            branch, most_relevant_conditions, trial_title
        )
        cost_sum += cost
        logger.info(
            f"GPTTrialFilter._evaluate_branch: Branch evaluation:\n"
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

    def process_branches(
        self, branches: List[str], conditions: List[str], trial_title: str
    ) -> Tuple[float, Dict[str, List[CriterionEvaluation]], float]:
        """
        Process branches sequentially.
        """
        branch_max_prob = 0.0
        branch_cost_sum = 0.0
        branch_results = {condition: [] for condition in conditions}
        global_need_to_note_list = []

        # Process each branch sequentially
        for branch in branches:
            # Evaluate branch without passing need_to_note_list to _evaluate_branch
            # since it already manages its own need_to_note_list
            branch_prob, branch_condition_evaluations, branch_cost = (
                self._evaluate_branch(branch, conditions, trial_title)
            )
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

    def _get_or_criterion_failure_reason(
        self, branch_results: Dict[str, List[CriterionEvaluation]], criterion: str
    ) -> Tuple[str, str, str]:
        """
        Analyze branch results to determine why a condition failed all branches of an OR criterion.

        Args:
            branch_results: Dictionary mapping conditions to their evaluations for each branch
            criterion: The original OR criterion that was evaluated

        Returns:
            Tuple containing:
            - The condition that failed
            - The criterion that failed
            - A detailed explanation of why it failed
        """
        # Find the condition that had the highest eligibility across all branches
        best_condition = None
        best_eligibility = -1.0
        best_reason = ""

        for condition, evaluations in branch_results.items():
            if not evaluations:
                continue

            # Get the best evaluation for this condition across all branches
            best_eval = max(evaluations, key=lambda x: x.eligibility)
            if best_eval.eligibility > best_eligibility:
                best_eligibility = best_eval.eligibility
                best_condition = condition
                best_reason = best_eval.reason

        if best_condition is None:
            return (
                "",
                criterion,
                "No conditions were evaluated against this criterion",
            )

        return (best_condition, criterion, best_reason)

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

                # Run process_branches and get results
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
                need_to_note_list = []
                most_relevant_conditions: List[str] = (
                    self.choose_most_relevant_conditions(
                        criterion,
                        conditions,
                        trial_title,
                        need_to_note_list=need_to_note_list,
                    )
                )

                # Only evaluate the most relevant condition
                probability, reason, cost = self.evaluate_inclusion_criterion(
                    criterion, most_relevant_conditions, trial_title
                )
                overall_probability *= probability
                total_cost += cost

                if probability <= 0.0:
                    failure_reason = (
                        ", ".join(most_relevant_conditions),
                        criterion,
                        reason,
                    )
                    break

        if overall_probability <= 0.0 and failure_reason is None:
            raise RuntimeError(
                "Illegal state: overall_probability <= 0.0 but no failure reason was recorded"
            )
        return overall_probability, failure_reason, total_cost

    def evaluate_trial(
        self,
        trial: ClinicalTrial,
        conditions: list[str],
        refresh_cache: bool = False,
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
            - failure_reason: If ineligible, a TrialFailureReason object describing why it failed. If eligible, None.
        """
        # 1) Evaluate the title first
        title_probability, title_reason, title_cost = self.evaluate_title(
            trial, conditions, refresh_cache
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
    refresh_cache: bool = False,
) -> Tuple[float, int]:
    """Process trials with given conditions and save results.

    Args:
        trials: List of clinical trials to process
        conditions: List of conditions to filter trials
        output_path: Path to save output files
        gpt_filter: Optional GPTTrialFilter instance for condition evaluation
        refresh_cache: Whether to refresh the cache of GPT responses
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
                trial, conditions, refresh_cache
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
