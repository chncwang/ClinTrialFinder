import json
import logging
from enum import Enum
from typing import TYPE_CHECKING

from base.disease_extractor import extract_disease_from_record
from base.drug_analyzer import analyze_drug_effectiveness

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from base.clinical_trial import ClinicalTrial
    from base.gpt_client import GPTClient
    from base.perplexity_client import PerplexityClient

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
        raise ValueError(f"Invalid JSON response: {e}. Response content: {response}")


def analyze_drugs_and_get_recommendation(
    clinical_record: str,
    trial: "ClinicalTrial",
    perplexity_client: "PerplexityClient",
    gpt_client: "GPTClient",
) -> tuple[str, str, float]:
    """
    Analyze drug effectiveness and generate AI recommendation.

    Returns:
        tuple containing:
            - Recommendation level (Strongly Recommended, Recommended, etc.)
            - Reason for the recommendation
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

    completion, cost = gpt_client.call_gpt(
        prompt=prompt,
        system_role=CLINICAL_TRIAL_SYSTEM_PROMPT,
        temperature=0.2,
    )
    total_cost += cost

    if completion is None:
        raise RuntimeError("Failed to get AI analysis")

    recommendation, reason = parse_recommendation_response(completion)
    logger.info(
        f"analyze_drugs_and_get_recommendation: Recommendation: {recommendation}"
    )
    logger.info(f"analyze_drugs_and_get_recommendation: Reason: {reason}")

    return recommendation, reason, total_cost
