import json
import requests
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

from base.perplexity import PerplexityClient

# Lazy import to avoid circular dependency
_gpt_client_type = None


def _get_gpt_client_type():
    global _gpt_client_type
    if _gpt_client_type is None:
        from base.gpt_client import GPTClient
        _gpt_client_type = GPTClient
    return _gpt_client_type


def classify_evidence_strength(
    drug_name: str, disease: str, analysis: str, gpt_client
) -> Tuple[str, float]:
    """
    Uses GPT to classify the strength of drug evidence for a specific disease.

    Returns:
        Tuple of (strength level "high"/"medium"/"low", cost)
    """
    prompt = (
        f"<task>Rate the strength of evidence for {drug_name} in treating {disease} based on the analysis below.</task>\n\n"
        f"<analysis>\n{analysis}\n</analysis>\n\n"
        '<output_format>Respond with a JSON object: {{"strength": "high"|"medium"|"low"}}\n'
        "- high: Published clinical trial results with efficacy data specific to this disease\n"
        "- medium: Some relevant data (e.g., early-phase trials, case reports, or indirect evidence)\n"
        '- low: No published clinical data for this disease, or only preclinical/theoretical rationale</output_format>'
    )

    system_role = "You are a clinical evidence evaluator. Assess drug evidence strength objectively."

    completion, cost = gpt_client.call_gpt(
        prompt=prompt,
        system_role=system_role,
        model="gpt-4.1-mini",
        temperature=0.1,
        response_format={"type": "json_object"},
    )

    try:
        result = json.loads(completion)
        strength = result.get("strength", "low").lower()
        if strength not in ("high", "medium", "low"):
            strength = "low"
    except (json.JSONDecodeError, AttributeError):
        logger.warning(f"classify_evidence_strength: Failed to parse response: {completion}")
        strength = "low"

    logger.info(f"classify_evidence_strength: {drug_name} for {disease} -> {strength}")
    return strength, cost


def analyze_drug_efficacy_cross_cancer(
    drug_name: str, perplexity_client: PerplexityClient
) -> Tuple[str, List[str], float]:
    """
    Analyzes a drug's efficacy across all cancer types (not limited to a specific disease).

    Returns:
        Tuple of (analysis text, citations list, cost)
    """
    system_prompt = (
        "<role>You are a pharmaceutical research expert with extensive knowledge of drug efficacy "
        "and clinical outcomes across multiple cancer types.</role>\n\n"
        "<task>Analyze and summarize the clinical trial results and efficacy data for a specific drug "
        "across ALL cancer types where it has been studied. Focus on objective response rates, "
        "survival data, and notable results in any tumor type.</task>"
    )

    user_prompt = (
        f"<drug_analysis_request>\n"
        f"Please analyze the clinical trial results for {drug_name} across all cancer types. "
        f"Include:\n"
        f"1. Which cancer types has this drug been tested in, and what were the results (ORR, PFS, OS)?\n"
        f"2. In which cancer types did it show the strongest efficacy?\n"
        f"3. What is the drug's mechanism of action and drug class (e.g., checkpoint inhibitor, ADC, bispecific)?\n"
        f"4. Any notable regulatory milestones, acquisitions, or investment signals?\n"
        f"Base your analysis on published research and clinical evidence.\n"
        f"</drug_analysis_request>"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        completion, citations, cost = perplexity_client.get_completion(
            messages, max_tokens=2000
        )
        return completion, citations, cost
    except requests.exceptions.RequestException as api_error:
        logger.error(f"analyze_drug_efficacy_cross_cancer: Error: {api_error}")
        raise


def analyze_drug_efficacy(
    drug_name: str, disease: str, perplexity_client: PerplexityClient
) -> Tuple[str, List[str], float]:
    """
    Analyzes the efficacy of a drug for treating a specific disease using Perplexity AI.

    Parameters:
    - drug_name (str): Name of the drug to analyze
    - disease (str): Disease or condition to analyze the drug's efficacy for
    - perplexity_client (PerplexityClient): Initialized Perplexity client

    Returns:
    - tuple[str, list[str], float]: A tuple containing:
        - The analysis text
        - List of citations
        - Cost of the API call
    
    Raises:
        requests.exceptions.RequestException: If the Perplexity API call fails
    """
    system_prompt = (
        "<role>You are a pharmaceutical research expert with extensive knowledge of drug efficacy "
        "and clinical outcomes. Your expertise includes analyzing published research and clinical evidence "
        "to evaluate drug efficacy for specific conditions.</role>\n\n"
        "<task>Analyze and summarize the efficacy of a specific drug for treating a given disease "
        "based on available research and clinical evidence. Focus on key findings regarding efficacy, "
        "safety, and current research status.</task>"
    )

    user_prompt = (
        f"<drug_analysis_request>\n"
        f"Please analyze the efficacy of {drug_name} for treating {disease}. "
        f"Include information about:\n"
        f"1. Clinical trial results and outcomes\n"
        f"2. Safety profile and major side effects\n"
        f"3. Current research status and evidence quality\n"
        f"Base your analysis on published research and clinical evidence.\n"
        f"</drug_analysis_request>"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        completion, citations, cost = perplexity_client.get_completion(
            messages, max_tokens=2000
        )
        return completion, citations, cost
    except requests.exceptions.RequestException as api_error:
        logger.error(f"analyze_drug_efficacy: Error analyzing drug efficacy: {api_error}")
        raise
