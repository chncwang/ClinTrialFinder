import requests
from typing import List, Tuple
from loguru import logger

from base.perplexity import PerplexityClient


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
    except requests.exceptions.RequestException as e:
        logger.error(f"analyze_drug_efficacy: Error analyzing drug efficacy: {e}")
        raise
