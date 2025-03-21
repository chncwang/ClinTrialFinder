import logging
from typing import List, Optional, Tuple

from base.perplexity import PerplexityClient

logger = logging.getLogger(__name__)


def analyze_drug_effectiveness(
    drug_name: str, disease: str, perplexity_client: PerplexityClient
) -> Tuple[Optional[str], List[str], float]:
    """
    Analyzes the effectiveness of a drug for treating a specific disease using Perplexity AI.

    Parameters:
    - drug_name (str): Name of the drug to analyze
    - disease (str): Disease or condition to analyze the drug's effectiveness for
    - perplexity_client (PerplexityClient): Initialized Perplexity client

    Returns:
    - tuple[str | None, list[str], float]: A tuple containing:
        - The analysis text (or None if failed)
        - List of citations
        - Cost of the API call
    """
    system_prompt = (
        "<role>You are a pharmaceutical research expert with extensive knowledge of drug effectiveness "
        "and clinical outcomes. Your expertise includes analyzing published research and clinical evidence "
        "to evaluate drug efficacy for specific conditions.</role>\n\n"
        "<task>Analyze and summarize the effectiveness of a specific drug for treating a given disease "
        "based on available research and clinical evidence. Focus on key findings regarding efficacy, "
        "safety, and current research status.</task>"
    )

    user_prompt = (
        f"<drug_analysis_request>\n"
        f"Please analyze the effectiveness of {drug_name} for treating {disease}. "
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
    except Exception as e:
        logger.error(f"Error analyzing drug effectiveness: {e}")
        raise
