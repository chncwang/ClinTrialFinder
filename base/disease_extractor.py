import logging
from typing import Tuple

from .gpt_client import GPTClient

# Configure logging
logger = logging.getLogger(__name__)


def extract_disease_from_record(
    clinical_record: str, gpt_client: GPTClient
) -> Tuple[str | None, float]:
    """
    Extracts the primary disease or condition from a clinical record using GPT.

    Parameters:
    - clinical_record (str): The patient's clinical record text
    - gpt_client (GPTClient): Initialized GPT client for making API calls

    Returns:
    - tuple[str | None, float]: A tuple containing the extracted disease name (or None if not found)
                               and the cost of the API call
    """
    prompt = (
        "Extract the primary disease or medical condition from the following clinical record. "
        "Return only the disease name without any additional text or explanation.\n\n"
        f"Clinical Record:\n{clinical_record}"
    )

    try:
        completion, cost = gpt_client.call_gpt(
            prompt=prompt,
            system_role="You are a medical expert focused on identifying primary medical conditions.",
            temperature=0.1,
        )
        if completion is None:
            logger.error("Failed to extract disease from clinical record")
            return None, cost

        disease_name = completion.strip()
        logger.info(f"Extracted disease name: {disease_name}")
        return disease_name, cost
    except Exception as e:
        logger.error(f"Error extracting disease from clinical record: {e}")
        return None, 0.0
