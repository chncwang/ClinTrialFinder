import json
import logging
from typing import List, Tuple

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


def get_parent_disease_categories(
    disease_name: str, gpt_client: GPTClient
) -> Tuple[List[str], float]:
    """
    Returns an array of relevant broader disease categories that include the input disease,
    focusing on specific and clinically relevant categories rather than overly general ones.

    For example, if the input is "glioblastoma multiforme", the output might include
    ["brain tumor", "central nervous system cancer"] but would exclude very general categories
    like "cancer", "malignancy", or "oncological disorder".

    Parameters:
    - disease_name (str): The specific disease or condition name
    - gpt_client (GPTClient): Initialized GPT client for making API calls

    Returns:
    - tuple[list[str], float]: A tuple containing the list of parent disease categories
                              and the cost of the API call
    """
    prompt = (
        "For the following specific disease or condition, provide a list of 2-3 broader disease categories "
        "that would include this condition. Focus on specific, clinically relevant categories, not overly general ones.\n\n"
        "For example:\n"
        "- For 'glioblastoma multiforme', good categories would be 'brain tumor' and 'central nervous system cancer'\n"
        "- For 'acute myeloid leukemia', good categories would be 'leukemia' and 'hematologic malignancy'\n\n"
        "AVOID overly general categories like 'cancer', 'malignancy', 'oncological disorder', 'disease', etc.\n\n"
        f"Disease: {disease_name}\n\n"
        "Return your response as a JSON object with a single key 'categories' containing an array of strings.\n"
        'Example format: {"categories": ["specific category", "broader category"]}\n'
        "Do not include any explanations or additional text, just the JSON object."
    )

    try:
        completion, cost = gpt_client.call_gpt(
            prompt=prompt,
            system_role="You are a medical expert with comprehensive knowledge of disease classification and taxonomy.",
            temperature=0.1,
        )
        if completion is None:
            logger.error(f"Failed to get parent disease categories for {disease_name}")
            return [], cost

        # Parse the response as JSON
        try:
            # Clean up the response to ensure it's valid JSON
            cleaned_response = completion.strip()

            # Parse the JSON response
            response_data = json.loads(cleaned_response)

            # Extract the categories
            if not isinstance(response_data, dict) or "categories" not in response_data:
                logger.error(
                    f"Response is not in the expected format: {cleaned_response}"
                )
                return [], cost

            categories = response_data["categories"]
            if not isinstance(categories, list):
                logger.error(f"Categories is not a list: {categories}")
                return [], cost

            # Ensure all items are strings
            categories = [str(item) for item in categories]

            # Filter out overly general categories
            too_general = [
                "cancer",
                "malignancy",
                "oncological disorder",
                "disease",
                "disorder",
                "condition",
                "illness",
            ]
            filtered_categories = [
                category
                for category in categories
                if not any(
                    category.lower() == general.lower() for general in too_general
                )
            ]

            logger.info(
                f"Parent disease categories for {disease_name}: {filtered_categories}"
            )
            return filtered_categories, cost

        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON response: {e}")
            logger.error(f"Raw response: {completion}")
            return [], cost
        except Exception as e:
            logger.error(f"Error processing parent disease categories response: {e}")
            logger.error(f"Raw response: {completion}")
            return [], cost

    except Exception as e:
        logger.error(f"Error getting parent disease categories for {disease_name}: {e}")
        return [], cost
