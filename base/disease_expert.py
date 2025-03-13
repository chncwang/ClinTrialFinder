import json
import logging
from typing import Dict, List, Tuple

from base.gpt_client import GPTClient
from base.utils import parse_json_response

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
        raise RuntimeError(f"Error extracting disease from clinical record: {e}")


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

        def validate_categories(data: Dict) -> bool:
            if not isinstance(data, dict) or "categories" not in data:
                return False
            categories = data["categories"]
            if not isinstance(categories, list):
                return False
            return True

        # Parse and validate the response
        response_data, total_cost = parse_json_response(
            completion,
            expected_type=dict,
            gpt_client=gpt_client,
            cost=cost,
            validation_func=validate_categories,
        )

        # Process the categories
        categories = response_data["categories"]
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
            if not any(category.lower() == general.lower() for general in too_general)
        ]

        logger.info(
            f"Parent disease categories for {disease_name}: {filtered_categories}"
        )
        return filtered_categories, total_cost

    except Exception as e:
        raise RuntimeError(
            f"Error getting parent disease categories for {disease_name}: {e}"
        )


def extract_conditions_from_record(
    clinical_record: str, gpt_client: GPTClient
) -> Tuple[List[str], float]:
    """
    Extracts relevant clinical conditions from a clinical record using GPT-4o.
    Returns a flat array of extracted values that are typically matched in clinical trials.

    Parameters:
    - clinical_record (str): The patient's clinical record text
    - gpt_client (GPTClient): Initialized GPT client for making API calls

    Returns:
    - tuple[list[str], float]: A tuple containing:
        - List of extracted values (conditions, demographics, clinical status, etc.)
        - Cost of the API call
    """
    prompt = (
        "Extract relevant clinical conditions from the following clinical record that are typically matched in clinical trials. "
        "Return the results as a JSON array of strings, where each string represents a piece of information. "
        "Focus on conditions and statuses that are critical for clinical trial matching.\n\n"
        "Include:\n"
        "- Key medical conditions\n"
        "- Essential patient demographics (age, gender)\n"
        "- Important clinical status (performance score, stage, etc.)\n\n"
        "Example format:\n"
        "[\n"
        '  "The patient has Type 2 Diabetes.",\n'
        '  "The patient is 65 years old.",\n'
        '  "The patient is male.",\n'
        '  "The patient has an ECOG PS of 1.",\n'
        '  "The patient is at Stage III."\n'
        "]\n\n"
        f"Clinical Record:\n{clinical_record}\n\n"
        "Return only the JSON array, no additional text or explanation."
    )

    try:
        completion, cost = gpt_client.call_gpt(
            prompt=prompt,
            system_role="You are a medical expert specialized in extracting structured patient information from clinical records.",
            temperature=0.1,
            model="gpt-4o",  # Use GPT-4o model
        )
        if completion is None:
            logger.error("Failed to extract information from clinical record")
            return [], cost

        # Parse and validate the response
        result, total_cost = parse_json_response(
            completion, expected_type=list, gpt_client=gpt_client, cost=cost
        )

        # Convert all items to strings and filter out empty strings
        extracted_values = [
            str(item).strip()
            for item in result
            if isinstance(item, (str, int, float)) and str(item).strip()
        ]

        logger.info(f"Extracted {len(extracted_values)} values from clinical record")
        return extracted_values, total_cost

    except Exception as e:
        raise RuntimeError(f"Error extracting information from clinical record: {e}")
