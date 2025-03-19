import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple

from base.gpt_client import GPTClient
from base.utils import parse_json_response, read_input_file

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
            response_format={"type": "json_object"},
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
            model="gpt-4o",
            temperature=0.1,
            response_format={"type": "json_object"},
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
    clinical_record_file: str, gpt_client: GPTClient
) -> List[str]:
    """
    Extract the patient's clinical history from a clinical record file in chronological order.

    Parameters:
    - clinical_record_file (str): Path to the clinical record file
    - gpt_client (GPTClient): Initialized GPT client for making API calls

    Returns:
    - List[str]: List of clinical history items in chronological order
    """
    try:
        # Read the clinical record
        clinical_record = read_input_file(clinical_record_file)
        logger.info(f"Read clinical record from {clinical_record_file}")

        # Get current time for relative time calculations
        current_time = datetime.now()
        current_time_str = current_time.strftime("%Y-%m-%d")

        # Extract clinical history in chronological order
        prompt = (
            f"Extract at most 10 most important key conditions AND demographic features from the patient's clinical record for clinical trials filtering. "
            f"Use relative time references based on today's date ({current_time_str}). "
            "Present the results as a JSON array of strings. Include relevant demographic information (age, sex, etc.) as the first item.\n\n"
            "Focus on:\n"
            "- Demographic features (age, sex, etc.)\n"
            "- Major medical conditions\n"
            "- Significant treatments and their status\n"
            "- Current clinical status\n"
            "- Other conditions relevant to eligibility\n\n"
            "Example format:\n"
            "[\n"
            '  "65-year-old female",\n'
            '  "Type 2 Diabetes, stable with medication",\n'
            '  "Stage 3 lung cancer with ongoing chemotherapy",\n'
            '  "Liver metastasis under monitoring",\n'
            '  "Hypertension managed with medication",\n'
            '  "Chronic kidney disease stage 2",\n'
            '  "Myocardial infarction 4 years ago",\n'
            '  "Osteoarthritis of the knee",\n'
            '  "Hypothyroidism on levothyroxine",\n'
            '  "COPD with occasional exacerbations",\n'
            '  "<other conditions relevant to eligibility>",\n'
            "]\n\n"
            "Guidelines:\n"
            "1. Always include demographic information as the first item\n"
            "2. Focus on the condition and its current status\n"
            "3. Include relevant treatment information if applicable\n"
            "4. Use relative time references (e.g., '2 years ago' instead of '2021')\n"
            "5. Keep descriptions concise and medically relevant\n"
            "6. Include any other conditions relevant to trial eligibility\n\n"
            f"Clinical Record:\n{clinical_record}\n\n"
            "Return only the JSON array, without any additional text or explanation."
        )

        completion, cost = gpt_client.call_gpt(
            prompt=prompt,
            system_role="You are a medical expert specialized in extracting clinical history from medical records.",
            temperature=0.1,
            model="gpt-4o",
            response_format={"type": "json_object"},
        )

        if completion:
            history, _ = parse_json_response(
                completion, expected_type=list, gpt_client=gpt_client, cost=cost
            )

            # Log history
            logger.info(f"Extracted {len(history)} clinical history items:")
            for item in history:
                logger.info(f"- {item}")

            return history

    except Exception as e:
        logger.error(f"Error processing clinical record: {e}")
        raise

    return []
