from datetime import datetime
from typing import Any, Dict, List, Tuple

from loguru import logger
from base.gpt_client import GPTClient
from base.utils import parse_json_response, read_input_file


def extract_disease_from_record(
    clinical_record: str, gpt_client: GPTClient
) -> Tuple[str, float]:
    """
    Extracts the primary disease or condition from a clinical record using GPT.

    Parameters:
    - clinical_record (str): The patient's clinical record text
    - gpt_client (GPTClient): Initialized GPT client for making API calls

    Returns:
    - tuple[str | None, float]: A tuple containing the extracted disease name (or None if not found) and the cost of the API call
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
            model="gpt-4.1",
            temperature=0.1,
            response_format={"type": "json_object"},
        )

        def validate_categories(data: Dict[str, Any]) -> bool:
            if "categories" not in data:
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
        response_data: Dict[str, List[str]] = response_data

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
    clinical_record_file: str, gpt_client: GPTClient, refresh_cache: bool = False
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

        # Use the existing function to extract conditions from content
        return extract_conditions_from_content(
            clinical_record=clinical_record,
            gpt_client=gpt_client,
            refresh_cache=refresh_cache,
        )

    except Exception as e:
        logger.error(f"Error processing clinical record: {e}")
        raise

    return []


def extract_conditions_from_content(
    clinical_record: str, gpt_client: GPTClient, refresh_cache: bool = False
) -> List[str]:
    """
    Extract the patient's clinical history directly from clinical record content in chronological order.

    Parameters:
    - clinical_record (str): The clinical record content
    - gpt_client (GPTClient): Initialized GPT client for making API calls
    - refresh_cache (bool): Whether to refresh the GPT cache

    Returns:
    - List[str]: List of clinical history items in chronological order
    """
    try:
        # Get current time for relative time calculations
        current_time = datetime.now()
        current_time_str = current_time.strftime("%Y-%m-%d")

        # Extract clinical history in chronological order
        prompt = (
            f"Extract at most 20 most important key conditions AND demographic features from the patient's clinical record for clinical trials filtering. "
            f"Use relative time references based on today's date ({current_time_str}). "
            "Present the results as a JSON array of strings. Include relevant demographic information (age, sex, etc.) as the first item.\n\n"
            "Focus on:\n"
            "- Demographic features (age, sex, etc.)\n"
            "- Major medical conditions\n"
            "- Significant treatments and their status\n"
            "- Current clinical status\n"
            "- Other conditions relevant to eligibility\n\n"
            "Example original record:\n"
            "<original record>\n"
            "Patient: Jane Doe\n"
            "DOB: 05/15/1958 (65 years old)\n"
            "Sex: Female\n"
            "Medical History:\n"
            "- Type 2 Diabetes Mellitus diagnosed in 2010, currently managed with Metformin 1000mg BID\n"
            "- Non-small cell lung cancer (Stage 3) diagnosed in April 2023, currently on cycle 4 of chemotherapy\n"
            "- Liver metastasis detected in June 2023, stable on recent imaging\n"
            "- Hypertension, well-controlled on Lisinopril 20mg daily\n"
            "- Chronic kidney disease (Stage 2) with eGFR 75 mL/min\n"
            "- Myocardial infarction in 2019, treated with stent placement\n"
            "- Osteoarthritis of the right knee, managed with NSAIDs as needed\n"
            "- Hypothyroidism, on levothyroxine 100mcg daily\n"
            "- COPD with history of 2 exacerbations in the past year, uses albuterol inhaler\n\n"
            "</original record>\n"
            "Given the above record and the current date of 2023-12-01, the result should be:\n"
            "[\n"
            '  "65-year-old female",\n'
            '  "Type 2 Diabetes Mellitus diagnosed 13 years ago, managed with Metformin",\n'
            '  "Non-small cell lung cancer (Stage 3) diagnosed 8 months ago, currently on chemotherapy",\n'
            '  "Liver metastasis detected 6 months ago, stable on recent imaging",\n'
            '  "Hypertension, well-controlled on Lisinopril",\n'
            '  "Chronic kidney disease (Stage 2) with eGFR 75 mL/min",\n'
            '  "Myocardial infarction 4 years ago, treated with stent placement",\n'
            '  "Osteoarthritis of the right knee, managed with NSAIDs as needed",\n'
            '  "Hypothyroidism, on levothyroxine",\n'
            '  "COPD with history of 2 exacerbations in the past year, uses albuterol inhaler"\n'
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
            model="gpt-4.1",
            refresh_cache=refresh_cache,
        )

        if completion:
            history, _ = parse_json_response(
                completion, expected_type=list, gpt_client=gpt_client, cost=cost
            )
            history: List[str] = history

            # Process the history to ensure all times are relative
            history = convert_absolute_to_relative_time(
                history, current_time, gpt_client
            )

            # Log history
            logger.info(f"Extracted {len(history)} clinical history items:")
            for item in history:
                logger.info(f"- {item}")

            return history

    except Exception as e:
        logger.error(f"Error processing clinical record content: {e}")
        raise

    return []


def has_absolute_time_reference(item: str, gpt_client: GPTClient) -> bool:
    """
    Check if a clinical history item contains absolute time references (years, dates) using GPT.

    Parameters:
    - item (str): The clinical history item to check
    - gpt_client (GPTClient): Initialized GPT client for making API calls

    Returns:
    - bool: True if the item contains absolute time references, False otherwise
    """
    prompt = (
        "Determine if the following text contains any absolute time references (specific years, dates, or month-year combinations).\n\n"
        f'Text: "{item}"\n\n'
        "Answer with only a single word - either 'yes' or 'no'."
    )

    try:
        completion, _ = gpt_client.call_gpt(
            prompt=prompt,
            system_role="You are a text analysis expert specialized in identifying time references in medical text.",
            temperature=0.0,
        )

        response = completion.strip().lower()
        return response == "yes"

    except Exception as e:
        logger.error(f"Error checking for absolute time references: {e}")
        return False


def convert_absolute_to_relative_time(
    history_items: List[str], current_time: datetime, gpt_client: GPTClient
) -> List[str]:
    """
    Process clinical history items to convert any absolute time references to relative ones.

    Parameters:
    - history_items (List[str]): List of clinical history items
    - current_time (datetime): Current datetime for relative time calculation
    - gpt_client (GPTClient): Initialized GPT client for making API calls

    Returns:
    - List[str]: Updated list with all time references in relative format
    """
    current_date_str = current_time.strftime("%Y-%m-%d")
    updated_items: List[str] = []

    for item in history_items:
        if has_absolute_time_reference(item, gpt_client):
            updated_item, _ = convert_item_to_relative_time(
                item, current_date_str, gpt_client
            )
            updated_items.append(updated_item)
        else:
            updated_items.append(item)

    return updated_items


def convert_item_to_relative_time(
    item: str, current_date: str, gpt_client: GPTClient
) -> Tuple[str, float]:
    """
    Convert a single clinical history item with absolute time references to use relative time.

    Parameters:
    - item (str): The clinical history item to convert
    - current_date (str): Current date string in YYYY-MM-DD format
    - gpt_client (GPTClient): Initialized GPT client for making API calls

    Returns:
    - Tuple[str, float]: Updated item with relative time and the cost of the API call
    """
    prompt = (
        f"Convert the following clinical history item to use relative time references (like 'X years ago', 'X months ago') "
        f"instead of absolute dates or years. Today's date is {current_date}.\n\n"
        f'Original: "{item}"\n\n'
        "Return only the converted text without any additional explanation or quotation marks."
    )

    try:
        completion, cost = gpt_client.call_gpt(
            prompt=prompt,
            system_role="You are a clinical data specialist converting absolute time references to relative ones.",
            temperature=0.1,
        )

        updated_item = completion.strip().strip("\"'")
        logger.info(f"Converted time reference: '{item}' → '{updated_item}'")
        return updated_item, cost

    except Exception as e:
        logger.error(f"Error converting time references: {e}")
        return item, 0.0


def is_oncology_disease(disease_name: str) -> bool:
    """
    Returns True if the disease name is related to oncology (cancer), otherwise False.

    Parameters:
    - disease_name (str): The name of the disease to classify

    Returns:
    - bool: True if the disease is oncology-related, False otherwise
    """
    oncology_keywords = [
        'cancer', 'carcinoma', 'sarcoma', 'leukemia', 'lymphoma', 'tumor', 'neoplasm',
        'onco', 'malignancy', 'melanoma', 'blastoma', 'myeloma', 'metastatic'
    ]
    disease_name_lower = disease_name.lower()
    return any(kw in disease_name_lower for kw in oncology_keywords)
