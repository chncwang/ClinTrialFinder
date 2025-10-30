import json
import os
import sys
from typing import Any, Callable, Dict, List, Tuple, TypeVar, Optional
import logging

logger = logging.getLogger(__name__)

from base.gpt_client import GPTClient

ReturnType = TypeVar("ReturnType")


def get_api_key(args_key: Optional[str], env_var: str = "OPENAI_API_KEY") -> str:
    """
    Get API key from command line arguments or environment variable.
    
    Args:
        args_key: API key from command line arguments
        env_var: Name of the environment variable to check (default: "OPENAI_API_KEY")
    
    Returns:
        API key as a string
        
    Raises:
        ValueError: If API key is not found
    """
    api_key = args_key or os.getenv(env_var)
    if not api_key:
        error_msg = (
            f"API key not found. Please provide it via command line argument "
            f"or set the {env_var} environment variable"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    return api_key


def read_input_file(file_path: str) -> str:
    """Read text content from a specified file path."""
    try:
        with open(file_path, "r") as file:
            content = file.read().strip()
            logger.info(f"read_input_file: Read input file from {file_path}")
            logger.info(f"read_input_file: File Content: {content}")
            return content
    except FileNotFoundError:
        logger.error(f"read_input_file: Input file not found: {file_path}")
        sys.exit(1)
    except Exception as read_error:
        logger.error(f"read_input_file: Error reading input file: {read_error}")
        sys.exit(1)


def parse_json_response(
    response: str,
    expected_type: type[ReturnType],
    gpt_client: GPTClient,
    cost: float = 0.0,
    validation_func: Callable[[Any], bool] | None = None,
) -> Tuple[ReturnType, float]:
    """
    Parse a JSON response and attempt to correct it if invalid.

    Parameters:
    - response: The JSON string to parse
    - expected_type: The expected type of the parsed result
    - gpt_client: GPT client for correction attempts
    - cost: Current cost of API calls
    - validation_func: Optional function to validate the parsed result

    Returns:
    - Tuple of (parsed result, total cost including corrections)

    Raises:
    - ValueError: If parsing and correction fail
    """
    try:
        # Clean up the response
        cleaned_response = response.strip()

        # First attempt to parse
        try:
            result = json.loads(cleaned_response)
            if not isinstance(result, expected_type):
                raise ValueError(f"Response is not of type {expected_type}")
            if validation_func and not validation_func(result):
                raise ValueError("Response failed validation")
            return result, cost

        except (json.JSONDecodeError, ValueError) as json_error:
            logger.warning(f"parse_json_response: Invalid JSON structure detected: {json_error}")
            logger.warning(
                f"parse_json_response: Attempting to correct response: {cleaned_response}"
            )

            # Attempt correction using GPT-4.1-mini
            correction_prompt = (
                f"Please convert the following text into a valid JSON {expected_type.__name__}. "
                f"Ensure no additional text or formatting is included:\n{cleaned_response}"
            )
            completion, correction_cost = gpt_client.call_gpt(
                prompt=correction_prompt,
                system_role="You are a JSON expert.",
                model="gpt-4.1-mini",
                temperature=0.0,
            )
            total_cost = cost + correction_cost

            try:
                result = json.loads(completion)
                if not isinstance(result, expected_type):
                    raise ValueError(
                        f"Corrected response is not of type {expected_type}"
                    )
                if validation_func and not validation_func(result):
                    raise ValueError("Corrected response failed validation")
                return result, total_cost

            except (json.JSONDecodeError, ValueError) as correction_error:
                logger.error(f"parse_json_response: Failed to correct JSON: {correction_error}")
                logger.error(
                    f"parse_json_response: Failed to convert to valid JSON: {completion}"
                )
                raise ValueError(
                    f"Failed to convert to valid JSON: {correction_error}. Response content: {completion}"
                )

    except Exception as parse_error:
        raise ValueError(f"Error parsing JSON response: {parse_error}. Raw response: {response}")


def load_json_list_file(file_path: str) -> List[Dict[str, Any]]:
    """Load and parse a JSON file containing a list of objects."""
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"load_json_list_file: File '{file_path}' not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        logger.error(
            f"load_json_list_file: File '{file_path}' is not a valid JSON file."
        )
        sys.exit(1)


def save_json_list_file(data: List[Dict[str, Any]], output_path: str, file_type: str = "results"):
    """Save a list of dictionaries to a JSON file.

    Args:
        data: The list of dictionaries to save
        output_path: The path to save to
        file_type: Type of file being saved (for logging purposes)
    """
    try:
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(
            f"save_json_list_file: {file_type.capitalize()} saved to {output_path}"
        )
    except Exception as save_error:
        logger.error(f"save_json_list_file: Error saving {file_type}: {str(save_error)}")
        sys.exit(1)


def create_gpt_client(
    api_key: str,
    cache_size: int = 100000,
    temperature: float = 0.1,
    max_retries: int = 3
) -> GPTClient:
    """
    Create a GPTClient with commonly used parameters.
    
    Args:
        api_key: OpenAI API key
        cache_size: Maximum number of cached responses to keep (default: 100000)
        temperature: Sampling temperature (default: 0.1)
        max_retries: Maximum number of retry attempts (default: 3)
    
    Returns:
        Configured GPTClient instance
    """
    return GPTClient(
        api_key=api_key,
        cache_size=cache_size,
        temperature=temperature,
        max_retries=max_retries,
    )
