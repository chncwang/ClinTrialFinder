import logging
import sys

logger = logging.getLogger(__name__)


def read_input_file(file_path: str) -> str:
    """Read text content from a specified file path."""
    try:
        with open(file_path, "r") as file:
            content = file.read().strip()
            logger.info(f"Read input file from {file_path}")
            logger.info(f"File Content: {content[:200]}...")
            return content
    except FileNotFoundError:
        logger.error(f"Input file not found: {file_path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error reading input file: {e}")
        sys.exit(1)
