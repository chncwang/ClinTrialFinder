import logging
import logging.config
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_logging(script_name: str, log_level: Optional[str] = None) -> str:
    """
    Setup standard Python logging for a script.
    
    Args:
        script_name: Name of the script (e.g., 'filter_trials')
        log_level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Path to the log file
    """
    # Set default log level
    level = log_level or "INFO"
    
    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Generate log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{script_name}_{timestamp}.log"
    log_file = logs_dir / log_filename
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ],
        force=True  # Override any existing configuration
    )
    
    # Disable noisy loggers
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    logging.getLogger('openai').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('base.pricing').setLevel(logging.WARNING)
    logging.getLogger('base.gpt_client').setLevel(logging.WARNING)
    logging.getLogger('base.prompt_cache').setLevel(logging.WARNING)
    
    return str(log_file)