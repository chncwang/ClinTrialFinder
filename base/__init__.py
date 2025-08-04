"""Base package for clinical trial analysis tools."""

import logging

# Configure logging for the base module - set to INFO level to reduce debug logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)

# Set the log level for all loggers in the base module to INFO
base_logger = logging.getLogger('base')
base_logger.setLevel(logging.INFO)

# Explicitly set INFO level for specific loggers that have debug statements
specific_loggers = [
    'base.gpt_client',
    'base.prompt_cache', 
    'base.pricing',
    'base.trial_expert',
    'base.perplexity',
    'base.drug_analyzer',
    'base.utils',
    'base.disease_expert'
]

for logger_name in specific_loggers:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    # Ensure the logger doesn't inherit a lower level from its parent
    logger.propagate = True

# Also set the root logger to INFO to ensure no debug messages get through
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

def ensure_info_logging():
    """Ensure all base module loggers are set to INFO level."""
    # Re-set the base logger to INFO
    base_logger = logging.getLogger('base')
    base_logger.setLevel(logging.INFO)
    
    # Re-set all specific loggers to INFO
    for logger_name in specific_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        logger.propagate = True
    
    # Ensure root logger is at INFO
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

# Call the function to ensure logging is set to INFO when module is imported
ensure_info_logging()
