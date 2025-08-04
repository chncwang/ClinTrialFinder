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
    logging.getLogger(logger_name).setLevel(logging.INFO)
