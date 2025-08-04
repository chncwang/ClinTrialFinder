"""Base package for clinical trial analysis tools."""

import logging

# Configure logging for the base module - set to INFO level to reduce debug logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Set the log level for all loggers in the base module to INFO
base_logger = logging.getLogger('base')
base_logger.setLevel(logging.INFO)
