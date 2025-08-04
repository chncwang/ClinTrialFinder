"""
Logging configuration utility for ClinTrialFinder.

This module provides centralized logging configuration management
by reading settings from a YAML configuration file.
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from loguru import logger


class LoggingConfig:
    """Centralized logging configuration manager."""
    
    def __init__(self, config_file: str = "config/logging_config.yaml"):
        """
        Initialize the logging configuration.
        
        Args:
            config_file: Path to the YAML configuration file
        """
        self.config_file = config_file
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        config_path = Path(self.config_file)
        
        if not config_path.exists():
            # Return default configuration if file doesn't exist
            return self._get_default_config()
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Warning: Could not load logging config from {self.config_file}: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration if config file is not available."""
        return {
            "global": {
                "level": "INFO",
                "format": "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
                "colorize": True
            },
            "console": {
                "enabled": True,
                "level": "INFO",
                "format": "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
                "colorize": True
            },
            "file": {
                "enabled": True,
                "level": "INFO",
                "format": "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
                "directory": "logs",
                "rotation": "10 MB",
                "retention": "30 days",
                "compression": "gz"
            },
            "loggers": {
                "disabled": [
                    "base.gpt_client",
                    "httpx",
                    "httpcore", 
                    "openai",
                    "urllib3",
                    "requests"
                ],
                "enabled": [
                    "base.prompt_cache",
                    "base.disease_expert",
                    "base.trial_expert",
                    "base.clinical_trial",
                    "base.utils"
                ]
            }
        }
    
    def setup_logging(
        self,
        script_name: str,
        log_level: Optional[str] = None,
        **kwargs: Any
    ) -> str:
        """
        Setup logging configuration for a specific script.
        
        Args:
            script_name: Name of the script (e.g., 'filter_specific_trial')
            log_level: Override log level from config
            nct_id: NCT ID for scripts that need it in filename
            **kwargs: Additional parameters for file naming
            
        Returns:
            Path to the log file
        """
        # Get script-specific configuration
        script_config = self.config.get("scripts", {}).get(script_name, {})
        
        # Determine log level (script-specific > parameter > global)
        level = log_level or script_config.get("level") or self.config["global"]["level"]
        
        # Create logs directory
        logs_dir = Path(self.config["file"]["directory"])
        logs_dir.mkdir(exist_ok=True)
        
        # Generate log filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_prefix = script_config.get("file_prefix", script_name)
        
        log_filename = f"{file_prefix}_{timestamp}.log"
        
        log_file = logs_dir / log_filename
        
        # Remove existing handlers
        logger.remove()
        
        # Get logger-specific levels
        logger_levels = self.config.get("loggers", {}).get("levels", {})
        
        # Set global level to DEBUG if we have any DEBUG loggers, so they can reach the filter
        if any(lvl.upper() == "DEBUG" for lvl in logger_levels.values()):
            global_level = "DEBUG"
        else:
            global_level = level.upper()
        
        # Add console handler if enabled
        if self.config["console"]["enabled"]:
            logger.add(
                sys.stdout,
                format=self.config["console"]["format"],
                level=global_level,
                colorize=self.config["console"]["colorize"],
                filter=lambda record: self._should_log_record(record, logger_levels, level.upper())
            )
        
        # Add file handler if enabled
        if self.config["file"]["enabled"]:
            logger.add(
                str(log_file),
                format=self.config["file"]["format"],
                level=global_level,
                rotation=self.config["file"]["rotation"],
                retention=self.config["file"]["retention"],
                compression=self.config["file"]["compression"],
                filter=lambda record: self._should_log_record(record, logger_levels, level.upper())
            )
        
        # Configure logger enable/disable settings
        self._configure_loggers()
        
        return str(log_file)
    
    def _should_log_record(self, record: Any, logger_levels: Dict[str, str], default_level: str) -> bool:
        """
        Determine if a log record should be logged based on logger-specific levels.
        
        Args:
            record: Loguru log record
            logger_levels: Dictionary mapping logger names to their levels
            default_level: Default log level for loggers not in logger_levels
            
        Returns:
            bool: True if the record should be logged, False otherwise
        """
        import logging
        
        # Get the logger name from the record
        logger_name = record["name"]
        
        # Get the level for this specific logger, or use default
        logger_level = logger_levels.get(logger_name, default_level)
        
        # Convert level strings to numeric values for comparison
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }
        
        logger_level_num = level_map.get(logger_level.upper(), logging.INFO)
        record_level_num = record["level"].no
        
        # Only log if the record's level is >= the logger's configured level
        return record_level_num >= logger_level_num
    
    def _configure_loggers(self):
        """Configure which loggers are enabled/disabled."""
        loggers_config = self.config.get("loggers", {})
        
        # Disable specified loggers
        for logger_name in loggers_config.get("disabled", []):
            logger.disable(logger_name)
        
        # Enable specified loggers
        for logger_name in loggers_config.get("enabled", []):
            logger.enable(logger_name)
    
    def get_script_config(self, script_name: str) -> Dict[str, Any]:
        """Get configuration for a specific script."""
        return self.config.get("scripts", {}).get(script_name, {})
    
    def reload_config(self):
        """Reload configuration from file."""
        self.config = self._load_config()


# Global instance for easy access
logging_config = LoggingConfig()


def setup_logging(
    script_name: str,
    log_level: Optional[str] = None,
    **kwargs: Any
) -> str:
    """
    Convenience function to setup logging for a script.
    
    Args:
        script_name: Name of the script
        log_level: Override log level
        nct_id: NCT ID for filename (if applicable)
        **kwargs: Additional parameters
        
    Returns:
        Path to the log file
    """
    return logging_config.setup_logging(script_name, log_level, **kwargs) 