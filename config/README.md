# Logging Configuration

This directory contains the centralized logging configuration for the ClinTrialFinder project.

## Overview

The logging system uses a YAML configuration file (`logging_config.yaml`) to centrally manage logging settings for all scripts in the project. This provides:

- **Consistency**: All scripts use the same logging format and settings
- **Flexibility**: Easy to modify logging behavior without changing code
- **Maintainability**: Centralized configuration management
- **Script-specific settings**: Different configurations for different types of scripts

## Configuration File

The `logging_config.yaml` file contains the following sections:

### Global Settings
- `level`: Default log level for all scripts
- `format`: Default log message format
- `colorize`: Whether to use colored output

### Console Output
- `enabled`: Whether to log to console
- `level`: Console log level
- `format`: Console log format (supports colors)
- `colorize`: Whether to use colored console output

### File Output
- `enabled`: Whether to log to files
- `level`: File log level
- `format`: File log format
- `directory`: Directory to store log files
- `rotation`: When to rotate log files (e.g., "10 MB", "1 day")
- `retention`: How long to keep log files (e.g., "30 days")
- `compression`: Compression format for rotated logs

### Script-Specific Settings
Each script can have its own configuration:
- `file_prefix`: Prefix for log filenames
- `level`: Script-specific log level

### Logger Management
- `disabled`: List of loggers to disable (e.g., noisy external libraries)
- `enabled`: List of loggers to explicitly enable

## Usage

### Basic Usage

```python
from base.logging_config import setup_logging

# Setup logging for a script
log_file = setup_logging("filter_trials", "INFO")
logger.info("Script started")
```



### Advanced Usage

```python
from base.logging_config import logging_config

# Get script-specific configuration
config = logging_config.get_script_config("filter_trials")

# Reload configuration from file
logging_config.reload_config()
```

## Example Script

See `scripts/example_logging_usage.py` for a complete example of how to use the logging configuration system.

## Migration from Old Logging

To migrate existing scripts to use the new logging configuration:

1. Import the setup_logging function:
   ```python
   from base.logging_config import setup_logging
   ```

2. Replace the existing logging setup with:
   ```python
   log_file = setup_logging("script_name", log_level)
   ```

3. Remove the old logging configuration code.

## Configuration Examples

### Development Configuration
```yaml
global:
  level: DEBUG
console:
  enabled: true
  level: DEBUG
file:
  enabled: true
  level: DEBUG
```

### Production Configuration
```yaml
global:
  level: INFO
console:
  enabled: false
file:
  enabled: true
  level: INFO
  rotation: "100 MB"
  retention: "90 days"
```

### Debug Configuration
```yaml
global:
  level: DEBUG
console:
  enabled: true
  level: DEBUG
file:
  enabled: true
  level: DEBUG
loggers:
  enabled:
    - "base.gpt_client"  # Enable all loggers for debugging
``` 