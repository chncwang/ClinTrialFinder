import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from base.clinical_trial import ClinicalTrial
from base.gpt_client import GPTClient
from base.perplexity import PerplexityClient
from base.trial_expert import RecommendationLevel, analyze_drugs_and_get_recommendation
from base.utils import read_input_file

# Configure logging
# Create logs directory if it doesn't exist
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

# Set up timestamp for log files
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"logs/analyze_filtered_trials_{timestamp}.log"

# Configure the root logger first
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()  # Output to console
    ],
    force=True  # Force reconfiguration to ensure our settings take effect
)

# Get the module logger
logger = logging.getLogger(__name__)

# Ensure the root logger level is set to INFO
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Configure base.trial_expert logger
trial_expert_logger = logging.getLogger("base.trial_expert")
trial_expert_logger.setLevel(logging.INFO)
trial_expert_logger.propagate = True  # Ensure logs propagate to root logger

# Make sure sys.stdout is flushed after each write
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore

# Log the filename being used
logger.info(f"All logs will be written to: {os.path.abspath(log_filename)}")


def process_trials_file(
    filename: str, 
    gpt_client: "GPTClient", 
    perplexity_client: "PerplexityClient", 
    clinical_record: str, 
    output_filename: str | None = None
) -> None:
    """Read and process the trials JSON file."""
    updated_trials: list[dict[str, Any]] = []
    trials: list[dict[str, Any]] = []
    total_cost: float = 0.0
    
    try:
        with open(filename, "r") as f:
            trials = json.load(f)

        if output_filename is None:
            output_filename = f"analyzed_{os.path.basename(filename)}"
            
        logger.info(f"Starting to process {len(trials)} trials from {filename}")

        # Prepare a list to store updated trial data
        updated_trials = []

        for index, trial_dict in enumerate(trials, start=1):
            trial = ClinicalTrial(trial_dict)
            logger.info(f"Processing trial {index}/{len(trials)}: {trial}")
            # Analyze the trial
            recommendation: RecommendationLevel
            reason: str
            drug_analyses: dict[str, str]
            cost: float
            recommendation, reason, drug_analyses, cost = (
                analyze_drugs_and_get_recommendation(
                    clinical_record,
                    trial=trial,
                    perplexity_client=perplexity_client,
                    gpt_client=gpt_client,
                )
            )
            total_cost += cost
            logger.info(f"Recommendation: {recommendation}")
            logger.info(f"Reason: {reason}")
            logger.info(f"Drug Analyses: {drug_analyses}")
            logger.info(f"Cost for this trial: ${cost:.6f}")
            logger.info(f"Total cost so far: ${total_cost:.6f}")

            # Add recommendation, reason, and drug analysis to the trial data
            trial_dict["recommendation_level"] = str(recommendation)
            trial_dict["reason"] = reason
            trial_dict["drug_analysis"] = drug_analyses
            updated_trials.append(trial_dict)

        # Write the analyzed trials to a new JSON file
        with open(output_filename, "w") as f:
            json.dump(updated_trials, f, indent=4)

        logger.info(
            f"Finished processing all trials. Output written to {output_filename}"
        )

    except FileNotFoundError:
        logger.error(f"File not found: {filename}")
        raise
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from {filename}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise
    
    finally:
        # Always log final summary
        logger.info("=" * 50)
        logger.info("Filtering process completed")
        logger.info(f"Total trials processed: {len(trials)}")
        logger.info(f"Eligible trials found: {len(updated_trials)}")
        logger.info(f"Total API cost: ${total_cost:.2f}")
        logger.info(f"Results saved to: {output_filename if output_filename else 'N/A'}")
        logger.info(f"Log file: {os.path.abspath(log_filename)}")
        logger.info("=" * 50)
        
        # Force flush all handlers
        for handler in logging.getLogger().handlers:
            handler.flush()
            
        # Also flush stdout/stderr explicitly
        sys.stdout.flush()
        sys.stderr.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process clinical trials from a JSON file."
    )
    parser.add_argument(
        "trials_json_file", help="Path to the input JSON file containing trial data"
    )
    parser.add_argument(
        "clinical_record_file", help="Path to the input clinical record file"
    )
    parser.add_argument(
        "--openai-api-key",
        help="OpenAI API key",
        default=os.getenv("OPENAI_API_KEY"),
    )
    parser.add_argument(
        "--perplexity-api-key",
        help="Perplexity API key",
        default=os.getenv("PERPLEXITY_API_KEY"),
    )
    parser.add_argument(
        "--output",
        help="Output JSON file path (default: analyzed_[input_filename])",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=None,
        help="Maximum number of trials to process",
    )
    args = parser.parse_args()

    try:
        # Initialize clients
        gpt_client = GPTClient(api_key=args.openai_api_key)
        perplexity_client = PerplexityClient(api_key=args.perplexity_api_key)

        clinical_record = read_input_file(args.clinical_record_file)
        
        # Process the trials
        process_trials_file(
            args.trials_json_file,
            gpt_client,
            perplexity_client,
            clinical_record,
            args.output,
        )
        
        # Final message just to be sure
        logger.info("Main process completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}", exc_info=True)
        raise
        
    finally:
        # Final flush of all logs
        for handler in logging.getLogger().handlers:
            handler.flush()
        sys.stdout.flush()
        sys.stderr.flush()
        
        logger.info("Script execution finished.")
