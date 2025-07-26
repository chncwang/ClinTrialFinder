#!/usr/bin/env python3
import argparse
import json
import logging
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Add parent directory to Python path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from base.clinical_trial import ClinicalTrialsParser
from base.gpt_client import GPTClient
from base.perplexity import PerplexityClient
from base.trial_expert import analyze_drugs_and_get_recommendation
from base.utils import read_input_file

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set specific loggers to INFO level to filter out debug messages
logging.getLogger("openai").setLevel(logging.INFO)
logging.getLogger("httpcore").setLevel(logging.INFO)
logging.getLogger("base").setLevel(logging.INFO)

# Create a single log file with timestamp for all loggers
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"assign_recommendation_by_id_{timestamp}.log"
file_handler = logging.FileHandler(log_filename)
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)

# Add handler to the root logger to capture all logs
root_logger = logging.getLogger()
root_logger.addHandler(file_handler)

# Configure base.trial_expert logger
trial_expert_logger = logging.getLogger("base.trial_expert")
trial_expert_logger.setLevel(logging.INFO)
trial_expert_logger.propagate = True  # Allow propagation to root logger

# Log the filename being used
logger.info(f"All logs will be written to: {os.path.abspath(log_filename)}")


def fetch_trial_data(nct_id: str) -> List[Dict[str, Any]]:
    """Fetch clinical trial data directly from ClinicalTrials.gov."""
    # Create a temporary file to store the spider output
    with tempfile.NamedTemporaryFile(
        mode="w+", suffix=".json", delete=False
    ) as tmp_file:
        temp_output = tmp_file.name
        logger.info(f"Using temporary file: {temp_output}")

    try:
        # Import scrapy components here to avoid circular imports
        from scrapy.crawler import CrawlerProcess
        from scrapy.utils.project import get_project_settings
        from clinical_trial_crawler.clinical_trial_crawler.spiders.clinical_trials_spider import (
            ClinicalTrialsSpider,
        )

        # Configure and run the spider with minimal settings
        settings = get_project_settings()
        settings.set("DOWNLOAD_DELAY", 1)
        settings.set(
            "USER_AGENT",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
        )

        process = CrawlerProcess(settings)
        process.crawl(
            ClinicalTrialsSpider, specific_trial=nct_id, output_file=temp_output
        )
        process.start()  # This will block until the crawl is complete

        # Read the results
        content = ""
        try:
            with open(temp_output, "r") as f:
                content = f.read().strip()
                if not content:
                    logger.error("No data was written to the output file")
                    return []
                logger.debug(f"fetch_trial_data: Read content: {content[:200]}...")
                return json.loads(content)
        except FileNotFoundError:
            logger.error(f"fetch_trial_data: Output file not found: {temp_output}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"fetch_trial_data: Failed to parse spider output: {e}")
            logger.error(f"fetch_trial_data: Content of output file: {content[:200]}...")
            return []

    finally:
        # Clean up the temporary file
        try:
            Path(temp_output).unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"Failed to delete temporary file {temp_output}: {e}")


def assign_recommendation_to_trial(
    clinical_record: str,
    nct_id: str,
    gpt_client: GPTClient,
    perplexity_client: PerplexityClient,
    trials_file: str | None = None,
) -> Dict[str, Any]:
    """
    Assign recommendation level to a specific clinical trial.
    
    Args:
        clinical_record: Patient's clinical record
        nct_id: NCT ID of the trial to analyze
        gpt_client: Initialized GPT client
        perplexity_client: Initialized Perplexity client
        trials_file: Optional path to JSON file containing trial data
        
    Returns:
        Dictionary containing the trial data with recommendation level assigned
    """
    logger.info(f"Starting recommendation assignment for trial {nct_id}")
    
    # Load trial data
    if trials_file and os.path.exists(trials_file):
        logger.info(f"Loading trial from file: {trials_file}")
        with open(trials_file, "r") as f:
            trials_data = json.load(f)
        trials_parser = ClinicalTrialsParser(trials_data)
        trial = trials_parser.get_trial_by_nct_id(nct_id)
        if not trial:
            raise ValueError(f"Trial {nct_id} not found in {trials_file}")
    else:
        logger.info(f"Fetching trial {nct_id} from ClinicalTrials.gov")
        trials_data = fetch_trial_data(nct_id)
        if not trials_data:
            raise ValueError(f"Failed to fetch trial {nct_id} from ClinicalTrials.gov")
        
        trials_parser = ClinicalTrialsParser(trials_data)
        trial = trials_parser.get_trial_by_nct_id(nct_id)
        if not trial:
            raise ValueError(f"Trial {nct_id} not found in fetched data")
    
    logger.info(f"Analyzing trial: {trial.identification.brief_title}")
    
    # Analyze the trial and get recommendation
    recommendation, reason, drug_analyses, cost = analyze_drugs_and_get_recommendation(
        clinical_record=clinical_record,
        trial=trial,
        perplexity_client=perplexity_client,
        gpt_client=gpt_client,
    )
    
    logger.info(f"Recommendation: {recommendation}")
    logger.info(f"Reason: {reason}")
    logger.info(f"Drug Analyses: {drug_analyses}")
    logger.info(f"Total cost: ${cost:.6f}")
    
    # Convert trial to dictionary and add recommendation data
    trial_dict = trial.to_dict()
    trial_dict["recommendation_level"] = str(recommendation)
    trial_dict["reason"] = reason
    trial_dict["drug_analysis"] = drug_analyses
    trial_dict["analysis_cost"] = cost
    
    return trial_dict


def main():
    parser = argparse.ArgumentParser(
        description="Assign recommendation level to a clinical trial by ID"
    )
    parser.add_argument(
        "nct_id", help="NCT ID of the trial to analyze"
    )
    parser.add_argument(
        "clinical_record_file", help="Path to the clinical record file"
    )
    parser.add_argument(
        "--trials-file",
        help="Path to JSON file containing trial data (optional, will fetch from ClinicalTrials.gov if not provided)"
    )
    parser.add_argument(
        "--output",
        help="Output JSON file path (default: recommendation_{nct_id}_{timestamp}.json)"
    )
    parser.add_argument(
        "--openai-api-key",
        help="OpenAI API key (alternatively, set OPENAI_API_KEY environment variable)",
    )
    parser.add_argument(
        "--perplexity-api-key",
        help="Perplexity API key (alternatively, set PERPLEXITY_API_KEY environment variable)",
    )
    parser.add_argument(
        "--cache-size",
        type=int,
        default=10000,
        help="Maximum number of cached responses to keep",
    )
    args = parser.parse_args()

    # Get API keys from arguments or environment
    api_key = args.openai_api_key or os.getenv("OPENAI_API_KEY")
    perplexity_api_key = args.perplexity_api_key or os.getenv("PERPLEXITY_API_KEY")

    if not api_key:
        logger.error(
            "OpenAI API key must be provided via --openai-api-key or OPENAI_API_KEY environment variable"
        )
        sys.exit(1)

    if not perplexity_api_key:
        logger.error(
            "Perplexity API key must be provided via --perplexity-api-key or PERPLEXITY_API_KEY environment variable"
        )
        sys.exit(1)

    try:
        # Read clinical record
        clinical_record = read_input_file(args.clinical_record_file)
        logger.info(f"Read clinical record from {args.clinical_record_file}")

        # Initialize clients
        gpt_client = GPTClient(api_key=api_key, cache_size=args.cache_size)
        perplexity_client = PerplexityClient(perplexity_api_key)

        # Assign recommendation to trial
        result = assign_recommendation_to_trial(
            clinical_record=clinical_record,
            nct_id=args.nct_id,
            gpt_client=gpt_client,
            perplexity_client=perplexity_client,
            trials_file=args.trials_file,
        )

        # Save result
        if args.output:
            output_file = args.output
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"recommendation_{args.nct_id}_{timestamp}.json"

        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)

        logger.info(f"Recommendation saved to: {output_file}")
        logger.info("=" * 50)
        logger.info("Recommendation assignment completed")
        logger.info(f"Trial: {args.nct_id}")
        logger.info(f"Recommendation: {result['recommendation_level']}")
        logger.info(f"Reason: {result['reason']}")
        logger.info(f"Cost: ${result['analysis_cost']:.2f}")
        logger.info(f"Output file: {output_file}")
        logger.info(f"Log file: {os.path.abspath(log_filename)}")
        logger.info("=" * 50)

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


if __name__ == "__main__":
    main() 