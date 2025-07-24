"""
Script to download clinical trials using the scrapy crawler.

Usage:
    python -m scripts.download_trials --condition "disease name" [options]

Options:
    --condition TEXT            Disease or condition to search for (required if not using --specific-trial)
    --exclude-completed         Exclude completed trials
    --output-file TEXT          Output file path (default: {condition}_trials.json)
    --specific-trial TEXT       Download a specific trial by NCT ID (required if not using --condition)
    --log-level TEXT            Set the log level (default: INFO)
    --include-broader           Also download trials for broader disease categories
    --openai-api-key TEXT       OpenAI API key for broader category identification
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Any, Dict

# Import the disease expert module for broader categories
from base.disease_expert import get_parent_disease_categories
from base.gpt_client import GPTClient


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Download clinical trials data")

    parser.add_argument(
        "--condition",
        type=str,
        help="Disease or condition to search for",
    )

    parser.add_argument(
        "--exclude-completed",
        action="store_true",
        help="Exclude completed trials",
        default=False,
    )

    parser.add_argument(
        "--output-file",
        type=str,
        help="Output file path (default: {condition}_trials.json)",
    )

    parser.add_argument(
        "--specific-trial",
        type=str,
        help="Download a specific trial by NCT ID",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the log level (default: INFO)",
    )

    parser.add_argument(
        "--include-broader",
        action="store_true",
        help="Also download trials for broader disease categories",
        default=False,
    )

    parser.add_argument(
        "--openai-api-key",
        type=str,
        help="OpenAI API key for broader category identification",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.condition and not args.specific_trial:
        parser.error("Either --condition or --specific-trial must be provided")

    # Check if broader categories are requested but no API key is provided
    if (
        args.include_broader
        and not args.openai_api_key
        and not os.getenv("OPENAI_API_KEY")
    ):
        parser.error(
            "The --include-broader option requires an OpenAI API key via --openai-api-key or OPENAI_API_KEY environment variable"
        )

    # Check if broader categories are requested with specific trial
    if args.include_broader and args.specific_trial:
        parser.error(
            "The --include-broader option cannot be used with --specific-trial"
        )

    # Set default output file if not provided
    if not args.output_file:
        if args.specific_trial:
            args.output_file = f"{args.specific_trial}.json"
        else:
            # Replace spaces with underscores for filename
            condition_filename = args.condition.replace(" ", "_").lower()
            if args.exclude_completed:
                args.output_file = f"{condition_filename}_uncompleted_trials.json"
            else:
                args.output_file = f"{condition_filename}_trials.json"

    return args


def get_output_filename(condition: str, args: argparse.Namespace) -> str:
    """Generate output filename based on condition and args."""
    base_name = condition.replace(" ", "_").lower()
    suffix = "_uncompleted" if args.exclude_completed else ""
    return f"{base_name}{suffix}_trials.json"


def download_trials(args: argparse.Namespace, condition: Optional[str] = None, output_file: Optional[str] = None) -> Tuple[bool, Optional[str]]:
    """Download clinical trials using scrapy."""
    # Get the project root directory and change to crawler directory
    project_root = Path(__file__).resolve().parent.parent
    crawler_dir = project_root / "clinical_trial_crawler"
    os.chdir(crawler_dir)

    # Use provided condition or from args
    condition = condition or args.condition

    # Use provided output file or generate from condition
    if output_file is None:
        if condition:
            output_file = os.path.join(project_root, get_output_filename(condition, args))
        else:
            # For specific trials, use the trial ID as filename
            output_file = os.path.join(project_root, f"{args.specific_trial}.json")

    # Build the scrapy command
    cmd = "python -m scrapy crawl clinical_trials"

    # Add condition if provided
    if condition:
        cmd += f' -a condition="{condition}"'

    # Add specific trial if provided
    if args.specific_trial:
        cmd += f" -a specific_trial={args.specific_trial}"

    # Add exclude completed flag if set
    if args.exclude_completed:
        cmd += " -a exclude_completed=true"

    # Add log level
    cmd += f" --loglevel={args.log_level}"

    # Add output file
    cmd += f' -O "{output_file}"'

    # Print the command being executing
    print(f"Executing command: {cmd}")

    try:
        # Execute the command
        process = subprocess.Popen(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate()

        # Get the return code
        return_code = process.returncode

        # Print any output
        if stdout:
            print(stdout.decode())
        if stderr:
            print(stderr.decode())

        if return_code == 0:
            print(f"Command completed successfully.")
            print(f"Output saved to: {output_file}")
            return True, output_file
        else:
            print(f"Command failed with exit code: {return_code}")
            return False, None

    except Exception as e:
        print(f"Error executing command: {e}")
        return False, None


def get_broader_categories(condition: str, api_key: str) -> List[str]:
    """Get broader disease categories for the given condition."""
    print(f"Identifying broader disease categories for: {condition}")

    # Initialize GPT client
    gpt_client = GPTClient(api_key)

    # Get broader categories
    try:
        categories, cost = get_parent_disease_categories(condition, gpt_client)
        print(f"Found {len(categories)} broader categories: {categories}")
        print(f"API cost: ${cost:.6f}")
        return categories
    except Exception as e:
        print(f"Error getting broader categories: {e}")
        return []


def load_trials_from_file(file_path: str) -> List[Dict[str, Any]]:
    """Load trials from a JSON file."""
    try:
        with open(file_path, "r") as f:
            trials: Any = json.load(f)
            if isinstance(trials, list):
                return trials  # type: ignore
            else:
                print(f"Warning: {file_path} does not contain a list of trials")
                return []
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []


def merge_trials(trials_list: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Merge multiple lists of trials, removing duplicates based on NCT ID."""
    unique_trials: Dict[str, Dict[str, Any]] = {}
    for trials in trials_list:
        for trial in trials:
            nct_id = trial.get("identification", {}).get("nct_id")
            if nct_id:
                unique_trials[nct_id] = trial
    return list(unique_trials.values())


def main():
    """Main entry point."""
    args = parse_arguments()

    # Set OpenAI API key if provided
    if args.openai_api_key:
        os.environ["OPENAI_API_KEY"] = args.openai_api_key

    # If not including broader categories, just download trials for the specified condition
    if not args.include_broader or args.specific_trial:
        success, _ = download_trials(args)
        sys.exit(0 if success else 1)

    # Get broader categories
    broader_categories = get_broader_categories(args.condition, args.openai_api_key)
    if not broader_categories:
        print("No broader categories found")
        sys.exit(1)

    # List to store all output files
    output_files: List[str] = []

    # Download trials for the original condition
    success, output_file = download_trials(args)
    if success and output_file:
        output_files.append(output_file)
    else:
        print("Failed to download trials for the original condition")
        sys.exit(1)

    # Download trials for each broader category
    for category in broader_categories:
        print(f"\nDownloading trials for broader category: {category}")
        category_output = get_output_filename(category, args)
        success, output_file = download_trials(
            args, condition=category, output_file=category_output
        )
        if success and output_file:
            output_files.append(output_file)
        else:
            print(f"Warning: Failed to download trials for category: {category}")

    # Merge all trials
    all_trials: List[List[Dict[str, Any]]] = []
    for file in output_files:
        trials = load_trials_from_file(file)
        print(f"Loaded {len(trials)} trials from {file}")
        all_trials.append(trials)

    # Create merged output filename
    base_condition = args.condition.replace(" ", "_").lower()
    suffix = "_uncompleted" if args.exclude_completed else ""
    merged_output = f"{base_condition}_with_broader{suffix}_trials.json"

    # Merge and save
    unique_trials = merge_trials(all_trials)
    print(f"\nTotal unique trials after merging: {len(unique_trials)}")

    try:
        with open(merged_output, "w") as f:
            json.dump(unique_trials, f, indent=2)
        print(f"Successfully saved merged trials to: {merged_output}")
    except Exception as e:
        print(f"Error writing merged trials to {merged_output}: {e}")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
