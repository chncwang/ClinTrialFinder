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

# Import the disease expert module for broader categories
try:
    from base.disease_expert import get_parent_disease_categories
    from base.gpt_client import GPTClient

    DISEASE_EXPERT_AVAILABLE = True
except ImportError:
    DISEASE_EXPERT_AVAILABLE = False


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

    # Check if broader categories are requested but disease_expert is not available
    if args.include_broader and not DISEASE_EXPERT_AVAILABLE:
        parser.error(
            "The --include-broader option requires the base.disease_expert module, which is not available"
        )

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


def download_trials(args, condition=None, output_file=None, append=False):
    """Download clinical trials using scrapy."""
    # Get the project root directory
    project_root = Path(__file__).parent.parent

    # Get the crawler directory
    crawler_dir = project_root / "clinical_trial_crawler"

    # Use provided condition and output_file or fall back to args
    condition = condition or args.condition

    # Make sure the output path is absolute
    if output_file:
        if not os.path.isabs(output_file):
            output_file = project_root / output_file
    else:
        if not os.path.isabs(args.output_file):
            output_file = project_root / args.output_file
        else:
            output_file = args.output_file

    # Change to the crawler directory
    os.chdir(crawler_dir)

    # Build the command string with proper quoting
    cmd = "scrapy crawl clinical_trials"

    # Add arguments
    if condition:
        # Properly escape the condition to handle spaces
        cmd += f' -a condition="{condition}"'

    if args.exclude_completed:
        cmd += " -a exclude_completed=true"

    if args.specific_trial:
        cmd += f" -a specific_trial={args.specific_trial}"

    # Set log level
    cmd += f" --loglevel={args.log_level}"

    # Add output file with -O (overwrite) or -o (append) flag
    output_flag = "-o" if append else "-O"
    cmd += f' {output_flag} "{output_file}"'

    # Print the command being executed
    print(f"Executing: {cmd}")

    try:
        # Execute the command and stream output in real-time
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Redirect stderr to stdout to capture all logs
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True,
        )

        # Print output in real-time
        print("\nScrapy Log Output:")
        print("-" * 50)

        for line in process.stdout:
            print(line.rstrip())

        # Wait for the process to complete
        return_code = process.wait()

        print("-" * 50)

        if return_code == 0:
            print(f"Command completed successfully.")
            print(f"Output {'appended to' if append else 'saved to'}: {output_file}")
            return True
        else:
            print(f"Command failed with exit code: {return_code}")
            return False

    except Exception as e:
        print(f"Error executing command: {e}")
        return False


def get_broader_categories(condition, api_key):
    """Get broader disease categories for the given condition."""
    if not DISEASE_EXPERT_AVAILABLE:
        print("Error: base.disease_expert module is not available")
        return []

    print(f"Identifying broader disease categories for: {condition}")

    # Initialize GPT client
    gpt_client = GPTClient(api_key)

    # Get broader categories
    categories, cost = get_parent_disease_categories(condition, gpt_client)

    print(f"Found {len(categories)} broader categories: {categories}")
    print(f"API cost: ${cost:.6f}")

    return categories


def merge_json_files(file_paths, output_path):
    """Merge multiple JSON files containing clinical trials data."""
    all_trials = []

    for file_path in file_paths:
        try:
            with open(file_path, "r") as f:
                trials = json.load(f)
                if isinstance(trials, list):
                    all_trials.extend(trials)
                    print(f"Added {len(trials)} trials from {file_path}")
                else:
                    print(f"Warning: {file_path} does not contain a list of trials")
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    # Remove duplicates based on NCT ID
    unique_trials = {}
    for trial in all_trials:
        nct_id = trial.get("identification", {}).get("nct_id")
        if nct_id:
            unique_trials[nct_id] = trial

    unique_trial_list = list(unique_trials.values())

    # Write the merged trials to the output file
    try:
        with open(output_path, "w") as f:
            json.dump(unique_trial_list, f, indent=2)
        print(
            f"Successfully merged {len(unique_trial_list)} unique trials to {output_path}"
        )
        return True
    except Exception as e:
        print(f"Error writing merged trials to {output_path}: {e}")
        return False


def main():
    """Main entry point."""
    args = parse_arguments()

    # If not including broader categories, just download trials for the specified condition
    if not args.include_broader or args.specific_trial:
        success = download_trials(args)
        sys.exit(0 if success else 1)

    # Get API key from args or environment
    api_key = args.openai_api_key or os.getenv("OPENAI_API_KEY")

    # Get broader categories
    broader_categories = get_broader_categories(args.condition, api_key)

    # Download trials for the original condition
    success = download_trials(args)
    if not success:
        print("Failed to download trials for the original condition")
        sys.exit(1)

    # Download trials for each broader category and append to the same file
    for category in broader_categories:
        print(f"\nDownloading trials for broader category: {category}")
        success = download_trials(
            args, condition=category, output_file=args.output_file, append=True
        )
        if not success:
            print(f"Warning: Failed to download trials for category: {category}")

    sys.exit(0)


if __name__ == "__main__":
    main()
