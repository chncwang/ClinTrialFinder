"""
Script to download clinical trials using the scrapy crawler.

Usage:
    python -m scripts.download_trials --condition "disease name" [options]

Options:
    --condition TEXT            Disease or condition to search for (required)
    --exclude-completed         Exclude completed trials
    --output-file TEXT          Output file path (default: {condition}_trials.json)
    --specific-trial TEXT       Download a specific trial by NCT ID
    --log-level TEXT           Set the log level (default: INFO)
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


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

    args = parser.parse_args()

    # Validate arguments
    if not args.condition and not args.specific_trial:
        parser.error("Either --condition or --specific-trial must be provided")

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


def download_trials(args):
    """Download clinical trials using scrapy."""
    # Get the project root directory
    project_root = Path(__file__).parent.parent

    # Get the crawler directory
    crawler_dir = project_root / "clinical_trial_crawler"

    # Make sure the output path is absolute
    if not os.path.isabs(args.output_file):
        output_file = project_root / args.output_file
    else:
        output_file = args.output_file

    # Change to the crawler directory
    os.chdir(crawler_dir)

    # Build the command string with proper quoting
    cmd = "scrapy crawl clinical_trials"

    # Add arguments
    if args.condition:
        # Properly escape the condition to handle spaces
        cmd += f' -a condition="{args.condition}"'

    if args.exclude_completed:
        cmd += " -a exclude_completed=true"

    if args.specific_trial:
        cmd += f" -a specific_trial={args.specific_trial}"

    # Set log level
    cmd += f" --loglevel={args.log_level}"

    # Add output file
    cmd += f' -O "{output_file}"'

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
            print(f"Output saved to: {output_file}")
            return True
        else:
            print(f"Command failed with exit code: {return_code}")
            return False

    except Exception as e:
        print(f"Error executing command: {e}")
        return False


def main():
    """Main entry point."""
    args = parse_arguments()
    success = download_trials(args)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
