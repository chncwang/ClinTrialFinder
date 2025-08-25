#!/usr/bin/env python3
"""
Test Script for GPT Categorization of False Positive Error Cases

This script demonstrates how to use the enhanced error case analyzer
to categorize false positive errors using GPT-5 (or GPT-4o as fallback).

Usage:
    python -m scripts.test_gpt_categorization <json_file_path>
"""

import os
import sys
import logging
from pathlib import Path

# Add the base directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.analyze_error_cases import ErrorCaseAnalyzer, setup_logging

def main():
    """Test the GPT categorization functionality."""
    if len(sys.argv) != 2:
        print("Usage: python -m scripts.test_gpt_categorization <json_file_path>")
        sys.exit(1)

    json_file_path = sys.argv[1]

    # Setup logging
    setup_logging("INFO")
    logger = logging.getLogger(__name__)

    # Check if API key is available
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable not set")
        logger.error("Please set your OpenAI API key to test GPT categorization")
        sys.exit(1)

    # Check if file exists
    if not Path(json_file_path).exists():
        logger.error(f"File {json_file_path} does not exist")
        sys.exit(1)

    logger.info("="*80)
    logger.info("TESTING GPT CATEGORIZATION OF FALSE POSITIVE ERROR CASES")
    logger.info("="*80)

    try:
        # Initialize analyzer with GPT client
        analyzer = ErrorCaseAnalyzer(json_file_path, api_key)

        # Load data
        logger.info("Loading error case data...")
        analyzer.load_data()

        # Log basic summary
        analyzer.log_summary()

        # Test GPT categorization
        logger.info("\n" + "="*80)
        logger.info("STARTING GPT CATEGORIZATION")
        logger.info("="*80)

        categorized_cases = analyzer.categorize_false_positives_with_gpt()

        if categorized_cases:
            # Export results
            output_path = f"test_categorized_results_{Path(json_file_path).stem}.csv"
            if analyzer.export_categorized_results(categorized_cases, output_path):
                logger.info(f"Results exported to: {output_path}")

                # Show summary
                logger.info("\n" + "-"*80)
                logger.info("CATEGORIZATION SUMMARY")
                logger.info("-"*80)
                for category, cases in categorized_cases.items():
                    logger.info(f"{category.replace('_', ' ').title()}: {len(cases)} cases")
            else:
                logger.error("Failed to export results")
        else:
            logger.warning("No cases were categorized")

    except Exception as e:
        logger.error(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    logger.info("\n" + "="*80)
    logger.info("TEST COMPLETED")
    logger.info("="*80)

if __name__ == "__main__":
    main()
