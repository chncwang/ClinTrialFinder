#!/usr/bin/env python3
"""
Error Cases Analysis Script

This script reads and analyzes error case JSON files from clinical trial filtering experiments.
It provides comprehensive analysis including statistics, patterns, and detailed examination of error cases.

Usage:
    python -m scripts.analyze_error_cases <json_file_path> [options]
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from collections import Counter, defaultdict
import pandas as pd
from datetime import datetime

# Import our custom error case classes
from base.error_case import ErrorCase, ErrorCaseCollection

# Global logger
logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    global logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger.setLevel(getattr(logging, level.upper()))


class ErrorCaseAnalyzer:
    """Analyzer for clinical trial error case data."""

    def __init__(self, json_file_path: str):
        """Initialize the analyzer with a JSON file path."""
        self.json_file_path = Path(json_file_path)
        self.collection: Optional[ErrorCaseCollection] = None

    def load_data(self) -> None:
        """Load and validate the JSON data."""
        if not self.json_file_path.exists():
            raise FileNotFoundError(f"File {self.json_file_path} does not exist.")

        # Use the ErrorCaseCollection to load and validate data
        self.collection = ErrorCaseCollection.from_file(str(self.json_file_path))
        logger.info(f"Successfully loaded {len(self.collection)} error cases from {self.json_file_path}")

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for the error cases."""
        if not self.collection:
            return {}

        return self.collection.get_statistics()

    def export_to_csv(self, output_path: str) -> bool:
        """Export error cases to CSV format."""
        try:
            if not self.collection:
                logger.error("No data loaded. Use load_data() first.")
                return False

            # Convert ErrorCase objects to dictionaries for pandas
            data_list = [case.to_dict() for case in self.collection]
            df = pd.DataFrame(data_list)
            df.to_csv(output_path, index=False)
            logger.info(f"Exported {len(df)} error cases to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            return False

    def log_summary(self):
        """Log a comprehensive summary of the error cases."""
        if not self.collection:
            logger.error("No data loaded. Use load_data() first.")
            return

        stats = self.get_summary_stats()

        logger.info("="*80)
        logger.info("ERROR CASES ANALYSIS SUMMARY")
        logger.info("="*80)

        logger.info(f"File: {self.json_file_path}")
        logger.info(f"Timestamp: {stats.get('timestamp', 'Unknown')}")
        logger.info(f"Total Error Cases: {stats.get('total_error_cases', 0)}")
        logger.info(f"False Positives: {stats.get('false_positives', 0)}")
        logger.info(f"False Negatives: {stats.get('false_negatives', 0)}")
        logger.info(f"Unique Patients: {stats.get('unique_patients', 0)}")
        logger.info(f"Unique Trials: {stats.get('unique_trials', 0)}")
        logger.info(f"Unique Diseases: {stats.get('unique_diseases', 0)}")

        if 'avg_suitability_probability' in stats:
            logger.info(f"Average Suitability Probability: {stats['avg_suitability_probability']:.3f}")
            logger.info(f"Min Suitability Probability: {stats['min_suitability_probability']:.3f}")
            logger.info(f"Max Suitability Probability: {stats['max_suitability_probability']:.3f}")

        logger.info("-"*80)
        logger.info("ERROR TYPE DISTRIBUTION")
        logger.info("-"*80)
        error_types = Counter(case.error_type for case in self.collection)
        for error_type, count in error_types.most_common():
            logger.info(f"{error_type}: {count}")

        logger.info("-"*80)
        logger.info("TOP DISEASES WITH ERRORS")
        logger.info("-"*80)
        diseases = Counter(case.disease_name for case in self.collection)
        for disease, count in diseases.most_common(10):
            logger.info(f"{disease}: {count}")

        logger.info("-"*80)
        logger.info("TOP PATIENTS WITH ERRORS")
        logger.info("-"*80)
        patients = Counter(case.patient_id for case in self.collection)
        for patient, count in patients.most_common(10):
            logger.info(f"{patient}: {count}")

        logger.info("-"*80)
        logger.info("SUITABILITY PROBABILITY DISTRIBUTION")
        logger.info("-"*80)
        prob_ranges: Dict[str, int] = defaultdict(int)
        for case in self.collection:
            prob = case.suitability_probability
            if prob >= 0.9:
                prob_ranges['0.9-1.0'] += 1
            elif prob >= 0.8:
                prob_ranges['0.8-0.9'] += 1
            elif prob >= 0.7:
                prob_ranges['0.7-0.8'] += 1
            elif prob >= 0.6:
                prob_ranges['0.6-0.7'] += 1
            else:
                prob_ranges['<0.6'] += 1

        for prob_range, count in sorted(prob_ranges.items()):
            logger.info(f"{prob_range}: {count}")

        logger.info("-"*80)
        logger.info("COMMON ERROR REASONS")
        logger.info("-"*80)
        reasons = Counter(case.reason for case in self.collection)
        for reason, count in reasons.most_common(5):
            logger.info(f"{reason}: {count}")


def main():
    """Main function to run the error case analyzer."""
    parser = argparse.ArgumentParser(
        description="Analyze clinical trial error case JSON files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m scripts.analyze_error_cases results/error_cases_20250825_063056.json
    python -m scripts.analyze_error_cases results/error_cases_20250825_063056.json --export-csv
        """
    )

    parser.add_argument('json_file', help='Path to the error cases JSON file')
    parser.add_argument('--export-csv', action='store_true',
                       help='Export error cases to CSV format')

    parser.add_argument('--output', help='Output CSV file path (default: error_cases_analysis.csv)')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Set the logging level (default: INFO)')

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    # Initialize analyzer
    analyzer = ErrorCaseAnalyzer(args.json_file)

    # Load data
    try:
        analyzer.load_data()
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        sys.exit(1)

    # Log summary
    analyzer.log_summary()

    # Export to CSV if requested
    if args.export_csv:
        output_path = args.output or f"error_cases_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        if analyzer.export_to_csv(output_path):
            logger.info("CSV export completed successfully")
        else:
            logger.error("CSV export failed")


if __name__ == "__main__":
    main()
