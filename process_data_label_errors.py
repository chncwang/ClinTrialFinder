#!/usr/bin/env python3
"""
Script to process categorized error cases and handle data label errors.
For each data label error case:
- If case_type is false_negative (label is true but should be false), record as true negative
- If case_type is false_positive (label is false but should be true), record as true positive
"""

import json
import csv
import logging
import argparse
from typing import List, Optional
from dataclasses import dataclass
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class ErrorCase:
    """Represents an error case from the input JSON file."""
    gpt_categorization: str
    case_type: str
    patient_id: str
    trial_id: str


@dataclass
class CorrectedRecord:
    """Represents a corrected record for output."""
    query_id: str
    corpus_id: str
    score: Optional[int]


def load_json_file(file_path: str) -> List[ErrorCase]:
    """Load and parse the JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        # Convert raw dict data to ErrorCase objects
        cases: List[ErrorCase] = []
        for item in raw_data:
            try:
                case = ErrorCase(
                    gpt_categorization=item.get('gpt_categorization', ''),
                    case_type=item.get('case_type', ''),
                    patient_id=item.get('patient_id', ''),
                    trial_id=item.get('trial_id', '')
                )
                cases.append(case)
            except Exception as e:
                logger.warning(f"Failed to parse case: {e}, skipping item: {item}")
                continue

        logger.info(f"Successfully loaded {len(cases)} cases from {file_path}")
        return cases
    except Exception as e:
        logger.error(f"Error loading JSON file: {e}")
        return []


def is_data_label_error(case: ErrorCase) -> bool:
    """Check if the case is a data label error."""
    return case.gpt_categorization == 'data_label_error'


def process_data_label_errors(cases: List[ErrorCase]) -> List[CorrectedRecord]:
    """Process data label error cases and create corrected records."""
    corrected_records: List[CorrectedRecord] = []

    for case in cases:
        if not is_data_label_error(case):
            continue

        case_type = case.case_type
        patient_id = case.patient_id
        trial_id = case.trial_id

        if not all([case_type, patient_id, trial_id]):
            logger.warning(f"Missing required fields in case: {case}")
            continue

        # Create corrected record
        corrected_record = CorrectedRecord(
            query_id=patient_id,
            corpus_id=trial_id,
            score=None  # Will be set based on case type
        )

        if case_type == 'false_negative':
            # Label was true but should be false (true negative)
            # Since this is a data label error, the correct label should be 0
            corrected_record.score = 0
            logger.info(f"False negative data label error: {patient_id} -> {trial_id} (corrected to 0)")

        elif case_type == 'false_positive':
            # Label was false but should be true (true positive)
            # Since this is a data label error, the correct label should be 1
            corrected_record.score = 1
            logger.info(f"False positive data label error: {patient_id} -> {trial_id} (corrected to 1)")

        else:
            logger.warning(f"Unknown case type '{case_type}' for {patient_id} -> {trial_id}")
            continue

        corrected_records.append(corrected_record)

    return corrected_records


def save_to_tsv(records: List[CorrectedRecord], output_file: str) -> None:
    """Save corrected records to TSV file."""
    if not records:
        logger.info("No records to save.")
        return

    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['query-id', 'corpus-id', 'score'], delimiter='\t')
            writer.writeheader()

            # Convert CorrectedRecord objects to dicts for CSV writer
            for record in records:
                writer.writerow({
                    'query-id': record.query_id,
                    'corpus-id': record.corpus_id,
                    'score': record.score
                })

        logger.info(f"Successfully saved {len(records)} corrected records to {output_file}")

    except Exception as e:
        logger.error(f"Error saving TSV file: {e}")


def main():
    """Main function to process the JSON file."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Process categorized error cases and handle data label errors"
    )
    parser.add_argument(
        "input_file",
        help="Input JSON file path (required)"
    )
    parser.add_argument(
        "-o", "--output",
        default="corrected_data_label_errors.tsv",
        help="Output TSV file path (default: corrected_data_label_errors.tsv)"
    )

    args = parser.parse_args()

    # Input file
    input_file = args.input_file

    # Output file
    output_file = args.output

    # Check if input file exists
    if not Path(input_file).exists():
        logger.error(f"Input file '{input_file}' not found.")
        return

    # Load JSON data
    cases = load_json_file(input_file)
    if not cases:
        return

    # Find data label error cases
    data_label_error_cases = [case for case in cases if is_data_label_error(case)]
    logger.info(f"Found {len(data_label_error_cases)} data label error cases")

    if not data_label_error_cases:
        logger.info("No data label error cases found.")
        return

    # Process data label errors
    corrected_records = process_data_label_errors(data_label_error_cases)

    if corrected_records:
        # Save corrected records to TSV
        save_to_tsv(corrected_records, output_file)

        # Print summary
        false_negatives = sum(1 for r in corrected_records if r.score == 0)
        false_positives = sum(1 for r in corrected_records if r.score == 1)

        logger.info(f"Summary:")
        logger.info(f"Total data label error cases processed: {len(corrected_records)}")
        logger.info(f"False negatives corrected to true negatives (score=0): {false_negatives}")
        logger.info(f"False positives corrected to true positives (score=1): {false_positives}")
        logger.info(f"Output saved to: {output_file}")
    else:
        logger.info("No corrected records generated.")


if __name__ == "__main__":
    main()
