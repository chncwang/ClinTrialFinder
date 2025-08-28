#!/usr/bin/env python3
"""
Script to process categorized error cases and handle data label errors.
For each data label error case:
- If case_type is false_negative (label is true but should be false), record as true negative
- If case_type is false_positive (label is false but should be true), record as true positive

The script can also accept an existing TSV file to merge with new processed records.
"""

import json
import csv
import logging
import argparse
from typing import List, Optional, Tuple
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


def load_existing_tsv(file_path: str) -> List[CorrectedRecord]:
    """Load existing TSV file with corrected records."""
    records: List[CorrectedRecord] = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')

            for row in reader:
                try:
                    # Handle both 'query-id' and 'query_id' column names
                    query_id = row.get('query-id') or row.get('query_id', '')
                    corpus_id = row.get('corpus-id') or row.get('corpus_id', '')
                    score_str = row.get('score', '')

                    if query_id and corpus_id:
                        score = None if score_str == '' else int(score_str)
                        record = CorrectedRecord(
                            query_id=query_id,
                            corpus_id=corpus_id,
                            score=score
                        )
                        records.append(record)
                except Exception as e:
                    logger.warning(f"Failed to parse row: {e}, skipping row: {row}")
                    continue

        logger.info(f"Successfully loaded {len(records)} existing records from {file_path}")
        return records
    except Exception as e:
        logger.error(f"Error loading existing TSV file: {e}")
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
            # Since this is a data label error, the correct label should be 1
            corrected_record.score = 1
            logger.info(f"False negative data label error: {patient_id} -> {trial_id} (corrected to 1)")

        elif case_type == 'false_positive':
            # Label was false but should be true (true positive)
            # Since this is a data label error, the correct label should be 2
            corrected_record.score = 2
            logger.info(f"False positive data label error: {patient_id} -> {trial_id} (corrected to 2)")

        else:
            logger.warning(f"Unknown case type '{case_type}' for {patient_id} -> {trial_id}")
            continue

        corrected_records.append(corrected_record)

    return corrected_records


def merge_records(new_records: List[CorrectedRecord], existing_records: List[CorrectedRecord]) -> List[CorrectedRecord]:
    """
    Merge new records with existing records.

    Logic:
    - X = existing records (input tuple set)
    - Y = new processed records
    - Z = final output where:
        - When X is absent: Z = Y
        - When X is present: Z = {z | z belongs to Y OR (z's query_id and corpus_id combination is not present in Y but present in X)}

    This means existing records cannot be modified by new processing, but won't be lost if they don't appear in new processing.
    """
    if not existing_records:
        logger.info("No existing records to merge with, returning new records only")
        return new_records

    # Create a dictionary of existing records for easy lookup and modification
    existing_records_dict: dict[Tuple[str, str], CorrectedRecord] = {}
    for record in existing_records:
        key = (record.query_id, record.corpus_id)
        existing_records_dict[key] = record

    # Add new records (these will overwrite existing ones with same query_id + corpus_id)
    for record in new_records:
        key = (record.query_id, record.corpus_id)
        existing_records_dict[key] = record

    # Convert back to list
    merged_records: List[CorrectedRecord] = list(existing_records_dict.values())

    logger.info(f"Merged {len(new_records)} new records with {len(existing_records)} existing records")
    logger.info(f"Final output contains {len(merged_records)} records")

    return merged_records


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

        logger.info(f"Successfully saved {len(records)} records to {output_file}")

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
    parser.add_argument(
        "-e", "--existing",
        help="Existing TSV file to merge with new processed records (optional)"
    )

    args = parser.parse_args()

    # Input file
    input_file = args.input_file

    # Output file
    output_file = args.output

    # Existing TSV file (optional)
    existing_tsv = args.existing

    # Check if input file exists
    if not Path(input_file).exists():
        logger.error(f"Input file '{input_file}' not found.")
        return

    # Load existing records if provided
    existing_records: List[CorrectedRecord] = []
    if existing_tsv:
        if not Path(existing_tsv).exists():
            logger.error(f"Existing TSV file '{existing_tsv}' not found.")
            return
        existing_records = load_existing_tsv(existing_tsv)

    # Load JSON data
    cases = load_json_file(input_file)
    if not cases:
        return

    # Find data label error cases
    data_label_error_cases = [case for case in cases if is_data_label_error(case)]
    logger.info(f"Found {len(data_label_error_cases)} data label error cases")

    if not data_label_error_cases:
        logger.info("No data label error cases found.")
        # If no new cases but we have existing records, still save them
        if existing_records:
            logger.info("Saving existing records only (no new cases to process)")
            save_to_tsv(existing_records, output_file)
        return

    # Process data label errors (this is Y in the user's notation)
    new_records = process_data_label_errors(data_label_error_cases)

    if new_records:
        # Merge new records with existing records (this creates Z in the user's notation)
        merged_records = merge_records(new_records, existing_records)

        # Save merged records to TSV
        save_to_tsv(merged_records, output_file)

        # Print summary
        false_negatives = sum(1 for r in new_records if r.score == 1)
        false_positives = sum(1 for r in new_records if r.score == 2)

        logger.info(f"Summary:")
        logger.info(f"New data label error cases processed: {len(new_records)}")
        logger.info(f"False negatives corrected to true negatives (score=1): {false_negatives}")
        logger.info(f"False positives corrected to true positives (score=2): {false_positives}")
        logger.info(f"Existing records preserved: {len(existing_records)}")
        logger.info(f"Total merged records: {len(merged_records)}")
        logger.info(f"Output saved to: {output_file}")
    else:
        logger.info("No corrected records generated.")
        # If no new records but we have existing records, still save them
        if existing_records:
            logger.info("Saving existing records only (no new cases to process)")
            save_to_tsv(existing_records, output_file)


if __name__ == "__main__":
    main()
