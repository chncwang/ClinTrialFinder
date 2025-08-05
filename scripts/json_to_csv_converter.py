#!/usr/bin/env python3
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

from base.clinical_trial import ClinicalTrial


def flatten_locations(locations: List[Dict[str, Any]]) -> Dict[str, str]:
    """Flatten location information into a single row."""
    if not locations:
        return {
            'facility': '',
            'city': '',
            'state': '',
            'country': '',
            'location_status': ''
        }
    
    # Use the first location as primary
    primary_location = locations[0]
    return {
        'facility': primary_location.get('facility', ''),
        'city': primary_location.get('city', ''),
        'state': primary_location.get('state', ''),
        'country': primary_location.get('country', ''),
        'location_status': primary_location.get('status', '')
    }


def flatten_phases(phases: List[int]) -> str:
    """Convert phases list to a readable string."""
    if not phases:
        return ''
    return ', '.join([f'Phase {phase}' for phase in sorted(phases)])


def flatten_conditions(conditions: List[str]) -> str:
    """Convert conditions list to a semicolon-separated string."""
    if not conditions:
        return ''
    return '; '.join(conditions)


def flatten_collaborators(collaborators: List[str]) -> str:
    """Convert collaborators list to a semicolon-separated string."""
    if not collaborators:
        return ''
    return '; '.join(collaborators)


def trial_to_csv_row(trial_data: Dict[str, Any], rank: int) -> Dict[str, Any]:
    """Convert a trial dictionary to a CSV row format."""
    # Create ClinicalTrial object to access structured data
    trial = ClinicalTrial(trial_data)
    
    # Flatten location information
    location_info = flatten_locations(trial_data['contacts_locations']['locations'])
    
    # Prepare the CSV row
    row = {
        'rank': rank,
        'nct_id': trial.identification.nct_id,
        'brief_title': trial.identification.brief_title,
        'acronym': trial.identification.acronym or '',
        'url': trial.identification.url,
        'overall_status': trial.status.overall_status,
        'start_date': trial.status.start_date.strftime('%Y-%m-%d') if trial.status.start_date else '',
        'completion_date': trial.status.completion_date.strftime('%Y-%m-%d') if trial.status.completion_date else '',
        'primary_completion_date': trial.status.primary_completion_date.strftime('%Y-%m-%d') if trial.status.primary_completion_date else '',
        'study_type': trial.design.study_type,
        'phases': flatten_phases(trial.design.phases),
        'enrollment': trial.design.enrollment,
        'gender': trial.eligibility.gender,
        'minimum_age': trial.eligibility.minimum_age,
        'maximum_age': trial.eligibility.maximum_age,
        'healthy_volunteers': trial.eligibility.healthy_volunteers,
        'lead_sponsor': trial.sponsor.lead_sponsor,
        'collaborators': flatten_collaborators(trial.sponsor.collaborators),
        'facility': location_info['facility'],
        'city': location_info['city'],
        'state': location_info['state'],
        'country': location_info['country'],
        'location_status': location_info['location_status'],
        'recommendation_level': trial_data.get('recommendation_level', ''),
        'analysis_reason': trial_data.get('reason', ''),
        'drug_analysis': trial_data.get('drug_analysis', ''),
        'is_recruiting': trial.is_recruiting,
        'study_duration_days': trial.study_duration_days or ''
    }
    
    return row


def convert_json_to_csv(input_file: str, output_file: Optional[str] = None) -> str:
    """
    Convert ranked trials JSON file to CSV format.
    
    Args:
        input_file: Path to the input JSON file
        output_file: Path to the output CSV file (optional)
    
    Returns:
        Path to the created CSV file
    """
    
    # Generate output filename if not provided
    if output_file is None:
        input_path = Path(input_file)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"{input_path.stem}_ranked_{timestamp}.csv"
    
    logger.info(f"Converting {input_file} to {output_file}")
    
    try:
        # Read JSON file
        with open(input_file, 'r') as f:
            trials_data = json.load(f)
        
        logger.info(f"Loaded {len(trials_data)} trials from JSON file")
        
        # Define CSV headers
        headers = [
            'rank', 'nct_id', 'brief_title', 'acronym', 'url',
            'overall_status', 'start_date', 'completion_date', 'primary_completion_date',
            'study_type', 'phases',
            'enrollment', 'gender', 'minimum_age', 'maximum_age',
            'healthy_volunteers', 'lead_sponsor', 'collaborators', 'facility', 'city',
            'state', 'country', 'location_status', 'recommendation_level',
            'analysis_reason', 'drug_analysis', 'is_recruiting', 'study_duration_days'
        ]
        
        # Convert trials to CSV rows
        rows: List[Dict[str, Any]] = []
        for rank, trial_data in enumerate(trials_data, 1):
            row = trial_to_csv_row(trial_data, rank)
            rows.append(row)
        
        # Write CSV file
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)
        
        logger.info(f"Successfully converted {len(rows)} trials to CSV format")
        logger.info(f"CSV file saved to: {output_file}")
        
        return output_file
        
    except FileNotFoundError:
        logger.error(f"Input file not found: {input_file}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON file: {e}")
        raise
    except Exception as e:
        logger.error(f"Error converting to CSV: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Convert ranked trials JSON file to CSV format."
    )
    parser.add_argument(
        "input_file",
        help="Path to the input JSON file containing ranked trials"
    )
    parser.add_argument(
        "--output-file", "-o",
        help="Path to the output CSV file (optional, will auto-generate if not provided)"
    )
    
    args = parser.parse_args()
    
    try:
        output_file = convert_json_to_csv(args.input_file, args.output_file)
        print(f"\nConversion completed successfully!")
        print(f"CSV file: {output_file}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 