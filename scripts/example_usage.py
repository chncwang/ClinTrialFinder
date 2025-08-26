#!/usr/bin/env python3
"""
Example Usage of CategorizedErrorCases Class

This script demonstrates how to use the new CategorizedErrorCases class
instead of the dictionary approach for managing categorized error cases.
"""

import sys
from pathlib import Path

# Add the parent directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent))

from scripts.analyze_error_cases import CategorizedErrorCases, ErrorCategory
from base.error_case import ErrorCase
from typing import Dict, List, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_error_case(trial_id: str, patient_id: str, disease: str) -> ErrorCase:
    """Create a sample error case for demonstration purposes."""
    # This is a simplified example - in practice, you'd load real data
    case = ErrorCase(
        trial_id=trial_id,
        patient_id=patient_id,
        disease_name=disease,
        suitability_probability=0.85,
        reason="Sample reason for demonstration",
        full_medical_record="Sample clinical record text",
        trial_title="Sample Clinical Trial",
        trial_criteria="Sample inclusion/exclusion criteria",
        error_type="false_positive",
        ground_truth_relevant=False,
        predicted_eligible=True,
        original_relevance_score=0.85
    )
    return case


def demonstrate_class_usage():
    """Demonstrate the usage of the CategorizedErrorCases class."""
    logger.info("=== Demonstrating CategorizedErrorCases Class Usage ===\n")

    # Create a new instance
    categorized_cases = CategorizedErrorCases()

    # Create some sample error cases
    sample_cases = [
        ("TRIAL_001", "PATIENT_001", "Diabetes"),
        ("TRIAL_002", "PATIENT_002", "Hypertension"),
        ("TRIAL_003", "PATIENT_003", "Asthma"),
        ("TRIAL_004", "PATIENT_004", "Diabetes"),
        ("TRIAL_005", "PATIENT_005", "Hypertension"),
    ]

    # Add cases to different categories
    categories = [
        ErrorCategory.EXCLUSION_CRITERIA_VIOLATION,
        ErrorCategory.INCLUSION_CRITERIA_VIOLATION,
        ErrorCategory.DATA_LABEL_ERROR,
        ErrorCategory.EXCLUSION_CRITERIA_VIOLATION,
        ErrorCategory.INCLUSION_CRITERIA_VIOLATION,
    ]

    for i, (trial_id, patient_id, disease) in enumerate(sample_cases):
        case = create_sample_error_case(trial_id, patient_id, disease)
        category = categories[i]

        categorized_cases.add_case(
            category=category,
            case=case,
            gpt_categorization=category.value,
            gpt_reasoning=f"Sample reasoning for {category.value}",
            model_used="gpt-5",
            cost=0.001
        )

        logger.info(f"Added case {trial_id} to category: {category.value}")

    logger.info(f"\nTotal cases: {categorized_cases.get_total_count()}")

    # Demonstrate category-specific methods
    logger.info("\n=== Category Statistics ===")
    stats = categorized_cases.get_category_statistics()
    for category_value, category_stats in stats.items():
        logger.info(f"{category_value}: {category_stats['count']} cases ({category_stats['percentage']:.1f}%)")

    # Demonstrate getting cases for specific categories
    logger.info("\n=== Cases by Category ===")
    for category in ErrorCategory:
        cases = categorized_cases.get_cases_for_category(category)
        logger.info(f"{category.value}: {len(cases)} cases")
        for case in cases:
            logger.info(f"  - {case.case.trial_id} ({case.case.disease_name})")

    # Demonstrate the items() method for backward compatibility
    logger.info("\n=== Using items() method (backward compatibility) ===")
    for category_name, cases in categorized_cases.items():
        logger.info(f"{category_name}: {len(cases)} cases")

    # Demonstrate the get() method for backward compatibility
    logger.info("\n=== Using get() method (backward compatibility) ===")
    for category in ErrorCategory:
        cases = categorized_cases.get(category.value, [])
        if cases:
            logger.info(f"{category.value}: {len(cases)} cases")
        else:
            logger.info(f"{category.value}: 0 cases")

    # Demonstrate printing methods
    logger.info("\n=== Printing Statistics ===")
    categorized_cases.print_category_statistics()

    logger.info("\n=== Printing Examples ===")
    categorized_cases.print_category_examples(max_examples=2)

    # Demonstrate export functionality
    logger.info("\n=== Export Functionality ===")
    output_path = "sample_categorized_cases.csv"
    if categorized_cases.export_to_csv(output_path):
        logger.info(f"Successfully exported to {output_path}")
    else:
        logger.error("Failed to export")

    # Demonstrate conversion to dictionary format
    logger.info("\n=== Converting to Dictionary Format ===")
    dict_format = categorized_cases.get_categories_dict()
    logger.info(f"Dictionary format has {len(dict_format)} categories")
    for category_name, cases in dict_format.items():
        logger.info(f"  {category_name}: {len(cases)} cases")

    logger.info("\n=== Demonstration Complete ===")


def demonstrate_vs_dictionary_approach():
    """Demonstrate the difference between the class approach and dictionary approach."""
    logger.info("\n=== Class vs Dictionary Approach Comparison ===\n")

    # Class approach
    logger.info("Class Approach:")
    categorized_cases = CategorizedErrorCases()

    # Add a case
    case = create_sample_error_case("TRIAL_001", "PATIENT_001", "Diabetes")
    categorized_cases.add_case(
        category=ErrorCategory.EXCLUSION_CRITERIA_VIOLATION,
        case=case,
        gpt_categorization="exclusion_criteria_violation",
        gpt_reasoning="Patient meets exclusion criteria",
        model_used="gpt-5",
        cost=0.001
    )

    # Get statistics
    stats = categorized_cases.get_category_statistics()
    logger.info(f"  Statistics: {stats}")

    # Dictionary approach (old way)
    logger.info("\nDictionary Approach (old way):")
    categorized_dict: Dict[str, List[Dict[str, Any]]] = {
        category.value: [] for category in ErrorCategory
    }

    # Add a case
    categorized_dict["exclusion_criteria_violation"].append({
        'case': case,
        'gpt_categorization': 'exclusion_criteria_violation',
        'gpt_reasoning': 'Patient meets exclusion criteria',
        'model_used': 'gpt-5',
        'cost': 0.001
    })

    # Get statistics (manual calculation)
    total_cases = sum(len(cases) for cases in categorized_dict.values())
    stats_dict: Dict[str, Dict[str, Any]] = {}
    for category in ErrorCategory:
        category_cases = categorized_dict.get(category.value, [])
        if category_cases:
            stats_dict[category.value] = {
                'count': len(category_cases),
                'percentage': len(category_cases) / total_cases * 100 if total_cases > 0 else 0,
            }
        else:
            stats_dict[category.value] = {
                'count': 0,
                'percentage': 0.0,
            }
    logger.info(f"  Statistics: {stats_dict}")

    logger.info("\nBenefits of Class Approach:")
    logger.info("  - Type safety and better IDE support")
    logger.info("  - Encapsulated logic for common operations")
    logger.info("  - Easier to extend with new functionality")
    logger.info("  - Better error handling and validation")
    logger.info("  - Maintains backward compatibility")


if __name__ == "__main__":
    try:
        demonstrate_class_usage()
        demonstrate_vs_dictionary_approach()
    except Exception as e:
        logger.error(f"Error during demonstration: {e}")
        sys.exit(1)
