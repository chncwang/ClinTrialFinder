#!/usr/bin/env python3
"""
Error Cases Analysis Script

This script reads and analyzes error case JSON files from clinical trial filtering experiments.
It provides comprehensive analysis including statistics, patterns, and detailed examination of error cases.
It also uses GPT-5 (or GPT-4o as fallback) to categorize false positive errors into specific error types.

Usage:
    python -m scripts.analyze_error_cases <json_file_path> [options]

    Options:
        --gpt-api-key KEY    OpenAI API key for GPT categorization
        --categorize-gpt     Use GPT to categorize false positive errors
        --log-level LEVEL    Set logging level (DEBUG, INFO, WARNING, ERROR)
"""

import argparse
import sys
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from collections import Counter, defaultdict
import pandas as pd
from datetime import datetime
from enum import Enum

# Import our custom error case classes
from base.error_case import ErrorCaseCollection, ErrorCase
from base.gpt_client import GPTClient

# Global logger
logger = logging.getLogger(__name__)


class ErrorCategory(Enum):
    """Enum representing the three categories of clinical trial filtering errors."""

    EXCLUSION_CRITERIA_VIOLATION = "exclusion_criteria_violation"
    INCLUSION_CRITERIA_VIOLATION = "inclusion_criteria_violation"
    DATA_LABEL_ERROR = "data_label_error"

    @classmethod
    def get_all_values(cls) -> List[str]:
        """Get all possible category values as a list of strings."""
        return [category.value for category in cls]

    @classmethod
    def is_valid_category(cls, category_str: str) -> bool:
        """Check if a string represents a valid error category."""
        return category_str in cls.get_all_values()

    @classmethod
    def from_value(cls, value: str) -> Optional['ErrorCategory']:
        """Get an ErrorCategory enum from its string value."""
        for category in cls:
            if category.value == value:
                return category
        return None

    @classmethod
    def get_categories_dict(cls) -> Dict[str, str]:
        """Get a dictionary mapping category values to their descriptions."""
        descriptions = {
            "exclusion_criteria_violation": "Patient meets one or more exclusion criteria",
            "inclusion_criteria_violation": "Patient fails to meet one or more inclusion criteria",
            "data_label_error": "Benchmark label or data is likely incorrect"
        }
        return {category.value: descriptions.get(category.value, "Unknown category") for category in cls}


class CategorizedErrorCase:
    """Represents a single categorized error case with all its metadata."""

    def __init__(self, case: Any, gpt_categorization: str, gpt_reasoning: str,
                 model_used: str, cost: float):
        self.case = case
        self.gpt_categorization = gpt_categorization
        self.gpt_reasoning = gpt_reasoning
        self.model_used = model_used
        self.cost = cost

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for backward compatibility."""
        return {
            'case': self.case,
            'gpt_categorization': self.gpt_categorization,
            'gpt_reasoning': self.gpt_reasoning,
            'model_used': self.model_used,
            'cost': self.cost
        }


class CategorizedErrorCases:
    """Container for categorized error cases organized by error category."""

    def __init__(self):
        """Initialize with empty lists for each error category."""
        self._categories: Dict[str, List[CategorizedErrorCase]] = {
            category.value: [] for category in ErrorCategory
        }

    def add_case(self, category: ErrorCategory, case: Any, gpt_categorization: str,
                 gpt_reasoning: str, model_used: str, cost: float) -> None:
        """Add a categorized error case to the appropriate category."""
        categorized_case = CategorizedErrorCase(
            case=case,
            gpt_categorization=gpt_categorization,
            gpt_reasoning=gpt_reasoning,
            model_used=model_used,
            cost=cost
        )
        self._categories[category.value].append(categorized_case)

    def get_cases_for_category(self, category: ErrorCategory) -> List[CategorizedErrorCase]:
        """Get all cases for a specific category."""
        return self._categories[category.value]

    def get_all_cases(self) -> List[CategorizedErrorCase]:
        """Get all cases across all categories."""
        all_cases: List[CategorizedErrorCase] = []
        for cases in self._categories.values():
            all_cases.extend(cases)
        return all_cases

    def get_category_count(self, category: ErrorCategory) -> int:
        """Get the count of cases for a specific category."""
        return len(self._categories[category.value])

    def get_total_count(self) -> int:
        """Get the total count of all cases."""
        return sum(len(cases) for cases in self._categories.values())

    def get_categories_dict(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get the dictionary format for backward compatibility."""
        return {
            category_key: [case.to_dict() for case in cases]
            for category_key, cases in self._categories.items()
        }

    def items(self):
        """Iterate over category-value pairs for backward compatibility."""
        return self._categories.items()

    def values(self):
        """Get all case lists for backward compatibility."""
        return self._categories.values()

    def get(self, key: str, default: Optional[List[CategorizedErrorCase]] = None) -> Optional[List[CategorizedErrorCase]]:
        """Get cases for a category by key for backward compatibility."""
        return self._categories.get(key, default)

    def get_category_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for each error category."""
        stats: Dict[str, Dict[str, Any]] = {}
        total_count = self.get_total_count()

        for category in ErrorCategory:
            category_count = self.get_category_count(category)
            stats[category.value] = {
                'count': category_count,
                'percentage': (category_count / total_count * 100) if total_count > 0 else 0,
            }

        return stats

    def get_category_examples(self, max_examples: int = 3) -> Dict[str, List[Dict[str, Any]]]:
        """Get example cases from each category for analysis."""
        examples: Dict[str, List[Dict[str, Any]]] = {}

        for category in ErrorCategory:
            category_cases = self.get_cases_for_category(category)
            if category_cases:
                # Take up to max_examples cases from this category
                examples[category.value] = [case.to_dict() for case in category_cases[:max_examples]]
            else:
                examples[category.value] = []

        return examples

    def print_category_statistics(self) -> None:
        """Print category statistics in a formatted way."""
        stats = self.get_category_statistics()
        if not stats:
            logger.info("No category statistics available")
            return

        logger.info("-" * 80)
        logger.info("ERROR CATEGORY STATISTICS")
        logger.info("-" * 80)

        for category_value, category_stats in stats.items():
            count = category_stats['count']
            percentage = category_stats['percentage']

            logger.info(f"{category_value.replace('_', ' ').title()}:")
            logger.info(f"  Count: {count}")
            logger.info(f"  Percentage: {percentage:.1f}%")
            logger.info("")

    def print_category_examples(self, max_examples: int = 3) -> None:
        """Print example cases from each category."""
        examples = self.get_category_examples(max_examples)

        logger.info("-" * 80)
        logger.info(f"CATEGORY EXAMPLES (max {max_examples} per category)")
        logger.info("-" * 80)

        for category in ErrorCategory:
            category_examples = examples[category.value]
            logger.info(f"\n{category.value.replace('_', ' ').title()}:")

            if not category_examples:
                logger.info("  No examples available")
                continue

            for i, case_info in enumerate(category_examples, 1):
                case = case_info['case']
                logger.info(f"  Example {i}:")
                logger.info(f"    Trial ID: {case.trial_id}")
                logger.info(f"    Patient ID: {case.patient_id}")
                logger.info(f"    Disease: {case.disease_name}")
                logger.info(f"    Suitability Probability: {case.suitability_probability:.3f}")
                logger.info(f"    Reason: {case.reason[:100]}...")
                if 'gpt_reasoning' in case_info:
                    logger.info(f"    GPT Reasoning: {case_info['gpt_reasoning']}")

    def export_to_csv(self, output_path: str) -> bool:
        """Export the categorized results to CSV."""
        try:
            if self.get_total_count() == 0:
                return False

            # Flatten the categorized results
            export_data: List[Dict[str, Any]] = []
            for category_name, cases in self._categories.items():
                for case in cases:
                    export_data.append({
                        'disease_name': case.case.disease_name,
                        'trial_title': case.case.trial_title,
                        'trial_criteria': case.case.trial_criteria,
                        'text_summary': case.case.text_summary,
                        'gpt_categorization': category_name,
                        'gpt_reasoning': case.gpt_reasoning
                    })

            df = pd.DataFrame(export_data)  # type: ignore
            df.to_csv(output_path, index=False)  # type: ignore
            return True

        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            return False

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

    def __init__(self, json_file_path: str, gpt_api_key: str):
        """Initialize the analyzer with a JSON file path and optional GPT API key."""
        self.json_file_path = Path(json_file_path)
        self.collection: Optional[ErrorCaseCollection] = None
        self.gpt_client: Optional[GPTClient] = None
        self.total_cost = 0.0  # Track total API costs
        self.case_costs: Dict[str, float] = {}    # Track costs per case

        if gpt_api_key:
            try:
                self.gpt_client = GPTClient(api_key=gpt_api_key)
                logger.info("GPT client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize GPT client: {e}")
                raise e
        else:
            raise ValueError("GPT API key is required for categorization")

    def load_data(self) -> None:
        """Load and validate the JSON data."""
        if not self.json_file_path.exists():
            raise FileNotFoundError(f"File {self.json_file_path} does not exist.")

        # Use the ErrorCaseCollection to load and validate data
        self.collection = ErrorCaseCollection.from_file(str(self.json_file_path))
        logger.info(f"Successfully loaded {len(self.collection)} error cases from {self.json_file_path}")

    def categorize_false_positives_with_gpt(self) -> Dict[str, List[Dict[str, Any]]]:
        """Use GPT to categorize false positive error cases into specific error types."""
        if not self.gpt_client:
            logger.error("GPT client not available. Skipping categorization.")
            raise ValueError("GPT client not available. Skipping categorization.")

        if not self.collection:
            logger.warning("No data loaded. Use load_data() first.")
            return {}

        # Filter for false positive cases only
        false_positive_cases: List[ErrorCase] = [case for case in self.collection if case.is_false_positive]
        logger.info(f"Found {len(false_positive_cases)} false positive cases to categorize")

        if not false_positive_cases:
            logger.info("No false positive cases found for categorization")
            return {}

        # Initialize categorized cases using the new class
        categorized_cases = CategorizedErrorCases()

        # Use GPT-5 for categorization
        model = "gpt-5"

        try:
            logger.info(f"Attempting categorization with {model}...")
            for i, case in enumerate(false_positive_cases):
                logger.info(f"Processing case {i+1}/{len(false_positive_cases)}: {case.trial_id}")

                try:
                    categorization, reasoning, cost = self._categorize_single_case(case, model)
                    # Track costs
                    self.total_cost += cost
                    self.case_costs[case.trial_id] = cost

                    categorized_cases.add_case(
                        category=categorization,
                        case=case,
                        gpt_categorization=categorization.value,
                        gpt_reasoning=reasoning,
                        model_used=model,
                        cost=cost
                    )
                except Exception as case_error:
                    logger.error(f"Failed to categorize case {case.trial_id}: {case_error}")
                    # Continue with next case instead of failing the entire process
                    raise case_error

                # Add a small delay to avoid rate limiting
                import time
                time.sleep(0.1)

            logger.info(f"Successfully processed all cases using {model}")

        except Exception as e:
            logger.error(f"Failed to categorize with {model}: {e}")
            raise e

        # Log summary of categorization
        logger.info("-"*80)
        logger.info("GPT CATEGORIZATION RESULTS")
        logger.info("-"*80)
        logger.info(f"Model used: {model}")
        # Log summary for each category
        for category in ErrorCategory:
            count = categorized_cases.get_category_count(category)
            description = category.value.replace('_', ' ').title()
            logger.info(f"{description}: {count}")

        return categorized_cases.get_categories_dict()

    def categorize_false_positives_with_gpt_class(self) -> 'CategorizedErrorCases':
        """Use GPT to categorize false positive error cases and return the class directly."""
        if not self.gpt_client:
            logger.error("GPT client not available. Skipping categorization.")
            raise ValueError("GPT client not available. Skipping categorization.")

        if not self.collection:
            logger.warning("No data loaded. Use load_data() first.")
            return CategorizedErrorCases()

        # Filter for false positive cases only
        false_positive_cases: List[ErrorCase] = [case for case in self.collection if case.is_false_positive]
        logger.info(f"Found {len(false_positive_cases)} false positive cases to categorize")

        if not false_positive_cases:
            logger.info("No false positive cases found for categorization")
            return CategorizedErrorCases()

        # Initialize categorized cases using the new class
        categorized_cases = CategorizedErrorCases()

        # Use GPT-5 for categorization
        model = "gpt-5"

        try:
            logger.info(f"Attempting categorization with {model}...")
            for i, case in enumerate(false_positive_cases):
                logger.info(f"Processing case {i+1}/{len(false_positive_cases)}: {case.trial_id}")

                try:
                    categorization, reasoning, cost = self._categorize_single_case(case, model)
                    # Track costs
                    self.total_cost += cost
                    self.case_costs[case.trial_id] = cost

                    categorized_cases.add_case(
                        category=categorization,
                        case=case,
                        gpt_categorization=categorization.value,
                        gpt_reasoning=reasoning,
                        model_used=model,
                        cost=cost
                    )
                except Exception as case_error:
                    logger.error(f"Failed to categorize case {case.trial_id}: {case_error}")
                    # Continue with next case instead of failing the entire process
                    raise case_error

                # Add a small delay to avoid rate limiting
                import time
                time.sleep(0.1)

            logger.info(f"Successfully processed all cases using {model}")

        except Exception as e:
            logger.error(f"Failed to categorize with {model}: {e}")
            raise e

        # Log summary of categorization
        logger.info("-"*80)
        logger.info("GPT CATEGORIZATION RESULTS")
        logger.info("-"*80)
        logger.info(f"Model used: {model}")
        # Log summary for each category
        for category in ErrorCategory:
            count = categorized_cases.get_category_count(category)
            description = category.value.replace('_', ' ').title()
            logger.info(f"{description}: {count}")

        return categorized_cases

    def _categorize_single_case(self, case: Any, model: str) -> tuple[ErrorCategory, str, float]:
        """Categorize a single false positive case using GPT.

        Returns:
            tuple: (ErrorCategory, str, float) - The error category, reasoning, and API cost
        """
        if not self.gpt_client:
            raise RuntimeError("GPT client not available")

        system_prompt = """You are a clinical trial domain expert.
You specialize in interpreting patient records and clinical trial eligibility criteria.
Your role is to analyze trial matching errors with deep knowledge of inclusion and exclusion logic.
You must be precise, concise, and output only structured JSON — no extra text, no markdown, no explanations."""

        user_prompt = f"""Categorize the following false positive error case into one of:
- "exclusion_criteria_violation": patient meets ≥1 exclusion criterion.
- "inclusion_criteria_violation": patient fails to meet ≥1 inclusion criterion.
- "data_label_error": benchmark label or data is likely incorrect.

Return JSON only, in this schema:
{{
  "reasoning": "≤60 words explaining your choice",
  "category": "exclusion_criteria_violation" | "inclusion_criteria_violation" | "data_label_error",
}}

Case details:

Clinical Record:
{case.text_summary}

Trial Title:
{case.trial_title}

Trial Criteria:
{case.trial_criteria}
"""

        try:
            response, cost = self.gpt_client.call_gpt(
                prompt=user_prompt,
                system_role=system_prompt,
                model=model,
            )

            # Log the cost for this categorization
            logger.info(f"GPT API cost for case {case.trial_id}: ${cost:.6f}")

            # Clean and validate the response
            response = response.strip()

            # Try to parse as JSON
            import json
            try:
                parsed_response = json.loads(response)
                category_str = parsed_response.get("category", "").lower()
                reasoning = parsed_response.get("reasoning", "No reasoning provided")
                logger.info(f"Categorization: {category_str}, Reasoning: {reasoning}")

                # Check if the response contains any valid category
                for category in ErrorCategory:
                    if category.value in category_str:
                        return category, reasoning, cost

                # If no valid category found, log and raise error
                error_msg = f"Unexpected GPT response category for case {case.trial_id}: {category_str}. Expected one of: {ErrorCategory.get_all_values()}"
                logger.warning(error_msg)
                raise ValueError(error_msg)

            except json.JSONDecodeError:
                # Fallback: try to extract category from plain text response
                response_lower = response.lower()
                for category in ErrorCategory:
                    if category.value in response_lower:
                        return category, "Category extracted from text response (JSON parsing failed)", cost

                # If no valid category found, log and raise error
                error_msg = f"Unexpected GPT response for case {case.trial_id}: {response}. Expected JSON with category and reasoning. Expected categories: {ErrorCategory.get_all_values()}"
                logger.warning(error_msg)
                raise ValueError(error_msg)

        except Exception as e:
            error_msg = f"Failed to categorize case {case.trial_id} with {model}: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for the error cases."""
        if not self.collection:
            return {}

        return self.collection.get_statistics()

    def get_category_statistics(self, categorized_cases: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
        """Get statistics for each error category."""
        if not categorized_cases:
            return {}

        stats: Dict[str, Dict[str, Any]] = {}
        for category in ErrorCategory:
            category_cases = categorized_cases.get(category.value, [])
            stats[category.value] = {
                'count': len(category_cases),
                'percentage': len(category_cases) / sum(len(cases) for cases in categorized_cases.values()) * 100 if any(categorized_cases.values()) else 0,
            }

        return stats

    def print_category_statistics(self, categorized_cases: Dict[str, List[Dict[str, Any]]]) -> None:
        """Print category statistics in a formatted way."""
        stats = self.get_category_statistics(categorized_cases)
        if not stats:
            logger.info("No category statistics available")
            return

        logger.info("-" * 80)
        logger.info("ERROR CATEGORY STATISTICS")
        logger.info("-" * 80)

        for category_value, category_stats in stats.items():
            count = category_stats['count']
            percentage = category_stats['percentage']

            logger.info(f"{category_value.replace('_', ' ').title()}:")
            logger.info(f"  Count: {count}")
            logger.info(f"  Percentage: {percentage:.1f}%")
            logger.info("")

    def get_category_examples(self, categorized_cases: Dict[str, List[Dict[str, Any]]], max_examples: int = 3) -> Dict[str, List[Dict[str, Any]]]:
        """Get example cases from each category for analysis."""
        examples: Dict[str, List[Dict[str, Any]]] = {}

        for category in ErrorCategory:
            category_cases = categorized_cases.get(category.value, [])
            if category_cases:
                # Take up to max_examples cases from this category
                examples[category.value] = category_cases[:max_examples]
            else:
                examples[category.value] = []

        return examples

    def print_category_examples(self, categorized_cases: Dict[str, List[Dict[str, Any]]], max_examples: int = 3) -> None:
        """Print example cases from each category."""
        examples = self.get_category_examples(categorized_cases, max_examples)

        logger.info("-" * 80)
        logger.info(f"CATEGORY EXAMPLES (max {max_examples} per category)")
        logger.info("-" * 80)

        for category in ErrorCategory:
            category_examples = examples[category.value]
            logger.info(f"\n{category.value.replace('_', ' ').title()}:")

            if not category_examples:
                logger.info("  No examples available")
                continue

            for i, case_info in enumerate(category_examples, 1):
                case = case_info['case']
                logger.info(f"  Example {i}:")
                logger.info(f"    Trial ID: {case.trial_id}")
                logger.info(f"    Patient ID: {case.patient_id}")
                logger.info(f"    Disease: {case.disease_name}")
                logger.info(f"    Suitability Probability: {case.suitability_probability:.3f}")
                logger.info(f"    Reason: {case.reason[:100]}...")
                if 'gpt_reasoning' in case_info:
                    logger.info(f"    GPT Reasoning: {case_info['gpt_reasoning']}")

    def validate_categorization(self, categorization: str) -> bool:
        """Validate if a categorization string is a valid error category."""
        return ErrorCategory.is_valid_category(categorization)

    def get_category_enum(self, categorization: str) -> Optional[ErrorCategory]:
        """Get the ErrorCategory enum from a categorization string."""
        return ErrorCategory.from_value(categorization)

    def print_available_categories(self) -> None:
        """Print all available error categories with their descriptions."""
        logger.info("-" * 80)
        logger.info("AVAILABLE ERROR CATEGORIES")
        logger.info("-" * 80)

    def print_cost_statistics(self) -> None:
        """Print cost statistics for GPT API usage."""
        if self.total_cost == 0.0:
            logger.info("No GPT API costs recorded")
            return

        logger.info("-" * 80)
        logger.info("GPT API COST STATISTICS")
        logger.info("-" * 80)
        logger.info(f"Total cost: ${self.total_cost:.6f}")
        logger.info(f"Number of cases processed: {len(self.case_costs)}")
        if self.case_costs:
            avg_cost = self.total_cost / len(self.case_costs)
            min_cost = min(self.case_costs.values())
            max_cost = max(self.case_costs.values())
            logger.info(f"Average cost per case: ${avg_cost:.6f}")
            logger.info(f"Minimum cost per case: ${min_cost:.6f}")
            logger.info(f"Maximum cost per case: ${max_cost:.6f}")
        logger.info("-" * 80)

    def get_cost_summary(self) -> Dict[str, Any]:
        """Get cost summary statistics as a dictionary."""
        if self.total_cost == 0.0:
            return {
                'total_cost': 0.0,
                'cases_processed': 0,
                'average_cost_per_case': 0.0,
                'min_cost_per_case': 0.0,
                'max_cost_per_case': 0.0
            }

        return {
            'total_cost': self.total_cost,
            'cases_processed': len(self.case_costs),
            'average_cost_per_case': self.total_cost / len(self.case_costs),
            'min_cost_per_case': min(self.case_costs.values()),
            'max_cost_per_case': max(self.case_costs.values()),
            'case_costs': self.case_costs
        }



    def export_categorized_results(self, categorized_cases: Dict[str, List[Dict[str, Any]]], output_path: str) -> bool:
        """Export the GPT-categorized results to CSV and JSON."""
        try:
            if not categorized_cases:
                logger.warning("No categorized cases to export")
                return False

            # Flatten the categorized results
            export_data: List[Dict[str, Any]] = []
            for category_name, cases in categorized_cases.items():
                for case_info in cases:
                    case = case_info['case']
                    export_data.append({
                        'disease_name': case.disease_name,
                        'trial_title': case.trial_title,
                        'trial_criteria': case.trial_criteria,
                        'text_summary': case.text_summary,
                        'gpt_categorization': category_name,
                        'gpt_reasoning': case_info.get('gpt_reasoning', '')
                    })

            # Export to CSV
            df = pd.DataFrame(export_data)  # type: ignore
            df.to_csv(output_path, index=False)  # type: ignore
            logger.info(f"Exported {len(df)} categorized cases to {output_path}")  # type: ignore

            # Export to JSON (same fields, human readable)
            json_path = output_path.replace('.csv', '.json')
            import json
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Exported {len(export_data)} categorized cases to {json_path}")

            return True

        except Exception as e:
            logger.error(f"Error exporting categorized results to CSV/JSON: {e}")
            return False

    def export_categorized_results_from_class(self, categorized_cases: 'CategorizedErrorCases', output_path: str) -> bool:
        """Export the GPT-categorized results to CSV and JSON using the CategorizedErrorCases class."""
        try:
            if not categorized_cases or categorized_cases.get_total_count() == 0:
                logger.warning("No categorized cases to export")
                return False

            # Use the class's built-in export method for CSV
            csv_success = categorized_cases.export_to_csv(output_path)

                                    # Also export to JSON (same fields, human readable)
            if csv_success:
                json_path = output_path.replace('.csv', '.json')
                import json
                export_data: List[Dict[str, Any]] = []
                for category_name, cases in categorized_cases.items():
                    for case_info in cases:
                        case = case_info.case
                        export_data.append({
                            'disease_name': case.disease_name,
                            'trial_title': case.trial_title,
                            'trial_criteria': case.trial_criteria,
                            'text_summary': case.text_summary,
                            'gpt_categorization': category_name,
                            'gpt_reasoning': case_info.gpt_reasoning
                        })

                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False)
                logger.info(f"Exported {len(export_data)} categorized cases to {json_path}")

            return csv_success

        except Exception as e:
            logger.error(f"Error exporting categorized results to CSV/JSON: {e}")
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
        description="Analyze clinical trial error case JSON files with GPT categorization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m scripts.analyze_error_cases results/error_cases_20250825_063056.json
    python -m scripts.analyze_error_cases results/error_cases_20250825_063056.json --gpt-api-key sk-...
        """
    )

    parser.add_argument('json_file', help='Path to the error cases JSON file')

    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Set the logging level (default: INFO)')
    parser.add_argument('--gpt-api-key', help='OpenAI API key for GPT categorization (overrides OPENAI_API_KEY env var)')
    parser.add_argument('--output', help='Output CSV file path for categorized results (default: auto-generated filename)')

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    # Get GPT API key for categorization
    gpt_api_key = args.gpt_api_key or os.getenv('OPENAI_API_KEY')
    if not gpt_api_key:
        logger.error("GPT API key not provided. Use --gpt-api-key or set OPENAI_API_KEY environment variable.")
        sys.exit(1)

    # Initialize analyzer
    analyzer = ErrorCaseAnalyzer(args.json_file, gpt_api_key)

    # Load data
    try:
        analyzer.load_data()
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        sys.exit(1)

    # Log summary
    analyzer.log_summary()

    # Perform GPT categorization
    logger.info("Starting GPT categorization of false positive cases...")
    categorized_cases = analyzer.categorize_false_positives_with_gpt()

    if categorized_cases:
        # Export categorized results
        output_path = args.output or f"categorized_error_cases_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        if analyzer.export_categorized_results(categorized_cases, output_path):
            logger.info("Categorized results export completed successfully")
        else:
            logger.error("Categorized results export failed")

        # Demonstrate enum usage
        logger.info("Demonstrating ErrorCategory enum usage...")
        analyzer.print_available_categories()
        analyzer.print_category_statistics(categorized_cases)
        analyzer.print_category_examples(categorized_cases, max_examples=2)

        # Display cost statistics
        analyzer.print_cost_statistics()



    # Final cost summary
    if analyzer.total_cost > 0:
        logger.info("=" * 80)
        logger.info("FINAL COST SUMMARY")
        logger.info("=" * 80)
        cost_summary = analyzer.get_cost_summary()
        logger.info(f"Total GPT API cost: ${cost_summary['total_cost']:.6f}")
        logger.info(f"Cost per case: ${cost_summary['average_cost_per_case']:.6f}")
        logger.info("=" * 80)


if __name__ == "__main__":
    main()
