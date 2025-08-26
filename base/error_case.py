"""
Error Case Module

This module defines the ErrorCase class for representing clinical trial filtering errors.
It provides a structured way to handle error case data with validation and utility methods.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime
import json


@dataclass
class ErrorCase:
    """Represents a single clinical trial filtering error case."""

    patient_id: str
    disease_name: str
    trial_id: str
    trial_title: str
    trial_criteria: str
    error_type: str  # 'false_positive' or 'false_negative'
    suitability_probability: float
    reason: str
    ground_truth_relevant: bool
    predicted_eligible: bool
    original_relevance_score: float
    full_medical_record: str = ""
    timestamp: Optional[str] = None

    def __post_init__(self):
        """Validate the error case data after initialization."""
        if self.error_type not in ['false_positive', 'false_negative']:
            raise ValueError(f"Invalid error_type: {self.error_type}. Must be 'false_positive' or 'false_negative'")

        if not 0.0 <= self.suitability_probability <= 1.0:
            raise ValueError(f"Invalid suitability_probability: {self.suitability_probability}. Must be between 0.0 and 1.0")

        if not self.patient_id or not self.disease_name or not self.trial_id:
            raise ValueError("patient_id, disease_name, and trial_id cannot be empty")

    @property
    def is_false_positive(self) -> bool:
        """Check if this is a false positive error."""
        return self.error_type == 'false_positive'

    @property
    def is_false_negative(self) -> bool:
        """Check if this is a false negative error."""
        return self.error_type == 'false_negative'

    @property
    def confidence_level(self) -> str:
        """Get a human-readable confidence level based on suitability probability."""
        if self.suitability_probability >= 0.9:
            return "Very High"
        elif self.suitability_probability >= 0.8:
            return "High"
        elif self.suitability_probability >= 0.7:
            return "Medium"
        elif self.suitability_probability >= 0.6:
            return "Low"
        else:
            return "Very Low"

    def to_dict(self) -> Dict[str, Any]:
        """Convert the error case to a dictionary."""
        return {
            'patient_id': self.patient_id,
            'disease_name': self.disease_name,
            'trial_id': self.trial_id,
            'trial_title': self.trial_title,
            'trial_criteria': self.trial_criteria,
            'error_type': self.error_type,
            'suitability_probability': self.suitability_probability,
            'reason': self.reason,
            'ground_truth_relevant': self.ground_truth_relevant,
            'predicted_eligible': self.predicted_eligible,
            'original_relevance_score': self.original_relevance_score,
            'full_medical_record': self.full_medical_record,
            'timestamp': self.timestamp
        }

    def to_json(self) -> str:
        """Convert the error case to a JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ErrorCase':
        """Create an ErrorCase instance from a dictionary."""
        return cls(**data)

    def __str__(self) -> str:
        """String representation of the error case."""
        return f"ErrorCase(patient={self.patient_id}, disease={self.disease_name}, trial={self.trial_id}, type={self.error_type})"

    def __repr__(self) -> str:
        """Detailed string representation of the error case."""
        return f"ErrorCase(patient_id='{self.patient_id}', disease_name='{self.disease_name}', trial_id='{self.trial_id}', error_type='{self.error_type}', suitability_probability={self.suitability_probability})"


@dataclass
class ErrorCaseCollection:
    """Collection of error cases with analysis capabilities."""

    error_cases: List[ErrorCase] = field(default_factory=list)
    timestamp: Optional[str] = None
    source_file: Optional[str] = None

    def __post_init__(self):
        """Set timestamp if not provided."""
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def add_error_case(self, error_case: ErrorCase) -> None:
        """Add an error case to the collection."""
        self.error_cases.append(error_case)

    def add_from_dict(self, data: Dict[str, Any]) -> None:
        """Add an error case from a dictionary."""
        error_case = ErrorCase.from_dict(data)
        self.add_error_case(error_case)

    def get_by_error_type(self, error_type: str) -> List[ErrorCase]:
        """Get all error cases of a specific type."""
        return [case for case in self.error_cases if case.error_type == error_type]

    def get_false_positives(self) -> List[ErrorCase]:
        """Get all false positive error cases."""
        return self.get_by_error_type('false_positive')

    def get_false_negatives(self) -> List[ErrorCase]:
        """Get all false negative error cases."""
        return self.get_by_error_type('false_negative')

    def get_high_confidence_errors(self, threshold: float = 0.8) -> List[ErrorCase]:
        """Get error cases with high confidence (suitability probability above threshold)."""
        return [case for case in self.error_cases if case.suitability_probability >= threshold]

    def get_low_confidence_errors(self, threshold: float = 0.6) -> List[ErrorCase]:
        """Get error cases with low confidence (suitability probability below threshold)."""
        return [case for case in self.error_cases if case.suitability_probability < threshold]

    @property
    def total_count(self) -> int:
        """Get the total number of error cases."""
        return len(self.error_cases)

    @property
    def false_positive_count(self) -> int:
        """Get the count of false positive errors."""
        return len(self.get_false_positives())

    @property
    def false_negative_count(self) -> int:
        """Get the count of false negative errors."""
        return len(self.get_false_negatives())

    @property
    def unique_patients(self) -> List[str]:
        """Get list of unique patient IDs."""
        return list(set(case.patient_id for case in self.error_cases))

    @property
    def unique_trials(self) -> List[str]:
        """Get list of unique trial IDs."""
        return list(set(case.trial_id for case in self.error_cases))

    @property
    def unique_diseases(self) -> List[str]:
        """Get list of unique disease names."""
        return list(set(case.disease_name for case in self.error_cases))

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the error cases."""
        if not self.error_cases:
            return {}

        prob_values = [case.suitability_probability for case in self.error_cases]

        stats = {
            'total_error_cases': self.total_count,
            'false_positives': self.false_positive_count,
            'false_negatives': self.false_negative_count,
            'unique_patients': len(self.unique_patients),
            'unique_trials': len(self.unique_trials),
            'unique_diseases': len(self.unique_diseases),
            'timestamp': self.timestamp,
            'avg_suitability_probability': sum(prob_values) / len(prob_values),
            'min_suitability_probability': min(prob_values),
            'max_suitability_probability': max(prob_values)
        }

        return stats

    def to_dict(self) -> Dict[str, Any]:
        """Convert the collection to a dictionary."""
        return {
            'timestamp': self.timestamp,
            'total_error_cases': self.total_count,
            'false_positives': self.false_positive_count,
            'false_negatives': self.false_negative_count,
            'error_cases': [case.to_dict() for case in self.error_cases]
        }

    def to_json(self) -> str:
        """Convert the collection to a JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    def save_to_file(self, file_path: str) -> None:
        """Save the collection to a JSON file."""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(self.to_json())

    @classmethod
    def from_file(cls, file_path: str) -> 'ErrorCaseCollection':
        """Create an ErrorCaseCollection from a JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        collection = cls(
            timestamp=data.get('timestamp'),
            source_file=file_path
        )

        for case_data in data.get('error_cases', []):
            collection.add_from_dict(case_data)

        return collection

    def __len__(self) -> int:
        """Return the number of error cases."""
        return len(self.error_cases)

    def __getitem__(self, index: int) -> ErrorCase:
        """Get an error case by index."""
        return self.error_cases[index]

    def __iter__(self):
        """Iterate over error cases."""
        return iter(self.error_cases)
