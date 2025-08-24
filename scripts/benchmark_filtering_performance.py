#!/usr/bin/env python3
"""
Benchmark script for evaluating filtering performance on TREC 2021 dataset.

This script evaluates the performance of clinical trial filtering algorithms
by comparing predicted eligible trials against ground truth relevance judgments.

PERFORMANCE OPTIMIZATION:
The script implements intelligent caching to minimize API calls:
- Patient conditions are extracted only once and cached for reuse
- Significantly reduces API costs and improves performance
- Cache statistics are tracked and reported for analysis

Usage:
    python scripts/benchmark_filtering_performance.py [options]

Example:
    python scripts/benchmark_filtering_performance.py \
        --dataset-path dataset/trec_2021 \
        --output results/benchmark_results.json \
        --api-key $OPENAI_API_KEY

    # Run benchmark on a specific patient ID only
    python scripts/benchmark_filtering_performance.py \
        --patient-id "patient_123" \
        --api-key $OPENAI_API_KEY

    # Run benchmark with limited total number of trials for faster testing
    # Note: Trials are allocated proportionally across patients, uses deterministic sampling (seed=42) for consistent results
    # Trials with original label 1 are excluded to focus sampling on label 0 and 2 trials
    python scripts/benchmark_filtering_performance.py \
        --max-trials 100 \
        --api-key $OPENAI_API_KEY

    # Run benchmark with limited number of patients for faster testing
    python scripts/benchmark_filtering_performance.py \
        --max-patients 50 \
        --api-key $OPENAI_API_KEY

    # Run benchmark only on cancer patients
    python scripts/benchmark_filtering_performance.py \
        --cancer-only \
        --api-key $OPENAI_API_KEY

    # Run benchmark on cancer patients with limited total trials
    # Note: Trials with original label 1 are excluded to focus sampling on label 0 and 2 trials
    python scripts/benchmark_filtering_performance.py \
        --cancer-only \
        --max-trials 100 \
        --api-key $OPENAI_API_KEY

    # Run benchmark with title-only evaluation (faster but less accurate)
    python scripts/benchmark_filtering_performance.py \
        --title-only \
        --api-key $OPENAI_API_KEY

    # Run benchmark with title-only evaluation on cancer patients with limited trials
    # Note: Trials with original label 1 are excluded to focus sampling on label 0 and 2 trials
    python scripts/benchmark_filtering_performance.py \
        --title-only \
        --cancer-only \
        --max-trials 100 \
        --api-key $OPENAI_API_KEY

    # Export conditions cache for analysis
    python scripts/benchmark_filtering_performance.py \
        --export-conditions-cache \
        --api-key $OPENAI_API_KEY

    # Show detailed cache status
    python scripts/benchmark_filtering_performance.py \
        --show-cache-status \
        --api-key $OPENAI_API_KEY

    # Run benchmark in strict cache mode (throws exception when cache is not hit)
    python scripts/benchmark_filtering_performance.py \
        --strict-cache-mode \
        --api-key $OPENAI_API_KEY
"""

import argparse
import json
import os
import sys
import random
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from enum import Enum
import logging
from tqdm import tqdm
import time


# Add parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from base.logging_config import setup_logging
from base.clinical_trial import ClinicalTrial
from base.disease_expert import extract_disease_from_record, is_oncology_disease, extract_conditions_from_content
from base.trial_expert import GPTTrialFilter, TrialFailureReason
from base.gpt_client import GPTClient

logger: logging.Logger = logging.getLogger(__name__)
# log_file will be set after parsing arguments
log_file: str = ""


class Patient:
    """
    A class to represent a patient case for clinical trial matching.

    This class encapsulates patient information from the TREC 2021 dataset,
    including the patient ID and medical record text content.
    """

    def __init__(self, patient_id: str, medical_record: str):
        """
        Initialize a Patient instance.

        Args:
            patient_id: Unique identifier for the patient
            medical_record: The medical record text describing the patient case
        """
        self.patient_id = patient_id
        self.medical_record = medical_record

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Patient':
        """
        Create a Patient instance from a dictionary.

        Args:
            data: Dictionary containing patient data with keys '_id' and 'text'

        Returns:
            Patient instance
        """
        return cls(
            patient_id=data['_id'],
            medical_record=data['text']
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the Patient instance to a dictionary.

        Returns:
            Dictionary representation of the patient
        """
        return {
            '_id': self.patient_id,
            'medical_record': self.medical_record
        }

    def get_patient_id(self) -> str:
        """
        Get the patient ID from the patient ID.
        In TREC dataset, patient_id matches patient_id.

        Returns:
            Patient ID string
        """
        return self.patient_id

    def get_text_summary(self, max_length: int = 100) -> str:
        """
        Get a summary of the patient medical record text.

        Args:
            max_length: Maximum length of the summary

        Returns:
            Truncated text summary
        """
        if len(self.medical_record) <= max_length:
            return self.medical_record
        return self.medical_record[:max_length] + "..."

    def __str__(self) -> str:
        """String representation of the patient."""
        return f"Patient(id={self.patient_id}, medical_record='{self.get_text_summary(50)}')"

    def __repr__(self) -> str:
        """Detailed string representation of the patient."""
        return f"Patient(patient_id='{self.patient_id}', medical_record='{self.medical_record}')"

    def __eq__(self, other: Any) -> bool:
        """Check if two patients are equal."""
        if not isinstance(other, Patient):
            return False
        return (self.patient_id == other.patient_id and
                self.medical_record == other.medical_record)

    def __hash__(self) -> int:
        """Hash based on patient_id."""
        return hash(self.patient_id)


class RelevanceJudgment:
    """
    A class to represent a single relevance judgment.

    This class encapsulates a single relevance judgment from the TREC 2021 dataset,
    including the patient ID, trial ID, and relevance score.
    """

    def __init__(self, patient_id: str, trial_id: str, relevance_score: int):
        """
        Initialize a RelevanceJudgment instance.

        Args:
            patient_id: Unique identifier for the patient
            trial_id: Unique identifier for the trial (NCT ID)
            relevance_score: Relevance score (0 = not relevant, >0 = relevant)
        """
        self.patient_id = patient_id
        self.trial_id = trial_id
        self.relevance_score = relevance_score

    @classmethod
    def from_tsv_line(cls, line: str) -> Optional['RelevanceJudgment']:
        """
        Create a RelevanceJudgment instance from a TSV line.

        Args:
            line: TSV line with format: patient_id\ttrial_id\trelevance_score

        Returns:
            RelevanceJudgment instance or None if line is invalid
        """
        parts = line.strip().split('\t')
        if len(parts) == 3:
            patient_id, trial_id, relevance = parts
            # Skip header row
            if relevance == 'score':
                return None
            try:
                return cls(patient_id, trial_id, int(relevance))
            except ValueError:
                # Skip lines that can't be converted to int
                return None
        return None

    def is_relevant(self) -> bool:
        """
        Check if the trial is considered relevant.

        Returns:
            True if relevance_score > 0, False otherwise
        """
        return self.relevance_score > 0

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the RelevanceJudgment instance to a dictionary.

        Returns:
            Dictionary representation of the relevance judgment
        """
        return {
            'patient_id': self.patient_id,
            'trial_id': self.trial_id,
            'relevance_score': self.relevance_score
        }

    def __str__(self) -> str:
        """String representation of the relevance judgment."""
        return f"RelevanceJudgment(patient_id='{self.patient_id}', trial_id='{self.trial_id}', relevance_score={self.relevance_score})"

    def __repr__(self) -> str:
        """Detailed string representation of the relevance judgment."""
        return f"RelevanceJudgment(patient_id='{self.patient_id}', trial_id='{self.trial_id}', relevance_score={self.relevance_score})"

    def __eq__(self, other: Any) -> bool:
        """Check if two RelevanceJudgment instances are equal."""
        if not isinstance(other, RelevanceJudgment):
            return False
        return (self.patient_id == other.patient_id and
                self.trial_id == other.trial_id and
                self.relevance_score == other.relevance_score)

    def __hash__(self) -> int:
        """Hash value for the relevance judgment."""
        return hash((self.patient_id, self.trial_id, self.relevance_score))


class ErrorType(Enum):
    """Enumeration of error types for trial evaluation."""
    CORRECT = "correct"
    FALSE_POSITIVE = "false_positive"
    FALSE_NEGATIVE = "false_negative"


class TrialEvaluationResult:
    """
    A class to represent the result of evaluating a single trial for a patient.

    This class stores the trial evaluation details including the prediction,
    ground truth, and evaluation metadata for error analysis.
    """

    def __init__(self,
                 trial_id: str,
                 trial_title: str,
                 predicted_eligible: bool,
                 ground_truth_relevant: bool,
                 suitability_probability: float,
                 reason: str,
                 api_cost: float,
                 original_relevance_score: Optional[int] = None):
        """
        Initialize a TrialEvaluationResult instance.

        Args:
            trial_id: Unique identifier for the trial
            trial_title: Title of the clinical trial
            predicted_eligible: Whether the trial was predicted as eligible
            ground_truth_relevant: Whether the trial is actually relevant (ground truth)
            suitability_probability: Probability score from GPT evaluation
            reason: Reasoning provided by GPT for the prediction
            api_cost: API cost for this trial evaluation
            original_relevance_score: Original relevance score from dataset (0, 1, or 2)
        """
        self.trial_id = trial_id
        self.trial_title = trial_title
        self.predicted_eligible = predicted_eligible
        self.ground_truth_relevant = ground_truth_relevant
        self.suitability_probability = suitability_probability
        self.reason = reason
        self.api_cost = api_cost
        self.original_relevance_score = original_relevance_score

    @property
    def is_error_case(self) -> bool:
        """Check if this is an error case (incorrect prediction or failed recall)."""
        return self.predicted_eligible != self.ground_truth_relevant

    @property
    def error_type(self) -> ErrorType:
        """Get the type of error for this trial evaluation."""
        if not self.is_error_case:
            return ErrorType.CORRECT
        elif self.predicted_eligible and not self.ground_truth_relevant:
            return ErrorType.FALSE_POSITIVE
        else:  # not predicted_eligible and ground_truth_relevant
            return ErrorType.FALSE_NEGATIVE

    def to_dict(self) -> Dict[str, Any]:
        """Convert the result to a dictionary representation."""
        return {
            'trial_id': self.trial_id,
            'trial_title': self.trial_title,
            'predicted_eligible': self.predicted_eligible,
            'ground_truth_relevant': self.ground_truth_relevant,
            'suitability_probability': self.suitability_probability,
            'reason': self.reason,
            'api_cost': self.api_cost,
            'is_error_case': self.is_error_case,
            'error_type': self.error_type.value,
            'original_relevance_score': self.original_relevance_score if self.original_relevance_score is not None else 'N/A'
        }

    def __str__(self) -> str:
        """String representation of the trial evaluation result."""
        return f"TrialEvaluationResult(trial_id='{self.trial_id}', predicted_eligible={self.predicted_eligible}, ground_truth_relevant={self.ground_truth_relevant}, original_relevance_score={self.original_relevance_score if self.original_relevance_score is not None else 'N/A'}, suitability_probability={self.suitability_probability:.3f})"

    def __repr__(self) -> str:
        """Detailed string representation of the trial evaluation result."""
        return f"TrialEvaluationResult(trial_id='{self.trial_id}', trial_title='{self.trial_title}', predicted_eligible={self.predicted_eligible}, ground_truth_relevant={self.ground_truth_relevant}, original_relevance_score={self.original_relevance_score if self.original_relevance_score is not None else 'N/A'}, suitability_probability={self.suitability_probability:.3f}, reason='{self.reason}', api_cost={self.api_cost:.4f})"


class PatientEvaluationResult:
    """
    A class to represent the results of evaluating filtering performance for a single patient.

    This class encapsulates all the metrics and information from evaluating a patient's
    clinical trial filtering performance, including precision, recall, F1-score, and
    processing statistics.
    """

    def __init__(self,
                 patient_id: str,
                 ground_truth_count: int,
                 retrieved_count: int,
                 predicted_eligible_count: int,
                 true_positives: int,
                 false_positives: int,
                 false_negatives: int,
                 precision: float,
                 recall: float,
                 f1_score: float,
                 accuracy: float,
                 processing_time: float,
                 api_cost: float,
                 error: Optional[str] = None,
                 skipped: bool = False,
                 trial_evaluation_results: Optional[List['TrialEvaluationResult']] = None):
        """
        Initialize a PatientEvaluationResult instance.

        Args:
            patient_id: Unique identifier for the patient
            ground_truth_count: Number of ground truth relevant trials
            retrieved_count: Total number of trials retrieved for evaluation
            predicted_eligible_count: Number of trials predicted as eligible
            true_positives: Number of true positive predictions
            false_positives: Number of false positive predictions
            false_negatives: Number of false negative predictions
            precision: Precision score (0.0 to 1.0)
            recall: Recall score (0.0 to 1.0)
            f1_score: F1 score (0.0 to 1.0)
            accuracy: Accuracy score (0.0 to 1.0)
            processing_time: Time taken to process the patient (seconds)
            api_cost: Total API cost for processing the patient
            error: Error message if processing failed, None if successful
            skipped: Whether the patient was skipped during processing
            trial_evaluation_results: List of individual trial evaluation results
        """
        self.patient_id = patient_id
        self.ground_truth_count = ground_truth_count
        self.retrieved_count = retrieved_count
        self.predicted_eligible_count = predicted_eligible_count
        self.true_positives = true_positives
        self.false_positives = false_positives
        self.false_negatives = false_negatives
        self.precision = precision
        self.recall = recall
        self.f1_score = f1_score
        self.accuracy = accuracy
        self.processing_time = processing_time
        self.api_cost = api_cost
        self.error = error
        self.skipped = skipped
        self.trial_evaluation_results = trial_evaluation_results or []

    @classmethod
    def create_success_result(cls,
                            patient_id: str,
                            ground_truth_trials: List[str],
                            all_trial_ids: List[str],
                            predicted_eligible_trials: List[str],
                            metrics: Dict[str, Any],
                            processing_time: float,
                            total_api_cost: float,
                            trial_evaluation_results: List['TrialEvaluationResult']) -> 'PatientEvaluationResult':
        """
        Create a successful evaluation result.

        Args:
            patient_id: Patient identifier
            ground_truth_trials: List of ground truth relevant trial IDs
            all_trial_ids: List of all trial IDs retrieved for evaluation
            predicted_eligible_trials: List of trial IDs predicted as eligible
            metrics: Dictionary containing calculated metrics
            processing_time: Time taken to process the patient
            total_api_cost: Total API cost for processing
            trial_evaluation_results: List of individual trial evaluation results

        Returns:
            PatientEvaluationResult instance for successful evaluation
        """
        result = cls(
            patient_id=patient_id,
            ground_truth_count=len(ground_truth_trials),
            retrieved_count=len(all_trial_ids),
            predicted_eligible_count=len(predicted_eligible_trials),
            true_positives=metrics['true_positives'],
            false_positives=metrics['false_positives'],
            false_negatives=metrics['false_negatives'],
            precision=metrics['precision'],
            recall=metrics['recall'],
            f1_score=metrics['f1_score'],
            accuracy=metrics['accuracy'],
            processing_time=processing_time,
            api_cost=total_api_cost,
            error=None,
            skipped=False
        )
        result.trial_evaluation_results = trial_evaluation_results
        return result

    @classmethod
    def create_error_result(cls,
                          patient_id: str,
                          ground_truth_trials: List[str],
                          error_message: str) -> 'PatientEvaluationResult':
        """
        Create an error evaluation result when processing fails.

        Args:
            patient_id: Patient identifier
            ground_truth_trials: List of ground truth relevant trial IDs
            error_message: Error message describing what went wrong

        Returns:
            PatientEvaluationResult instance for failed evaluation
        """
        return cls(
            patient_id=patient_id,
            ground_truth_count=len(ground_truth_trials),
            retrieved_count=0,
            predicted_eligible_count=0,
            true_positives=0,
            false_positives=0,
            false_negatives=0,
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            accuracy=0.0,
            processing_time=0.0,
            api_cost=0.0,
            error=error_message,
            skipped=False
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the result to a dictionary representation.

        Returns:
            Dictionary representation of the evaluation result
        """
        result_dict: Dict[str, Any] = {
            'patient_id': self.patient_id,
            'ground_truth_count': self.ground_truth_count,
            'retrieved_count': self.retrieved_count,
            'predicted_eligible_count': self.predicted_eligible_count,
            'true_positives': self.true_positives,
            'false_positives': self.false_positives,
            'false_negatives': self.false_negatives,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'accuracy': self.accuracy,
            'processing_time': self.processing_time,
            'api_cost': self.api_cost,
            'error': self.error,
            'skipped': self.skipped
        }

        # Add trial evaluation results if available
        if hasattr(self, 'trial_evaluation_results') and self.trial_evaluation_results:
            result_dict['trial_evaluation_results'] = [trial.to_dict() for trial in self.trial_evaluation_results]

        return result_dict

    def is_successful(self) -> bool:
        """
        Check if the evaluation was successful.

        Returns:
            True if no error occurred and patient wasn't skipped
        """
        return self.error is None and not self.skipped

    def __str__(self) -> str:
        """String representation of the evaluation result."""
        if self.is_successful():
            return (f"PatientEvaluationResult(patient_id='{self.patient_id}', "
                   f"precision={self.precision:.3f}, recall={self.recall:.3f}, "
                   f"f1={self.f1_score:.3f}, time={self.processing_time:.2f}s)")
        else:
            return f"PatientEvaluationResult(patient_id='{self.patient_id}', error='{self.error}')"

    def __repr__(self) -> str:
        """Detailed string representation of the evaluation result."""
        return (f"PatientEvaluationResult(patient_id='{self.patient_id}', "
               f"ground_truth_count={self.ground_truth_count}, "
               f"retrieved_count={self.retrieved_count}, "
               f"predicted_eligible_count={self.predicted_eligible_count}, "
               f"true_positives={self.true_positives}, "
               f"false_positives={self.false_positives}, "
               f"false_negatives={self.false_negatives}, "
               f"precision={self.precision}, recall={self.recall}, "
               f"f1_score={self.f1_score}, accuracy={self.accuracy}, "
               f"processing_time={self.processing_time}, api_cost={self.api_cost}, "
               f"error={self.error}, skipped={self.skipped})")


class FilteringBenchmark:
    """
    Benchmark class for evaluating clinical trial filtering performance.

    This class implements an optimized caching mechanism for patient conditions extraction.
    The extract_conditions_from_content method is called only once per patient, with results
    stored in memory for reuse across multiple trial evaluations. This significantly reduces
    API calls and improves performance when evaluating multiple trials for the same patient.

    Key optimizations:
    - Conditions cache: Patient conditions are extracted once and cached
    - Cache hit rate tracking: Monitor cache efficiency
    - Pre-extraction support: Option to warm up cache before benchmark
    """

    def __init__(self, dataset_path: str, api_key: str, cache_size: int = 100000, strict_cache_mode: bool = False, enable_validation: bool = False):
        """
        Initialize the benchmark.

        Args:
            dataset_path: Path to TREC 2021 dataset directory
            api_key: OpenAI API key for GPT filtering (required)
            cache_size: Size of GPT response cache
            strict_cache_mode: If True, throws exception when cache is not hit (assumes all cache should hit)
            enable_validation: If True, enables cache file validation after writing
        """
        if not api_key:
            logger.error("API key is required for disease extraction")
            sys.exit(1)

        self.dataset_path = Path(dataset_path)
        self.api_key = api_key
        self.cache_size = cache_size
        self.strict_cache_mode = strict_cache_mode
        self.enable_validation = enable_validation

        # Initialize GPT client
        self.gpt_client = GPTClient(api_key=self.api_key, cache_size=self.cache_size, strict_cache_mode=self.strict_cache_mode, enable_validation=self.enable_validation)

        # Initialize trials attribute
        self.trials: Dict[str, ClinicalTrial] = {}

        # Initialize patient-trial mapping for max-trials allocation
        self.patient_trial_mapping: Dict[str, set[str]] = {}

        # Initialize conditions cache to avoid repeated API calls
        self.conditions_cache: Dict[str, List[str]] = {}

        # Load dataset
        self._load_dataset()

    def _load_dataset(self):
        """Load TREC 2021 dataset components."""
        logger.info("Loading TREC 2021 dataset...")

        # Load patients (from queries.jsonl file which contains patient medical records)
        patients_file = self.dataset_path / "queries.jsonl"
        self.patients: List[Patient] = []
        with open(patients_file, 'r') as f:
            for line in f:
                patient_data = json.loads(line)
                self.patients.append(Patient.from_dict(patient_data))

        # Log the number of patients and the first 10 patients
        logger.info(f"Loaded {len(self.patients)} patients")
        logger.info("First 10 patients:")
        for i, patient in enumerate(self.patients[:10], 1):
            logger.info(f"  {i}. {patient.patient_id}: {patient.get_text_summary(100)}")

        # Load relevance judgments
        qrels_file = self.dataset_path / "qrels" / "test.tsv"
        self.relevance_judgments: List[RelevanceJudgment] = []
        with open(qrels_file, 'r') as f:
            for line in f:
                judgment = RelevanceJudgment.from_tsv_line(line)
                if judgment is not None:
                    self.relevance_judgments.append(judgment)

        # Validate that the set of patient_ids in relevance_judgments is the same as the set of patient_ids in patients
        relevance_patient_ids = set(judgment.patient_id for judgment in self.relevance_judgments)
        patient_ids = set(patient.patient_id for patient in self.patients)
        if relevance_patient_ids != patient_ids:
            raise ValueError("The set of patient_ids in relevance_judgments is not the same as the set of patient_ids in patients")

        # Log the number of relevance judgments and the first 10 relevance judgments
        logger.info(f"Loaded {len(self.relevance_judgments)} relevance judgments")
        logger.info("First 10 relevance judgments:")
        for i, judgment in enumerate(self.relevance_judgments[:10], 1):
            logger.info(f"  {i}. {judgment}")

        # Extract unique trial IDs from relevance judgments
        trial_ids = set(judgment.trial_id for judgment in self.relevance_judgments)
        logger.info(f"Found {len(trial_ids)} unique trial IDs in relevance judgments")

        # Check if trial data file exists, if not download trials
        trials_file = self.dataset_path / "retrieved_trials.json"
        if not trials_file.exists():
            logger.info("Trial data file not found. Downloading trials...")
            try:
                self.download_trials(trial_ids, delay=0.1, timeout=30, save_individual=False, individual_format="json")  # Use default delay and timeout for initial download
                # Verify the file was created
                if not trials_file.exists():
                    raise RuntimeError("Trial data file was not created after download")
            except Exception as e:
                logger.error(f"Failed to download trials: {e}")
                raise RuntimeError(f"Failed to download trials: {e}")
        else:
            logger.info(f"Trial data file found: {trials_file}")

        # Load trial data
        self.trials = self._load_trials(trials_file)
        if not self.trials:
            raise RuntimeError("No trials were loaded from the trial data file")
        logger.info(f"Loaded {len(self.trials)} trials")

        # Validate trial coverage
        missing_trials = self.get_missing_trials()
        if missing_trials:
            logger.warning(f"{len(missing_trials)} trials are missing from the dataset. Continuing with available trials.")
            logger.warning(f"Missing trials: {missing_trials}")
        else:
            logger.info("All required trials are available")

        # Log coverage statistics
        coverage_stats = self.get_trial_coverage_stats()
        logger.info(f"Trial coverage: {coverage_stats['coverage_percentage']:.1f}% ({coverage_stats['total_available']}/{coverage_stats['total_required']})")

        logger.info(f"Loaded {len(self.patients)} patients")
        logger.info(f"Loaded {len(self.relevance_judgments)} relevance judgments")
        logger.info(f"Loaded {len(self.trials)} trials")

    def download_trials(self, trial_ids: set[str], delay: float = 0.1, timeout: int = 30, save_individual: bool = False, individual_format: str = "json"):
        """Download trials for the given trial IDs."""
        import requests
        import json
        import time
        import warnings
        import urllib3

        # Suppress SSL warnings to clean up output
        warnings.filterwarnings('ignore', message='Unverified HTTPS request')
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        # ClinicalTrials.gov API endpoint
        api_base_url = "https://clinicaltrials.gov/api/v2/studies"

        trials_data: List[ClinicalTrial] = []
        total_trials = len(trial_ids)

        # Create progress bar for trial downloading
        with tqdm(total=total_trials, desc="Downloading trials", unit="trial") as pbar:
            for trial_id in trial_ids:
                try:
                    # Update progress bar description with current trial ID
                    pbar.set_description(f"Downloading {trial_id}")

                    # API parameters
                    params = {
                        "format": "json",
                        "fields": "ProtocolSection",
                        "markupFormat": "markdown"
                    }

                    # Make API request with SSL verification disabled to handle certificate issues
                    url = f"{api_base_url}/{trial_id}"
                    try:
                        response = requests.get(url, params=params, timeout=timeout, verify=False)
                        logger.debug(f"Response for {trial_id}: {response.status_code}")

                        if response.status_code == 200:
                            data = response.json()
                            logger.debug(f"Data for {trial_id}: {data}")

                            if "protocolSection" in data:
                                # Extract trial data using the same structure as the spider
                                trial_data = self._extract_trial_data(data["protocolSection"])
                                trials_data.append(trial_data)

                                # Save individual trial file if requested
                                if save_individual:
                                    individual_file = self.dataset_path / f"trial_{trial_id}.json"
                                    with open(individual_file, 'w') as f:
                                        json.dump(trial_data, f, indent=2)
                            else:
                                logger.warning(f"No protocol section found for {trial_id}")

                        else:
                            logger.warning(f"Failed to download {trial_id}, status: {response.status_code}")

                    except requests.exceptions.SSLError as ssl_error:
                        logger.warning(f"SSL error for {trial_id}, retrying without verification: {ssl_error}")
                        # Retry without SSL verification
                        response = requests.get(url, params=params, timeout=timeout, verify=False)
                        if response.status_code == 200:
                            data = response.json()
                            if "protocolSection" in data:
                                trial_data = self._extract_trial_data(data["protocolSection"])
                                trials_data.append(trial_data)

                                # Save individual trial file if requested
                                if save_individual:
                                    individual_file = self.dataset_path / f"trial_{trial_id}.json"
                                    with open(individual_file, 'w') as f:
                                        json.dump(trial_data, f, indent=2)
                            else:
                                logger.warning(f"No protocol section found for {trial_id}")
                        else:
                            logger.warning(f"Failed to download {trial_id} even without SSL verification, status: {response.status_code}")

                    # Rate limiting - be respectful to the API
                    time.sleep(delay)  # Delay between requests

                except Exception as e:
                    logger.error(f"Error downloading {trial_id}: {e}")
                finally:
                    # Update progress bar regardless of success/failure
                    pbar.update(1)

        # Save all trials to file
        if trials_data:
            output_path = self.dataset_path / "retrieved_trials.json"

            # Load existing trials if file exists to avoid overwriting
            existing_trials: List[Dict[str, Any]] = []
            if output_path.exists():
                try:
                    with open(output_path, 'r') as f:
                        existing_trials = json.load(f)
                    logger.info(f"Loaded {len(existing_trials)} existing trials")
                except Exception as e:
                    logger.warning(f"Could not load existing trials: {e}")
                    existing_trials = []

            # Merge existing and new trials, avoiding duplicates
            existing_trial_ids = {trial.get('identification', {}).get('nct_id') for trial in existing_trials if trial.get('identification', {}).get('nct_id')}
            new_trials_only = [trial.to_dict() for trial in trials_data if trial.identification.nct_id not in existing_trial_ids]

            all_trials: List[Dict[str, Any]] = existing_trials + new_trials_only
            logger.info(f"Saving {len(all_trials)} total trials (existing: {len(existing_trials)}, new unique: {len(new_trials_only)})")

            with open(output_path, 'w') as f:
                json.dump(all_trials, f, indent=2)
        else:
            raise RuntimeError("No trials were successfully downloaded")

    def _extract_trial_data(self, protocol: Dict[str, Any]) -> ClinicalTrial:
        """Extract trial data from protocol section and return ClinicalTrial object."""
        def safe_get(d: Any, *keys: str, default: Any = None) -> Any:
            """Safely get nested dictionary values"""
            for key in keys:
                if not isinstance(d, dict):
                    return default
                d = d.get(str(key), default)  # type: ignore
                if d is None:
                    return default
            return d

        trial_dict = {
            "identification": {
                "nct_id": safe_get(protocol, "identificationModule", "nctId"),
                "url": (
                    f"https://clinicaltrials.gov/study/{safe_get(protocol, 'identificationModule', 'nctId')}"
                    if safe_get(protocol, "identificationModule", "nctId")
                    else None
                ),
                "brief_title": safe_get(protocol, "identificationModule", "briefTitle"),
                "official_title": safe_get(protocol, "identificationModule", "officialTitle"),
                "acronym": safe_get(protocol, "identificationModule", "acronym"),
                "org_study_id": safe_get(protocol, "identificationModule", "orgStudyIdInfo", "id"),
            },
            "status": {
                "overall_status": safe_get(protocol, "statusModule", "overallStatus"),
                "start_date": safe_get(protocol, "statusModule", "startDateStruct", "date"),
                "completion_date": safe_get(protocol, "statusModule", "completionDateStruct", "date"),
                "primary_completion_date": safe_get(protocol, "statusModule", "primaryCompletionDateStruct", "date"),
            },
            "description": {
                "brief_summary": safe_get(protocol, "descriptionModule", "briefSummary"),
                "detailed_description": safe_get(protocol, "descriptionModule", "detailedDescription"),
                "conditions": safe_get(protocol, "descriptionModule", "conditions"),
                "keywords": safe_get(protocol, "descriptionModule", "keywords"),
            },
            "design": {
                "study_type": safe_get(protocol, "designModule", "studyType"),
                "phases": safe_get(protocol, "designModule", "phases", default=[]),
                "enrollment": safe_get(protocol, "designModule", "enrollmentInfo", "count"),
                "arms": [
                    {
                        "name": safe_get(arm, "label"),
                        "type": safe_get(arm, "type"),
                        "description": safe_get(arm, "description"),
                        "interventions": safe_get(arm, "interventionNames", default=[]),
                    }
                    for arm in safe_get(protocol, "armsInterventionsModule", "armGroups", default=[])
                ],
            },
            "eligibility": {
                "criteria": safe_get(protocol, "eligibilityModule", "eligibilityCriteria"),
                "gender": safe_get(protocol, "eligibilityModule", "sex"),
                "minimum_age": safe_get(protocol, "eligibilityModule", "minimumAge"),
                "maximum_age": safe_get(protocol, "eligibilityModule", "maximumAge"),
                "healthy_volunteers": safe_get(protocol, "eligibilityModule", "healthyVolunteers"),
            },
            "contacts_locations": {
                "locations": [
                    {
                        "facility": safe_get(loc, "facility", "name") or safe_get(loc, "name"),
                        "city": safe_get(loc, "facility", "city") or safe_get(loc, "city"),
                        "state": safe_get(loc, "facility", "state") or safe_get(loc, "city"),
                        "country": safe_get(loc, "facility", "country") or safe_get(loc, "city"),
                        "status": safe_get(loc, "status"),
                    }
                    for loc in safe_get(protocol, "contactsLocationsModule", "locations", default=[])
                ],
            },
            "sponsor": {
                "lead_sponsor": safe_get(protocol, "sponsorCollaboratorsModule", "leadSponsor", "name"),
                "collaborators": [
                    safe_get(collab, "name", default="")
                    for collab in safe_get(protocol, "sponsorCollaboratorsModule", "collaborators", default=[])
                ],
            },
        }

        return ClinicalTrial(trial_dict)

    def _load_trials(self, trials_file: Path) -> Dict[str, ClinicalTrial]:
        """Load trial data from file."""
        try:
            with open(trials_file, 'r') as f:
                trials_data: List[Dict[str, Any]] = json.load(f)

            # Convert to dictionary with NCT ID as key for faster lookup
            trials_dict: Dict[str, ClinicalTrial] = {}
            for trial in trials_data:
                nct_id = trial.get('identification', {}).get('nct_id')
                if nct_id:
                    trials_dict[nct_id] = ClinicalTrial(trial)

            logger.info(f"Successfully loaded {len(trials_dict)} trials")
            return trials_dict

        except Exception as e:
            logger.error(f"Error loading trials: {e}")
            return {}

    def get_trial_data(self, trial_id: str) -> Optional[ClinicalTrial]:
        """
        Get trial data by NCT ID.

        Args:
            trial_id: NCT ID of the trial

        Returns:
            Trial data object or None if not found
        """
        return self.trials.get(trial_id)

    def get_missing_trials(self) -> List[str]:
        """
        Get list of trial IDs that are referenced in relevance judgments but not available in trials data.

        Returns:
            List of missing trial IDs
        """
        available_trials = set(self.trials.keys())
        required_trials = set(judgment.trial_id for judgment in self.relevance_judgments)
        missing_trials = required_trials - available_trials
        return list(missing_trials)

    def get_trial_coverage_stats(self) -> Dict[str, Any]:
        """
        Get statistics about trial coverage.

        Returns:
            Dictionary with coverage statistics
        """
        available_trials: set[str] = set(self.trials.keys())
        required_trials: set[str] = set(judgment.trial_id for judgment in self.relevance_judgments)
        missing_trials: set[str] = required_trials - available_trials
        additional_trials: set[str] = available_trials - required_trials
        additional_trials_count: int = len(additional_trials)

        total_required: int = len(required_trials)
        total_available: int = len(available_trials)
        total_missing: int = len(missing_trials)
        coverage_percentage: float = ((total_available - additional_trials_count) / total_required * 100) if total_required > 0 else 0.0

        return {
            'total_required': total_required,
            'total_available': total_available,
            'total_missing': total_missing,
            'coverage_percentage': coverage_percentage,
            'missing_trials': list(missing_trials)
        }

    def print_coverage_summary(self, verbose: bool = False):
        """Print a summary of trial coverage status."""
        coverage_stats = self.get_trial_coverage_stats()

        logger.info("\n" + "="*60)
        logger.info("TRIAL COVERAGE SUMMARY")
        logger.info("="*60)
        logger.info(f"Total trials required: {coverage_stats['total_required']}")
        logger.info(f"Total trials available: {coverage_stats['total_available']}")
        logger.info(f"Total trials missing: {coverage_stats['total_missing']}")
        logger.info(f"Coverage: {coverage_stats['coverage_percentage']:.2f}%")

        if coverage_stats['missing_trials']:
            if verbose:
                logger.info(f"\nMissing trials ({len(coverage_stats['missing_trials'])}):")
                for i, trial_id in enumerate(coverage_stats['missing_trials'], 1):
                    logger.info(f"  {i:2d}. {trial_id}")
            else:
                logger.info(f"\nMissing trials: {', '.join(coverage_stats['missing_trials'])}")

            logger.info(f"To improve coverage, you can:")
            logger.info(f"  1. Check if trials are available on ClinicalTrials.gov")
            logger.info(f"  2. Verify network connectivity and API access")
            logger.info(f"  3. Manually download missing trials if needed")

        logger.info("="*60 + "\n")

    def validate_trial_coverage(self) -> bool:
        """
        Validate that all trials referenced in relevance judgments are available in trials data.

        Returns:
            True if all trials are available, False otherwise
        """
        missing_trials = self.get_missing_trials()
        if missing_trials:
            logger.warning(f"{len(missing_trials)} trials are missing: {missing_trials[:10]}{'...' if len(missing_trials) > 10 else ''}")
            return False
        else:
            logger.info("All required trials are available")
            return True

    def get_ground_truth_trials(self, patient_id: str) -> List[str]:
        """
        Get ground truth relevant trials for a patient.

        Args:
            patient_id: Patient identifier

        Returns:
            List of NCT IDs of relevant trials (only those available in the dataset)
        """
        relevant_trials: List[str] = []
        available_trials = set(self.trials.keys())

        for judgment in self.relevance_judgments:
            if judgment.patient_id == patient_id and judgment.is_relevant():
                # Only include trials that are actually available in the dataset
                if judgment.trial_id in available_trials:
                    relevant_trials.append(judgment.trial_id)
                else:
                    logger.debug(f"Trial {judgment.trial_id} is not available in dataset, skipping")

        if not relevant_trials:
            logger.warning(f"Patient ID '{patient_id}' not found in relevance judgments or has no relevant trials available in dataset.")

        return relevant_trials

    def calculate_metrics(self, predicted_eligible: List[str], ground_truth: List[str]) -> Dict[str, float]:
        """
        Calculate precision, recall, F1-score, and accuracy.

        Args:
            predicted_eligible: List of predicted eligible trial IDs
            ground_truth: List of ground truth relevant trial IDs

        Returns:
            Dictionary with metrics
        """
        predicted_set = set(predicted_eligible)
        ground_truth_set = set(ground_truth)

        true_positives = len(predicted_set & ground_truth_set)
        false_positives = len(predicted_set - ground_truth_set)
        false_negatives = len(ground_truth_set - predicted_set)

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = true_positives / len(ground_truth_set) if ground_truth_set else 0.0

        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': accuracy,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }

    def evaluate_filtering_performance(self, patient: Patient, gpt_filter: GPTTrialFilter, title_only: bool = False) -> PatientEvaluationResult:
        """
        Evaluate filtering performance for a single patient.

        Args:
            patient: Patient object
            gpt_filter: GPTTrialFilter instance for trial filtering
            title_only: Whether to use title-only evaluation instead of full trial evaluation

        Returns:
            PatientEvaluationResult with evaluation metrics
        """
        patient_id = patient.patient_id

        # Extract disease name if GPT client is available
        disease_name: str = "Unknown"
        if self.gpt_client:
            disease_name, _ = extract_disease_from_record(patient.medical_record, self.gpt_client)
        else:
            raise ValueError("GPT client not initialized")

        logger.info(f"Evaluating patient: {patient_id} text: {patient.get_text_summary(100)} disease: {disease_name}")

        # Get ground truth relevant trials
        ground_truth_trials = self.get_ground_truth_trials(patient_id)
        logger.info(f"Ground truth relevant trials: {len(ground_truth_trials)}")

        try:
            # Call title check and regard trials that pass the title check as eligible
            start_time = time.time()

            # Extract conditions from patient record
            conditions = self._get_cached_conditions(patient_id, patient.medical_record)
            logger.info(f"Extracted conditions:")
            for condition in conditions:
                logger.info(f"  - {condition}")

            # Get trials for this specific patient (either all trials or allocated trials if max-trials is specified)
            if hasattr(self, 'patient_trial_mapping') and self.patient_trial_mapping:
                # Use allocated trials for this patient
                all_trial_ids = list(self.patient_trial_mapping.get(patient_id, set()))
                logger.info(f"Evaluating {len(all_trial_ids)} allocated trials for patient {patient_id}")
            else:
                # Use all available trials (no max-trials limit)
                all_trial_ids = list(self.trials.keys())
                logger.info(f"Evaluating {len(all_trial_ids)} trials for patient {patient_id}")

            # Check if patient has any trials to evaluate
            if not all_trial_ids:
                logger.warning(f"Patient {patient_id} has no trials to evaluate")
                return PatientEvaluationResult.create_error_result(
                    patient_id=patient_id,
                    ground_truth_trials=ground_truth_trials,
                    error_message="No trials allocated to this patient"
                )

            # Evaluate each trial using title check
            predicted_eligible_trials: List[str] = []
            total_api_cost = 0.0
            trial_evaluation_results: List[TrialEvaluationResult] = []

            evaluation_desc = f"Evaluating trials for {patient_id} ({'title-only' if title_only else 'full evaluation'})"
            for trial_id in tqdm(all_trial_ids, desc=evaluation_desc):
                try:
                    trial: ClinicalTrial = self.trials[trial_id]

                    # Use GPTTrialFilter to evaluate the trial based on title_only parameter
                    if title_only:
                        # Title-only evaluation (faster but less accurate)
                        suitability_probability: float
                        reason: str
                        cost: float
                        suitability_probability, reason, cost = gpt_filter.evaluate_title(
                            trial, conditions, refresh_cache=False
                        )
                        logger.info(f"Trial {trial_id} title suitability probability: {suitability_probability}, reason: {reason}, cost: {cost}")
                        is_eligible: bool = suitability_probability > 0.0
                    else:
                        # Full trial evaluation (title + inclusion criteria)
                        is_eligible: bool
                        cost: float
                        failure_reason: Optional[TrialFailureReason]
                        is_eligible, cost, failure_reason = gpt_filter.evaluate_trial(
                            trial, conditions, refresh_cache=False
                        )
                        if is_eligible:
                            suitability_probability: float = 1.0
                            reason: str = "Trial passed full evaluation (title + inclusion criteria)"
                        else:
                            if failure_reason is None:
                                raise RuntimeError(f"Trial {trial_id} failed full evaluation but no failure reason was recorded")
                            suitability_probability: float = 0.0
                            if not failure_reason:
                                raise RuntimeError(f"Trial {trial_id} failed full evaluation but no failure reason was recorded")
                            failure_details: str = failure_reason.failure_details if failure_reason.failure_details else 'Empty'
                            failed_condition: str = failure_reason.failed_condition if failure_reason.failed_condition else 'Empty'
                            failed_criterion: str = failure_reason.failed_criterion if failure_reason.failed_criterion else 'Empty'
                            reason: str = f"Trial failed full evaluation: {failure_reason.message}\n failed_condition: {failed_condition}\n failed_criterion: {failed_criterion}\n failure_details: {failure_details} "
                        logger.info(f"Trial {trial_id} full evaluation result: eligible={is_eligible}, reason: {reason}, cost: {cost}")

                    total_api_cost += cost

                    # Determine ground truth relevance
                    ground_truth_relevant = trial_id in ground_truth_trials

                    # Get original relevance score from relevance judgments
                    original_relevance_score = None
                    for judgment in self.relevance_judgments:
                        if judgment.patient_id == patient_id and judgment.trial_id == trial_id:
                            original_relevance_score = judgment.relevance_score
                            break

                    trial_result = TrialEvaluationResult(
                        trial_id=trial_id,
                        trial_title=trial.identification.brief_title or trial.identification.official_title or trial.identification.acronym or trial_id,
                        predicted_eligible=is_eligible,
                        ground_truth_relevant=ground_truth_relevant,
                        suitability_probability=suitability_probability,
                        reason=reason,
                        api_cost=cost,
                        original_relevance_score=original_relevance_score
                    )
                    trial_evaluation_results.append(trial_result)

                    if is_eligible:
                        predicted_eligible_trials.append(trial_id)
                        evaluation_method = "title check" if title_only else "full evaluation"
                        logger.debug(f"Trial {trial_id} passed {evaluation_method}")
                    else:
                        evaluation_method = "title check" if title_only else "full evaluation"
                        logger.debug(f"Trial {trial_id} failed {evaluation_method}: {reason}")

                except Exception as e:
                    logger.warning(f"Error evaluating trial {trial_id}: {str(e)}")
                    raise e
                    # Get trial title safely, fallback to trial_id if trial object is not available
                    trial_title = trial_id
                    try:
                        if 'trial' in locals():
                            trial_obj = locals().get('trial')
                            if trial_obj:
                                trial_title = trial_obj.identification.brief_title or trial_obj.identification.official_title or trial_obj.identification.acronym or trial_id
                    except:
                        pass

                    # Get original relevance score from relevance judgments
                    original_relevance_score = None
                    for judgment in self.relevance_judgments:
                        if judgment.patient_id == patient_id and judgment.trial_id == trial_id:
                            original_relevance_score = judgment.relevance_score
                            break

                    trial_result = TrialEvaluationResult(
                        trial_id=trial_id,
                        trial_title=trial_title,
                        predicted_eligible=False,
                        ground_truth_relevant=False,
                        suitability_probability=0.0,
                        reason=f"Evaluation failed: {e}",
                        api_cost=0.0,
                        original_relevance_score=original_relevance_score
                    )
                    trial_evaluation_results.append(trial_result)
                    continue

            processing_time = time.time() - start_time

            # Calculate performance metrics
            metrics = self.calculate_metrics(predicted_eligible_trials, ground_truth_trials)

            evaluation_method = "title check" if title_only else "full evaluation (title + inclusion criteria)"
            logger.info(f"Patient {patient_id}: {len(predicted_eligible_trials)} trials passed {evaluation_method} out of {len(all_trial_ids)} total trials")
            logger.info(f"Performance metrics: Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, F1={metrics['f1_score']:.3f}")

            return PatientEvaluationResult.create_success_result(
                patient_id=patient_id,
                ground_truth_trials=ground_truth_trials,
                all_trial_ids=all_trial_ids,
                predicted_eligible_trials=predicted_eligible_trials,
                metrics=metrics,
                processing_time=processing_time,
                total_api_cost=total_api_cost,
                trial_evaluation_results=trial_evaluation_results
            )

        except Exception as e:
            logger.error(f"Error processing patient {patient_id}: {str(e)}")
            raise e

    def run_benchmark(self, gpt_filter: GPTTrialFilter, max_patients: Optional[int] = None, output_path: Optional[str] = None, title_only: bool = False) -> Dict[str, Any]:
        """
        Run the complete benchmark.

        Args:
            gpt_filter: GPTTrialFilter instance for trial filtering
            max_patients: Maximum number of patients to process (for testing)
            output_path: Path to the output file for error case export
            title_only: Whether to use title-only evaluation instead of full trial evaluation
        Returns:
            Dictionary with overall benchmark results
        """
        evaluation_method = "title-only evaluation (evaluate_title)" if title_only else "full trial evaluation (evaluate_trial)"
        logger.info(f"Starting filtering performance benchmark using {evaluation_method}...")

        patients_to_process: List[Patient] = self.patients[:max_patients] if max_patients else self.patients

        # Log cache strategy information
        logger.info("=" * 60)
        logger.info("CACHING STRATEGY")
        logger.info("=" * 60)
        logger.info(f"Conditions will be extracted once per patient and cached for reuse")
        logger.info(f"Total patients to process: {len(patients_to_process)}")
        logger.info(f"Current cache size: {len(self.conditions_cache)}")
        logger.info(f"Expected API calls saved: {len(patients_to_process) - len(self.conditions_cache)}")
        logger.info(f"Cache efficiency: {len(self.conditions_cache) / len(patients_to_process) * 100:.1f}% pre-cached")
        logger.info("=" * 60)

        # Extract disease names and record their indices
        disease_data: List[tuple[str, int]] = []
        for i, patient in enumerate(patients_to_process):
            disease_name = extract_disease_from_record(patient.medical_record, self.gpt_client)[0]
            disease_data.append((disease_name, i))

        disease_names: List[str] = [data[0] for data in disease_data]
        logger.info(f"Disease names: {disease_names}")

        # Filter oncology diseases and keep their indices
        oncology_disease_data: List[tuple[str, int]] = [(disease_name, idx) for disease_name, idx in disease_data if is_oncology_disease(disease_name)]
        oncology_disease_names: List[str] = [data[0] for data in oncology_disease_data]
        oncology_indices: List[int] = [data[1] for data in oncology_disease_data]
        logger.info(f"Oncology disease names: {oncology_disease_names}")
        logger.info(f"Oncology disease indices: {oncology_indices}")

        # For each oncology disease name, log the number of trials under that disease in test.tsv
        for disease_name, idx in oncology_disease_data:
            # Get the patient directly by index
            patient = patients_to_process[idx]

            # Get trial IDs for this specific patient from relevance judgments
            disease_trial_ids: set[str] = set()
            for judgment in self.relevance_judgments:
                if judgment.patient_id == patient.patient_id:
                    disease_trial_ids.add(judgment.trial_id)

            # Count total trials for this disease (all trials associated with this patient)
            total_trial_count = len(disease_trial_ids)

            logger.info(f"Disease '{disease_name}' (patient {patient.patient_id}): {total_trial_count} trials")

        results: List[PatientEvaluationResult] = []
        total_processing_time = 0.0
        total_api_cost = 0.0

        # Create progress bar for patient processing
        progress_desc = f"Processing patients ({evaluation_method})"
        with tqdm(total=len(patients_to_process), desc=progress_desc, unit="patient") as pbar:
            for i, patient in enumerate(patients_to_process, 1):
                # Update progress bar description with current patient ID
                pbar.set_description(f"Processing {patient.patient_id}")

                conditions = self._get_cached_conditions(patient.patient_id, patient.medical_record)
                logger.info(f"Conditions: {conditions} for patient medical record: {patient.medical_record}")

                result: PatientEvaluationResult = self.evaluate_filtering_performance(patient, gpt_filter, title_only)
                results.append(result)

                total_processing_time += result.processing_time
                total_api_cost += result.api_cost

                # Update progress bar
                pbar.update(1)

                # Log progress every 10 patients
                if i % 10 == 0:
                    cache_stats = self.get_cache_statistics()
                    logger.info(f"Processed {i} patients. Total time: {total_processing_time:.2f}s, Total cost: ${total_api_cost:.4f}")
                    logger.info(f"Cache status: {cache_stats['cached_patients']}/{cache_stats['total_patients']} patients cached ({cache_stats['cache_hit_rate']:.1%})")

        # Calculate aggregate metrics
        successful_results: List[PatientEvaluationResult] = [r for r in results if r.error is None and not r.skipped]
        skipped_results: List[PatientEvaluationResult] = [r for r in results if r.skipped]
        failed_results: List[PatientEvaluationResult] = [r for r in results if r.error is not None and not r.skipped]
        for failed_result in failed_results:
            logger.warning(f"Failed result: {failed_result}")

        if successful_results:
            avg_precision = sum(r.precision for r in successful_results) / len(successful_results)
            avg_recall = sum(r.recall for r in successful_results) / len(successful_results)
            avg_f1 = sum(r.f1_score for r in successful_results) / len(successful_results)
            avg_accuracy = sum(r.accuracy for r in successful_results) / len(successful_results)
        else:
            avg_precision = avg_recall = avg_f1 = avg_accuracy = 0.0

        # Calculate trial-level metrics (across all trials regardless of patients)
        all_trial_results: List[TrialEvaluationResult] = []
        for result in successful_results:
            if hasattr(result, 'trial_evaluation_results') and result.trial_evaluation_results:
                all_trial_results.extend(result.trial_evaluation_results)

        if all_trial_results:
            # Calculate trial-level precision, recall, and F1
            trial_true_positives = sum(1 for trial in all_trial_results if trial.predicted_eligible and trial.ground_truth_relevant)
            trial_false_positives = sum(1 for trial in all_trial_results if trial.predicted_eligible and not trial.ground_truth_relevant)
            trial_false_negatives = sum(1 for trial in all_trial_results if not trial.predicted_eligible and trial.ground_truth_relevant)

            trial_precision = trial_true_positives / (trial_true_positives + trial_false_positives) if (trial_true_positives + trial_false_positives) > 0 else 0.0
            trial_recall = trial_true_positives / (trial_true_positives + trial_false_negatives) if (trial_true_positives + trial_false_negatives) > 0 else 0.0
            trial_f1 = 2 * (trial_precision * trial_recall) / (trial_precision + trial_recall) if (trial_precision + trial_recall) > 0 else 0.0
            trial_accuracy = (trial_true_positives + sum(1 for trial in all_trial_results if not trial.predicted_eligible and not trial.ground_truth_relevant)) / len(all_trial_results)
        else:
            trial_precision = trial_recall = trial_f1 = trial_accuracy = 0.0

        # Get trial coverage statistics
        coverage_stats = self.get_trial_coverage_stats()

        # Calculate MD5 hash of all trial IDs for consistency verification
        all_trial_ids = sorted(list(self.trials.keys()))
        trial_ids_concat = ''.join(all_trial_ids)
        trial_ids_md5 = hashlib.md5(trial_ids_concat.encode('utf-8')).hexdigest()

        logger.info("=" * 60)
        logger.info("TRIAL IDS CONSISTENCY VERIFICATION")
        logger.info("=" * 60)
        logger.info(f"Total trials in benchmark: {len(all_trial_ids)}")
        logger.info(f"Concatenated trial IDs MD5 hash: {trial_ids_md5}")
        logger.info(f"First 5 trial IDs: {all_trial_ids[:5]}")
        logger.info(f"Last 5 trial IDs: {all_trial_ids[-5:]}")
        logger.info("=" * 60)

        benchmark_results = {
            'timestamp': datetime.now().isoformat(),
            'dataset_path': str(self.dataset_path),
            'evaluation_method': 'title_only' if title_only else 'full_trial_evaluation',
            'trial_coverage': coverage_stats,
            'trial_ids_consistency': {
                'total_trials': len(all_trial_ids),
                'trial_ids_md5_hash': trial_ids_md5,
                'first_5_trial_ids': all_trial_ids[:5],
                'last_5_trial_ids': all_trial_ids[-5:]
            },
            'total_patients': len(patients_to_process),
            'successful_patients': len(successful_results),
            'skipped_patients': len(skipped_results),
            'failed_patients': len(failed_results),
            'total_processing_time': total_processing_time,
            'total_api_cost': total_api_cost,
            'average_processing_time_per_patient': total_processing_time / len(patients_to_process) if patients_to_process else 0.0,
            'average_api_cost_per_patient': total_api_cost / len(patients_to_process) if patients_to_process else 0.0,
            'conditions_cache_stats': self.get_cache_statistics(),
            'metrics': {
                'average_precision': avg_precision,
                'average_recall': avg_recall,
                'average_f1_score': avg_f1,
                'average_accuracy': avg_accuracy
            },
            'trial_level_metrics': {
                'trial_precision': trial_precision,
                'trial_recall': trial_recall,
                'trial_f1_score': trial_f1,
                'trial_accuracy': trial_accuracy,
                'total_trials_evaluated': len(all_trial_results)
            },
            'detailed_results': [r.to_dict() for r in results]
        }

        evaluation_method = "title-only evaluation (evaluate_title)" if title_only else "full trial evaluation (evaluate_trial)"
        logger.info("Benchmark completed!")
        logger.info(f"Evaluation method used: {evaluation_method}")
        logger.info(f"Total patients processed: {len(patients_to_process)}")
        logger.info(f"Successful patients: {len(successful_results)}")
        logger.info(f"Skipped patients: {len(skipped_results)}")
        logger.info(f"Failed patients: {len(failed_results)}")
        # Calculate effective available trials (excluding additional trials that aren't required)
        effective_available: set[str] = set(judgment.trial_id for judgment in self.relevance_judgments) & set(self.trials.keys())
        logger.info(f"Trial coverage: {coverage_stats['coverage_percentage']:.2f}% ({len(effective_available)}/{coverage_stats['total_required']})")
        logger.info(f"Total processing time: {total_processing_time:.2f}s")
        logger.info(f"Total API cost: ${total_api_cost:.4f}")
        logger.info(f"Average precision: {avg_precision:.4f}")
        logger.info(f"Average recall: {avg_recall:.4f}")
        logger.info(f"Average F1-score: {avg_f1:.4f}")

        # Log trial-level metrics
        logger.info("=" * 60)
        logger.info("TRIAL-LEVEL METRICS (across all trials regardless of patients)")
        logger.info("=" * 60)
        logger.info(f"Total trials evaluated: {len(all_trial_results)}")
        logger.info(f"Trial-level precision: {trial_precision:.4f}")
        logger.info(f"Trial-level recall: {trial_recall:.4f}")
        logger.info(f"Trial-level F1-score: {trial_f1:.4f}")
        logger.info(f"Trial-level accuracy: {trial_accuracy:.4f}")
        logger.info("=" * 60)

        # Log conditions cache statistics
        cache_stats = self.get_cache_statistics()
        logger.info("=" * 60)
        logger.info("CONDITIONS CACHE STATISTICS")
        logger.info("=" * 60)
        logger.info(f"Total patients: {cache_stats['total_patients']}")
        logger.info(f"Cached patients: {cache_stats['cached_patients']}")
        logger.info(f"Cache hit rate: {cache_stats['cache_hit_rate']:.2%}")
        logger.info(f"Cache size: {cache_stats['cache_size']}")

        # Calculate and log cache efficiency metrics
        if cache_stats['total_patients'] > 0:
            api_calls_saved = cache_stats['total_patients'] - cache_stats['cached_patients']
            efficiency_gain = (api_calls_saved / cache_stats['total_patients']) * 100 if cache_stats['total_patients'] > 0 else 0
            logger.info(f"API calls saved through caching: {api_calls_saved}")
            logger.info(f"Efficiency gain: {efficiency_gain:.1f}%")

            # Estimate cost savings (assuming each API call costs money)
            estimated_cost_per_call = 0.01  # Rough estimate, can be adjusted
            estimated_cost_saved = api_calls_saved * estimated_cost_per_call
            logger.info(f"Estimated cost savings: ${estimated_cost_saved:.4f}")
        logger.info("=" * 60)

        # Log trial IDs consistency verification
        all_trial_ids = sorted(list(self.trials.keys()))
        trial_ids_concat = ''.join(all_trial_ids)
        trial_ids_md5 = hashlib.md5(trial_ids_concat.encode('utf-8')).hexdigest()

        logger.info("=" * 60)
        logger.info("TRIAL IDS CONSISTENCY VERIFICATION")
        logger.info("=" * 60)
        logger.info(f"Total trials in benchmark: {len(all_trial_ids)}")
        logger.info(f"Concatenated trial IDs MD5 hash: {trial_ids_md5}")
        logger.info(f"First 5 trial IDs: {all_trial_ids[:5]}")
        logger.info(f"Last 5 trial IDs: {all_trial_ids[-5:]}")
        logger.info("=" * 60)

        # Log all error cases for analysis
        self.log_error_cases(successful_results)

        # Export error cases to a separate file for detailed analysis
        if output_path:
            error_cases_file = self.export_error_cases(successful_results, output_path)
            if error_cases_file:
                logger.info(f"Error cases exported to: {error_cases_file}")
        else:
            logger.info("No output path provided, skipping error case export")

        return benchmark_results

    def log_error_cases(self, successful_results: List[PatientEvaluationResult]) -> None:
        """
        Log all error cases (incorrectly predicted or failed to recall) for analysis.

        Args:
            successful_results: List of successful patient evaluation results
        """
        logger.info("=" * 80)
        logger.info("ERROR CASE ANALYSIS")
        logger.info("=" * 80)

        all_error_cases: List[tuple[PatientEvaluationResult, 'TrialEvaluationResult']] = []

        # Collect all error cases from successful results
        for result in successful_results:
            if hasattr(result, 'trial_evaluation_results') and result.trial_evaluation_results:
                for trial_result in result.trial_evaluation_results:
                    if trial_result.is_error_case:
                        all_error_cases.append((result, trial_result))

        if not all_error_cases:
            logger.info("No error cases found - all predictions were correct!")
            return

        logger.info(f"Found {len(all_error_cases)} error cases:")
        logger.info(f"- False positives: {len([case for _, case in all_error_cases if case.error_type == ErrorType.FALSE_POSITIVE])}")
        logger.info(f"- False negatives: {len([case for _, case in all_error_cases if case.error_type == ErrorType.FALSE_NEGATIVE])}")
        logger.info("")

        # Group error cases by type
        false_positive_cases = [case for _, case in all_error_cases if case.error_type == ErrorType.FALSE_POSITIVE]
        false_negative_cases = [case for _, case in all_error_cases if case.error_type == ErrorType.FALSE_NEGATIVE]

        # Log false positive cases
        if false_positive_cases:
            self._log_error_cases("FALSE POSITIVE CASES",
                                 [case for case in all_error_cases if case[1].error_type == ErrorType.FALSE_POSITIVE])

        # Log false negative cases
        if false_negative_cases:
            self._log_error_cases("FALSE NEGATIVE CASES",
                                 [case for case in all_error_cases if case[1].error_type == ErrorType.FALSE_NEGATIVE])

    def _log_error_cases(self, title: str, error_cases: List[tuple[PatientEvaluationResult, TrialEvaluationResult]]) -> None:
        """Helper method to log error cases with consistent formatting."""
        logger.info(title)
        logger.info("-" * 60)
        for i, (patient_result, trial_result) in enumerate(error_cases):
            logger.info(f"Case {i+1}:")
            logger.info(f"  Patient ID: {patient_result.patient_id}")
            logger.info(f"  Disease: {self._get_patient_disease(patient_result.patient_id)}")
            conditions = self._get_patient_extracted_conditions(patient_result.patient_id)
            if conditions:
                logger.info(f"  Extracted Conditions:")
                for condition in conditions:
                    logger.info(f"    - {condition}")
            else:
                logger.info(f"  Extracted Conditions: No conditions available")
            logger.info(f"  Medical Record: {self._get_patient_full_medical_record(patient_result.patient_id)}")
            logger.info(f"  Trial ID: {trial_result.trial_id}")
            logger.info(f"  Trial Title: {trial_result.trial_title}")
            logger.info(f"  Original Label: {trial_result.original_relevance_score if trial_result.original_relevance_score is not None else 'N/A'}")
            logger.info(f"  Error Type: {trial_result.error_type}")
            logger.info(f"  Suitability Probability: {trial_result.suitability_probability:.4f}")
            logger.info(f"  Reason: {trial_result.reason}")
            logger.info("")

    def _get_patient_disease(self, patient_id: str) -> str:
        """
        Get the disease name for a patient.

        Args:
            patient_id: Patient identifier

        Returns:
            Disease name or "Unknown" if not found
        """
        for patient in self.patients:
            if patient.patient_id == patient_id:
                try:
                    disease_name, _ = extract_disease_from_record(patient.medical_record, self.gpt_client)
                    return disease_name
                except:
                    return "Unknown"
        return "Unknown"

    def export_error_cases(self, successful_results: List[PatientEvaluationResult], output_path: str) -> Optional[str]:
        """
        Export error cases to a separate JSON file for detailed analysis.

        Args:
            successful_results: List of successful patient evaluation results
            output_path: Path to the main benchmark results file

        Returns:
            Path to the error cases file, or None if no errors found
        """
        all_error_cases: List[Dict[str, Any]] = []

        # Collect all error cases from successful results
        for result in successful_results:
            if hasattr(result, 'trial_evaluation_results') and result.trial_evaluation_results:
                for trial_result in result.trial_evaluation_results:
                    if trial_result.is_error_case:
                        # Get patient text summary and disease
                        patient_text_summary = self._get_patient_text_summary(result.patient_id)
                        disease_name = self._get_patient_disease(result.patient_id)

                        # Get trial criteria from the loaded trial data
                        trial_criteria = "No criteria available"
                        if trial_result.trial_id in self.trials:
                            trial_criteria = self.trials[trial_result.trial_id].eligibility.criteria

                        error_case = {
                            'patient_id': result.patient_id,
                            'text_summary': patient_text_summary,
                            'disease_name': disease_name,
                            'trial_id': trial_result.trial_id,
                            'trial_title': trial_result.trial_title,
                            'trial_criteria': trial_criteria,
                            'error_type': trial_result.error_type.value,
                            'suitability_probability': trial_result.suitability_probability,
                            'reason': trial_result.reason,
                            'ground_truth_relevant': trial_result.ground_truth_relevant,
                            'predicted_eligible': trial_result.predicted_eligible,
                            'original_relevance_score': trial_result.original_relevance_score if trial_result.original_relevance_score is not None else 'N/A'
                        }
                        all_error_cases.append(error_case)

        if not all_error_cases:
            return None

        # Create error cases file path
        output_dir = Path(output_path).parent
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        error_cases_file = output_dir / f"error_cases_{timestamp}.json"

        # Export to JSON
        with open(error_cases_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'total_error_cases': len(all_error_cases),
                'false_positives': len([case for case in all_error_cases if case['error_type'] == ErrorType.FALSE_POSITIVE.value]),
                'false_negatives': len([case for case in all_error_cases if case['error_type'] == ErrorType.FALSE_NEGATIVE.value]),
                'error_cases': all_error_cases
            }, f, indent=2)

        return str(error_cases_file)

    def _get_patient_text_summary(self, patient_id: str) -> str:
        """
        Get a text summary for a patient.

        Args:
            patient_id: Patient identifier

        Returns:
            Text summary or "No text available" if not found
        """
        for patient in self.patients:
            if patient.patient_id == patient_id:
                return patient.get_text_summary(200)  # Get first 200 characters
        return "No text available"

    def _get_patient_full_medical_record(self, patient_id: str) -> str:
        """
        Get the full medical record for a patient.

        Args:
            patient_id: Patient identifier

        Returns:
            Full medical record text or "No medical record available" if not found
        """
        for patient in self.patients:
            if patient.patient_id == patient_id:
                return patient.medical_record
        return "No medical record available"

    def _get_patient_extracted_conditions(self, patient_id: str) -> List[str]:
        """
        Get the extracted conditions for a patient.

        Args:
            patient_id: Patient identifier

        Returns:
            List of extracted conditions or empty list if not found
        """
        for patient in self.patients:
            if patient.patient_id == patient_id:
                try:
                    conditions = self._get_cached_conditions(patient_id, patient.medical_record)
                    return conditions
                except Exception as e:
                    logger.error(f"Error extracting conditions for patient {patient_id}: {str(e)}")
                    return []
        return []

    def _get_cached_conditions(self, patient_id: str, medical_record: str) -> List[str]:
        """
        Get conditions from cache or extract them if not cached.

        This method implements the core caching optimization. It checks if conditions
        for a patient have already been extracted and cached. If not, it calls
        extract_conditions_from_content and stores the result in the cache for
        future use. This ensures that the expensive API call is made only once
        per patient, regardless of how many trials are evaluated.

        Args:
            patient_id: Patient identifier
            medical_record: Patient's medical record text

        Returns:
            Extracted conditions as a list of strings

        Note:
            The first call for each patient will trigger an API call to extract
            conditions. Subsequent calls will use the cached result, improving
            performance and reducing API costs.
        """
        if patient_id not in self.conditions_cache:
            logger.info(f"Extracting conditions for patient {patient_id} (not in cache)")
            conditions = extract_conditions_from_content(medical_record, self.gpt_client, convert_time=False)
            self.conditions_cache[patient_id] = conditions
            logger.info(f"Cached conditions for patient {patient_id}: {conditions}")
        else:
            logger.debug(f"Using cached conditions for patient {patient_id} (cache hit)")

        return self.conditions_cache[patient_id]

    def get_cache_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the conditions cache.

        Returns:
            Dictionary with cache statistics
        """
        return {
            'total_patients': len(self.patients),
            'cached_patients': len(self.conditions_cache),
            'cache_hit_rate': len(self.conditions_cache) / len(self.patients) if self.patients else 0.0,
            'cache_size': len(self.conditions_cache)
        }

    def clear_conditions_cache(self) -> None:
        """
        Clear the conditions cache to free memory.
        """
        cache_size = len(self.conditions_cache)
        self.conditions_cache.clear()
        logger.info(f"Cleared conditions cache ({cache_size} entries)")

    def pre_extract_all_conditions(self) -> None:
        """
        Pre-extract conditions for all patients to warm up the cache.
        This can be useful for analysis or to ensure all conditions are available upfront.
        """
        logger.info("Pre-extracting conditions for all patients...")
        total_patients = len(self.patients)

        with tqdm(total=total_patients, desc="Pre-extracting conditions", unit="patient") as pbar:
            for patient in self.patients:
                if patient.patient_id not in self.conditions_cache:
                    self._get_cached_conditions(patient.patient_id, patient.medical_record)
                pbar.update(1)

        cache_stats = self.get_cache_statistics()
        logger.info(f"Pre-extraction complete. Cache status: {cache_stats['cached_patients']}/{cache_stats['total_patients']} patients cached ({cache_stats['cache_hit_rate']:.1%})")

    def export_conditions_cache(self, output_path: str) -> str:
        """
        Export the conditions cache to a JSON file for analysis.

        Args:
            output_path: Path to the output file

        Returns:
            Path to the exported cache file
        """
        cache_file = Path(output_path).parent / f"conditions_cache_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        cache_data = {
            'timestamp': datetime.now().isoformat(),
            'cache_statistics': self.get_cache_statistics(),
            'cached_conditions': self.conditions_cache
        }

        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)

        logger.info(f"Conditions cache exported to: {cache_file}")
        return str(cache_file)

    def show_cache_status(self) -> None:
        """
        Display current cache status and statistics.
        """
        cache_stats = self.get_cache_statistics()
        logger.info("=" * 40)
        logger.info("CURRENT CACHE STATUS")
        logger.info("=" * 40)
        logger.info(f"Total patients: {cache_stats['total_patients']}")
        logger.info(f"Cached patients: {cache_stats['cached_patients']}")
        logger.info(f"Cache hit rate: {cache_stats['cache_hit_rate']:.2%}")
        logger.info(f"Cache size: {cache_stats['cache_size']}")
        if cache_stats['total_patients'] > 0:
            api_calls_saved = cache_stats['total_patients'] - cache_stats['cached_patients']
            efficiency_gain = (api_calls_saved / cache_stats['total_patients']) * 100
            logger.info(f"API calls saved: {api_calls_saved}")
            logger.info(f"Efficiency gain: {efficiency_gain:.1f}%")
        logger.info("=" * 40)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark filtering performance on TREC 2021 dataset"
    )
    parser.add_argument(
        "--dataset-path",
        default="dataset/trec_2021",
        help="Path to TREC 2021 dataset directory"
    )
    parser.add_argument(
        "--output",
        default=f"results/benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        help="Output file for benchmark results"
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Output directory for benchmark results"
    )
    parser.add_argument(
        "--api-key",
        required=True,
        help="OpenAI API key (required for disease extraction, alternatively set OPENAI_API_KEY environment variable)"
    )
    parser.add_argument(
        "--cache-size",
        type=int,
        default=100000,
        help="Size of GPT response cache"
    )
    parser.add_argument(
        "--strict-cache-mode",
        action="store_true",
        help="Enable strict cache mode - throws exception when cache is not hit (assumes all cache should hit)"
    )
    parser.add_argument(
        "--enable-validation",
        action="store_true",
        help="Enable cache file validation after writing (validates that cache keys are found in saved files)"
    )
    parser.add_argument(
        "--max-patients",
        type=int,
        help="Maximum number of patients to process (for testing)"
    )
    parser.add_argument(
        "--patient-id",
        help="Run benchmark only on a specific patient ID"
    )
    parser.add_argument(
        "--save-coverage",
        action="store_true",
        help="Save coverage statistics to a separate file"
    )
    parser.add_argument(
        "--coverage-filename",
        help="Custom filename for coverage statistics (without extension)"
    )
    parser.add_argument(
        "--coverage-only",
        action="store_true",
        help="Only show trial coverage status without running benchmark"
    )
    parser.add_argument(
        "--show-help",
        action="store_true",
        help="Show additional help information about trial coverage"
    )
    parser.add_argument(
        "--verbose-coverage",
        action="store_true",
        help="Show detailed information about missing trials"
    )
    parser.add_argument(
        "--export-conditions-cache",
        action="store_true",
        help="Export the conditions cache to a separate file for analysis"
    )
    parser.add_argument(
        "--show-cache-status",
        action="store_true",
        help="Show detailed cache status at the end of the benchmark"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of trials to download in a single batch"
    )
    parser.add_argument(
        "--download-delay",
        type=float,
        default=0.1,
        help="Delay in seconds between individual trial downloads"
    )
    parser.add_argument(
        "--download-timeout",
        type=int,
        default=30,
        help="Timeout in seconds for individual trial downloads"
    )
    parser.add_argument(
        "--save-individual-trials",
        action="store_true",
        help="Save individual trial files in addition to the combined file"
    )
    parser.add_argument(
        "--individual-trial-format",
        choices=["json", "jsonl"],
        default="json",
        help="Format for individual trial files (json or jsonl)"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        help="Maximum total number of trials to use for the benchmark (for faster testing). Trials are allocated proportionally across patients based on their original trial counts. Trials with original label 1 are excluded to focus sampling on label 0 and 2 trials. Uses deterministic sampling with fixed seed for consistent results across executions and devices."
    )
    parser.add_argument(
        "--cancer-only",
        action="store_true",
        help="Run benchmark only on cancer patients. When combined with --max-trials, allocates trials proportionally across cancer patients based on their original trial counts. Trials with original label 1 are excluded from sampling."
    )
    parser.add_argument(
        "--title-only",
        action="store_true",
        help="Use title-only evaluation (evaluate_title) instead of full trial evaluation (evaluate_trial). Title-only is faster but less accurate."
    )

    args = parser.parse_args()

    # Show additional help if requested
    if args.show_help:
        logger.info("\n" + "="*80)
        logger.info("TRIAL COVERAGE HELP")
        logger.info("="*80)
        logger.info("This script evaluates clinical trial filtering performance on the TREC 2021 dataset.")
        logger.info("It requires downloading clinical trial data from ClinicalTrials.gov.")
        logger.info("\nTRIAL COVERAGE:")
        logger.info("- The script checks if all trials referenced in the dataset are available")
        logger.info("- Missing trials can occur due to network issues, API limits, or trial unavailability")
        logger.info("- Use --coverage-only to check status without running the full benchmark")
        logger.info("\nTRIAL SAMPLING:")
        logger.info("- Use --max-trials to limit the total number of trials for faster testing")
        logger.info("- Trials are allocated proportionally across patients based on their original trial counts")
        logger.info("- Trials with original label 1 are excluded to focus sampling on label 0 and 2 trials")
        logger.info("- Sampling is deterministic (seed=42) for consistent results across executions and devices")
        logger.info("- This ensures reproducible benchmark results regardless of system differences")
        logger.info("\nCANCER PATIENT FILTERING:")
        logger.info("- Use --cancer-only to run benchmark only on cancer patients")
        logger.info("- When combined with --max-trials, allocates trials proportionally across cancer patients based on their original trial counts")
        logger.info("- Trials with original label 1 are excluded from sampling when using --max-trials")
        logger.info("- This is useful for oncology-specific clinical trial research")
        logger.info("\nEVALUATION METHODS:")
        logger.info("- Default: Uses evaluate_trial() for comprehensive evaluation (title + inclusion criteria)")
        logger.info("- --title-only: Uses evaluate_title() for faster title-only evaluation")
        logger.info("- Title-only is faster but less accurate than full evaluation")
        logger.info("\nCACHE MODES:")
        logger.info("- Default: Normal cache mode - makes API calls when cache misses occur")
        logger.info("- --strict-cache-mode: Throws exception when cache is not hit (assumes all cache should hit)")
        logger.info("- Strict mode is useful for testing scenarios where you want to ensure all responses come from cache")
        logger.info("- --enable-validation: Enables cache file validation after writing (validates cache keys in saved files)")
        logger.info("\nRECOMMENDED WORKFLOW:")
        logger.info("1. First run: python script.py --coverage-only")
        logger.info("2. Check coverage status: python script.py --coverage-only")
        logger.info("3. Run benchmark: python script.py")
        logger.info("4. Run benchmark on specific patient: python script.py --patient-id <patient_id>")
        logger.info("5. Run benchmark with limited total trials: python script.py --max-trials 100 (excludes label 1 trials)")
        logger.info("6. Run benchmark only on cancer patients: python script.py --cancer-only")
        logger.info("7. Run benchmark on cancer patients with limited total trials: python script.py --cancer-only --max-trials 100 (excludes label 1 trials)")
        logger.info("8. Run benchmark with title-only evaluation (faster): python script.py --title-only")
        logger.info("9. Run benchmark with title-only evaluation on cancer patients: python script.py --title-only --cancer-only")
        logger.info("10. Run benchmark in strict cache mode: python script.py --strict-cache-mode")
        logger.info("\nTROUBLESHOOTING:")
        logger.info("- Check network connectivity and ClinicalTrials.gov access")
        logger.info("- Verify API rate limits and settings")
        logger.info("- Some trials may be permanently unavailable or withdrawn")
        logger.info("="*80 + "\n")
        return

    # Get API key (required for disease extraction)
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")

    if not api_key:
        logger.error("ERROR: OpenAI API key is required for disease extraction.")
        logger.error("Please provide it via --api-key argument or set the OPENAI_API_KEY environment variable.")
        sys.exit(1)

    # Setup logging with specified level
    global log_file

    # Set specific loggers to DEBUG if main log level is DEBUG
    specific_loggers = None
    if args.log_level == "DEBUG":
        specific_loggers = {"base.prompt_cache": "DEBUG"}

    log_file = setup_logging("benchmark_filtering", args.log_level, specific_loggers)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate output filename if not specified
    if args.output == f"results/benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json":
        output_filename = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_path = output_dir / output_filename
    else:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Run benchmark
    benchmark = FilteringBenchmark(args.dataset_path, api_key, args.cache_size, args.strict_cache_mode, args.enable_validation)

    # Show initial coverage summary
    benchmark.print_coverage_summary(args.verbose_coverage)

    # Save coverage statistics if requested
    if args.save_coverage:
        coverage_stats = benchmark.get_trial_coverage_stats()
        if args.coverage_filename:
            coverage_file = output_dir / f"{args.coverage_filename}.json"
        else:
            coverage_file = output_dir / f"coverage_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(coverage_file, 'w') as f:
            json.dump(coverage_stats, f, indent=2)
        logger.info(f"Coverage statistics saved to: {coverage_file}")

    # If only coverage check is requested, exit here
    if args.coverage_only:
        logger.info("Coverage check completed. Exiting.")
        return

    # Filter patients by patient ID if specified
    if args.patient_id:
        # Find the specific patient
        target_patient = None
        for patient in benchmark.patients:
            if patient.patient_id == args.patient_id:
                target_patient = patient
                break

        if target_patient is None:
            logger.error(f"Patient ID '{args.patient_id}' not found in the dataset.")
            logger.info(f"Available patient IDs: {[p.patient_id for p in benchmark.patients[:10]]}{'...' if len(benchmark.patients) > 10 else ''}")
            sys.exit(1)

        logger.info(f"Running benchmark on single patient: {args.patient_id}")
        logger.info(f"Patient text: {target_patient.get_text_summary(200)}")

        # Override max_patients to 1 and set patients to only the target patient
        args.max_patients = 1
        benchmark.patients = [target_patient]

    # Initialize variables for cancer filtering
    cancer_trial_ids: set[str] = set()

    # Filter for cancer patients if --cancer-only is specified
    if args.cancer_only:
        logger.info("Filtering for cancer patients only...")

        # Extract disease names for all patients to identify cancer patients
        # This uses the same disease extraction logic as the main benchmark
        cancer_patients: List[Patient] = []

        for patient in benchmark.patients:
            try:
                disease_name = extract_disease_from_record(patient.medical_record, benchmark.gpt_client)[0]
                if is_oncology_disease(disease_name):
                    cancer_patients.append(patient)

                    # Collect trial IDs associated with this cancer patient
                    # These trials will be used for sampling when --max-trials is specified
                    # Exclude trials with original label 1
                    for judgment in benchmark.relevance_judgments:
                        if judgment.patient_id == patient.patient_id and judgment.relevance_score != 1:
                            cancer_trial_ids.add(judgment.trial_id)

                    logger.info(f"Cancer patient {patient.patient_id}: {disease_name}")
                else:
                    logger.debug(f"Non-cancer patient {patient.patient_id}: {disease_name}")
            except Exception as e:
                logger.warning(f"Failed to extract disease for patient {patient.patient_id}: {e}")
                continue

        if not cancer_patients:
            logger.error("No cancer patients found in the dataset!")
            sys.exit(1)

        # Update patients list to only include cancer patients
        # This ensures the benchmark only processes cancer patients
        benchmark.patients = cancer_patients
        logger.info(f"Found {len(cancer_patients)} cancer patients out of {len(benchmark.patients) + len([p for p in benchmark.patients if p not in cancer_patients])} total patients")

        # If max_trials is specified, we need to allocate trials proportionally across cancer patients
        # This ensures that when limiting trials, we allocate based on original trial counts for each cancer patient
        # Trials with original label 1 are excluded from the allocation
        if args.max_trials is not None:
            logger.info(f"Will allocate {args.max_trials} trials proportionally across cancer patients based on their original trial counts (excluding label 1 trials)")

    # Determine the number of trials to use for the benchmark
    num_trials_to_use = args.max_trials if args.max_trials is not None else len(benchmark.trials)

    logger.debug(f"args.max_trials = {args.max_trials}")
    logger.debug(f"num_trials_to_use = {num_trials_to_use}")
    logger.debug(f"len(benchmark.trials) = {len(benchmark.trials)}")

    # Sample trials deterministically if max_trials is specified
    # Note: Trials with original label 1 are excluded from sampling to focus on label 0 and 2 trials
    if args.max_trials is not None:
        logger.debug("Entering max-trials logic")
        logger.info("Excluding trials with original label 1 from sampling to focus on label 0 and 2 trials")

        # Set a fixed seed for deterministic sampling across different executions and devices
        # This ensures that the same trials are selected regardless of:
        # - When the script is run
        # - Which device/system it's run on
        # - The order of trials in the original dataset
        random.seed(42)  # Fixed seed for reproducibility

        # Calculate trial allocation per patient based on original trial counts
        # Exclude trials with original label 1 from the allocation calculation
        patient_trial_counts: Dict[str, int] = {}
        for patient in benchmark.patients:
            patient_trial_ids: set[str] = set()
            for judgment in benchmark.relevance_judgments:
                if judgment.patient_id == patient.patient_id and judgment.relevance_score != 1:
                    patient_trial_ids.add(judgment.trial_id)
            patient_trial_counts[patient.patient_id] = len(patient_trial_ids)

        total_original_trials = sum(patient_trial_counts.values())

        # Count and log how many trials were excluded
        total_trials_before_filter = sum(len([j for j in benchmark.relevance_judgments if j.patient_id == p.patient_id]) for p in benchmark.patients)
        excluded_trials = total_trials_before_filter - total_original_trials
        logger.info(f"Excluded {excluded_trials} trials with original label 1 from sampling")

        logger.info(f"Original trial distribution across patients (excluding label 1 trials):")
        for patient_id, count in patient_trial_counts.items():
            logger.info(f"  {patient_id}: {count} trials")
        logger.info(f"Total original trials (excluding label 1): {total_original_trials}")

        # Allocate max_trials proportionally across patients
        patient_trial_allocations: Dict[str, int] = {}
        remaining_trials = num_trials_to_use

        for patient_id, original_count in patient_trial_counts.items():
            if total_original_trials > 0:
                # Calculate proportional allocation (rounded down)
                proportional_allocation = int((original_count / total_original_trials) * num_trials_to_use)
                patient_trial_allocations[patient_id] = proportional_allocation
                remaining_trials -= proportional_allocation
            else:
                patient_trial_allocations[patient_id] = 0

        # Distribute remaining trials to patients with highest original counts
        if remaining_trials > 0:
            sorted_patients = sorted(patient_trial_counts.items(), key=lambda x: x[1], reverse=True)
            for patient_id, _ in sorted_patients:
                if remaining_trials > 0:
                    patient_trial_allocations[patient_id] += 1
                    remaining_trials -= 1
                else:
                    break

        logger.info(f"Trial allocation with max_trials={num_trials_to_use} (excluding label 1 trials):")
        for patient_id, allocation in patient_trial_allocations.items():
            logger.info(f"  {patient_id}: {allocation} trials (originally had {patient_trial_counts[patient_id]} non-label-1 trials)")

        # Sample trials for each patient based on their allocation
        sampled_trial_ids: set[str] = set()
        patient_trial_mapping: Dict[str, set[str]] = {}  # Map patient_id to their allocated trials

        for patient_id, allocation in patient_trial_allocations.items():
            if allocation > 0:
                # Get trials for this patient, excluding those with original label 1
                patient_trial_ids: set[str] = set()
                for judgment in benchmark.relevance_judgments:
                    if judgment.patient_id == patient_id and judgment.relevance_score != 1:
                        patient_trial_ids.add(judgment.trial_id)

                # Filter to available trials (those that exist in benchmark.trials)
                available_patient_trials = patient_trial_ids.intersection(set(benchmark.trials.keys()))

                if args.cancer_only:
                    # When cancer-only is specified, only use cancer-related trials for this patient
                    available_patient_trials = available_patient_trials.intersection(cancer_trial_ids)

                # Sample from available trials for this patient
                if len(available_patient_trials) > 0:
                    sorted_patient_trials = sorted(list(available_patient_trials))
                    num_to_sample = min(allocation, len(sorted_patient_trials))
                    sampled_patient_trials = random.sample(sorted_patient_trials, num_to_sample)
                    sampled_trial_ids.update(sampled_patient_trials)
                    patient_trial_mapping[patient_id] = set(sampled_patient_trials)

                    logger.info(f"  Sampled {num_to_sample} trials for patient {patient_id} (excluding original label 1)")
                else:
                    logger.warning(f"  No available trials for patient {patient_id} after filtering (excluding original label 1)")
                    patient_trial_mapping[patient_id] = set()

        # Store the patient-trial mapping in the benchmark object for use during evaluation
        benchmark.patient_trial_mapping = patient_trial_mapping

        # Filter trials to only include sampled ones
        benchmark.trials = {k: v for k, v in benchmark.trials.items() if k in sampled_trial_ids}

        logger.info(f"Sampled {len(sampled_trial_ids)} trials deterministically for benchmark (seed=42, excluding label 1 trials).")
        logger.info(f"First 5 sampled trial IDs: {sorted(list(sampled_trial_ids))[:5]}")
        logger.info(f"Last 5 sampled trial IDs: {sorted(list(sampled_trial_ids))[-5:]}")

        # Re-check coverage after sampling to show updated statistics
        logger.info("Re-checking trial coverage after sampling...")
        benchmark.print_coverage_summary(args.verbose_coverage)
    else:
        if args.cancer_only:
            logger.info(f"Using all cancer-related trials for benchmark ({len(cancer_trial_ids)} trials, excluding label 1).")
        else:
            logger.info(f"Using all available trials for benchmark ({len(benchmark.trials)} trials, excluding label 1).")

    # Check current trial coverage
    coverage_stats = benchmark.get_trial_coverage_stats()
    if coverage_stats['total_missing'] > 0:
        logger.warning(f"{coverage_stats['total_missing']} trials are missing from the dataset.")
        logger.warning("The benchmark will continue with available trials.")

    gpt_filter = GPTTrialFilter(api_key=api_key, cache_size=args.cache_size)

    results = benchmark.run_benchmark(gpt_filter, args.max_patients, str(output_path), args.title_only)

    # Save results
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Export conditions cache if requested
    if args.export_conditions_cache:
        try:
            cache_file = benchmark.export_conditions_cache(str(output_path))
            logger.info(f"Conditions cache exported to: {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to export conditions cache: {e}")

    # Log final trial IDs consistency verification
    all_trial_ids = sorted(list(benchmark.trials.keys()))
    trial_ids_concat = ''.join(all_trial_ids)
    trial_ids_md5 = hashlib.md5(trial_ids_concat.encode('utf-8')).hexdigest()

    logger.info("=" * 60)
    logger.info("FINAL TRIAL IDS CONSISTENCY VERIFICATION")
    logger.info("=" * 60)
    logger.info(f"Total trials in benchmark: {len(all_trial_ids)}")
    logger.info(f"Concatenated trial IDs MD5 hash: {trial_ids_md5}")
    logger.info(f"First 5 trial IDs: {all_trial_ids[:5]}")
    logger.info(f"Last 5 trial IDs: {all_trial_ids[-5:]}")
    logger.info("=" * 60)
    logger.info(f"Benchmark results saved to: {output_path}")
    logger.info(f"Log file: {log_file}")

    # Show detailed cache status if requested
    if args.show_cache_status:
        benchmark.show_cache_status()


if __name__ == "__main__":
    main()