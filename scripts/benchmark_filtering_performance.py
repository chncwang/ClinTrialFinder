#!/usr/bin/env python3
"""
Benchmark script for evaluating filtering performance on TREC 2021 dataset.

This script evaluates the performance of clinical trial filtering algorithms
by comparing predicted eligible trials against ground truth relevance judgments.

Usage:
    python scripts/benchmark_filtering_performance.py [options]

Example:
    python scripts/benchmark_filtering_performance.py \
        --dataset-path dataset/trec_2021 \
        --output results/benchmark_results.json \
        --api-key $OPENAI_API_KEY
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from tqdm import tqdm


# Add parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from base.logging_config import setup_logging
from base.clinical_trial import ClinicalTrial

logger: logging.Logger = logging.getLogger(__name__)
# log_file will be set after parsing arguments
log_file: str = ""


class Query:
    """
    A class to represent a clinical trial query.
    
    This class encapsulates query information from the TREC 2021 dataset,
    including the query ID and text content.
    """
    
    def __init__(self, query_id: str, text: str):
        """
        Initialize a Query instance.
        
        Args:
            query_id: Unique identifier for the query
            text: The query text describing the patient case
        """
        self.query_id = query_id
        self.text = text
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Query':
        """
        Create a Query instance from a dictionary.
        
        Args:
            data: Dictionary containing query data with keys '_id' and 'text'
            
        Returns:
            Query instance
        """
        return cls(
            query_id=data['_id'],
            text=data['text']
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the Query instance to a dictionary.
        
        Returns:
            Dictionary representation of the query
        """
        return {
            '_id': self.query_id,
            'text': self.text
        }
    
    def get_patient_id(self) -> str:
        """
        Get the patient ID from the query ID.
        In TREC dataset, query_id matches patient_id.
        
        Returns:
            Patient ID string
        """
        return self.query_id
    
    def get_text_summary(self, max_length: int = 100) -> str:
        """
        Get a summary of the query text.
        
        Args:
            max_length: Maximum length of the summary
            
        Returns:
            Truncated text summary
        """
        if len(self.text) <= max_length:
            return self.text
        return self.text[:max_length] + "..."
    
    def __str__(self) -> str:
        """String representation of the query."""
        return f"Query(id={self.query_id}, text='{self.get_text_summary(50)}')"
    
    def __repr__(self) -> str:
        """Detailed string representation of the query."""
        return f"Query(query_id='{self.query_id}', text='{self.text}')"
    
    def __eq__(self, other: Any) -> bool:
        """Check if two queries are equal."""
        if not isinstance(other, Query):
            return False
        return (self.query_id == other.query_id and 
                self.text == other.text)
    
    def __hash__(self) -> int:
        """Hash based on query_id."""
        return hash(self.query_id)


class RelevanceJudgment:
    """
    A class to represent a single relevance judgment.
    
    This class encapsulates a single relevance judgment from the TREC 2021 dataset,
    including the query ID, trial ID, and relevance score.
    """
    
    def __init__(self, query_id: str, trial_id: str, relevance_score: int):
        """
        Initialize a RelevanceJudgment instance.
        
        Args:
            query_id: Unique identifier for the query
            trial_id: Unique identifier for the trial (NCT ID)
            relevance_score: Relevance score (0 = not relevant, >0 = relevant)
        """
        self.query_id = query_id
        self.trial_id = trial_id
        self.relevance_score = relevance_score
    
    @classmethod
    def from_tsv_line(cls, line: str) -> Optional['RelevanceJudgment']:
        """
        Create a RelevanceJudgment instance from a TSV line.
        
        Args:
            line: TSV line with format: query_id\ttrial_id\trelevance_score
            
        Returns:
            RelevanceJudgment instance or None if line is invalid
        """
        parts = line.strip().split('\t')
        if len(parts) == 3:
            query_id, trial_id, relevance = parts
            # Skip header row
            if relevance == 'score':
                return None
            try:
                return cls(query_id, trial_id, int(relevance))
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
            'query_id': self.query_id,
            'trial_id': self.trial_id,
            'relevance_score': self.relevance_score
        }
    
    def __str__(self) -> str:
        """String representation of the relevance judgment."""
        return f"RelevanceJudgment(query_id='{self.query_id}', trial_id='{self.trial_id}', relevance_score={self.relevance_score})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the relevance judgment."""
        return f"RelevanceJudgment(query_id='{self.query_id}', trial_id='{self.trial_id}', relevance_score={self.relevance_score})"
    
    def __eq__(self, other: Any) -> bool:
        """Check if two RelevanceJudgment instances are equal."""
        if not isinstance(other, RelevanceJudgment):
            return False
        return (self.query_id == other.query_id and 
                self.trial_id == other.trial_id and 
                self.relevance_score == other.relevance_score)
    
    def __hash__(self) -> int:
        """Hash value for the relevance judgment."""
        return hash((self.query_id, self.trial_id, self.relevance_score))


class FilteringBenchmark:
    """Benchmark class for evaluating clinical trial filtering performance."""
    
    def __init__(self, dataset_path: str, api_key: Optional[str] = None, cache_size: int = 100000):
        """
        Initialize the benchmark.
        
        Args:
            dataset_path: Path to TREC 2021 dataset directory
            api_key: OpenAI API key for GPT filtering (optional)
            cache_size: Size of GPT response cache
        """
        self.dataset_path = Path(dataset_path)
        self.api_key = api_key
        self.cache_size = cache_size
        
        # Initialize trials attribute
        self.trials: Dict[str, Dict[str, Any]] = {}
        
        # Load dataset
        self._load_dataset()
        
    def _load_dataset(self):
        """Load TREC 2021 dataset components."""
        logger.info("FilteringBenchmark._load_dataset: Loading TREC 2021 dataset...")
        
        # Load queries
        queries_file = self.dataset_path / "queries.jsonl"
        self.queries: List[Query] = []
        with open(queries_file, 'r') as f:
            for line in f:
                query_data = json.loads(line)
                self.queries.append(Query.from_dict(query_data))
        
        # Log the number of queries and the first 10 queries
        logger.info(f"FilteringBenchmark._load_dataset: Loaded {len(self.queries)} queries")
        logger.info("FilteringBenchmark._load_dataset: First 10 queries:")
        for i, query in enumerate(self.queries[:10], 1):
            logger.info(f"  {i}. {query.query_id}: {query.get_text_summary(100)}")
        
        # Load relevance judgments
        qrels_file = self.dataset_path / "qrels" / "test.tsv"
        self.relevance_judgments: List[RelevanceJudgment] = []
        with open(qrels_file, 'r') as f:
            for line in f:
                judgment = RelevanceJudgment.from_tsv_line(line)
                if judgment is not None:
                    self.relevance_judgments.append(judgment)
        
        # Validate that the set of query_ids in relevance_judgments is the same as the set of query_ids in queries
        relevance_query_ids = set(judgment.query_id for judgment in self.relevance_judgments)
        query_ids = set(query.query_id for query in self.queries)
        if relevance_query_ids != query_ids:
            raise ValueError("The set of query_ids in relevance_judgments is not the same as the set of query_ids in queries")
        
        # Log the number of relevance judgments and the first 10 relevance judgments
        logger.info(f"FilteringBenchmark._load_dataset: Loaded {len(self.relevance_judgments)} relevance judgments")
        logger.info("FilteringBenchmark._load_dataset: First 10 relevance judgments:")
        for i, judgment in enumerate(self.relevance_judgments[:10], 1):
            logger.info(f"  {i}. {judgment}")
        
        # Extract unique trial IDs from relevance judgments
        trial_ids = set(judgment.trial_id for judgment in self.relevance_judgments)
        logger.info(f"FilteringBenchmark._load_dataset: Found {len(trial_ids)} unique trial IDs in relevance judgments")
        
        # Check if trial data file exists, if not download trials
        trials_file = self.dataset_path / "retrieved_trials.json"
        if not trials_file.exists():
            logger.info("FilteringBenchmark._load_dataset: Trial data file not found. Downloading trials...")
            try:
                self.download_trials(trial_ids, delay=0.1, timeout=30, save_individual=False, individual_format="json")  # Use default delay and timeout for initial download
                # Verify the file was created
                if not trials_file.exists():
                    raise RuntimeError("Trial data file was not created after download")
            except Exception as e:
                logger.error(f"FilteringBenchmark._load_dataset: Failed to download trials: {e}")
                raise RuntimeError(f"Failed to download trials: {e}")
        else:
            logger.info(f"FilteringBenchmark._load_dataset: Trial data file found: {trials_file}")
        
        # Load trial data
        self.trials = self._load_trials(trials_file)
        if not self.trials:
            raise RuntimeError("No trials were loaded from the trial data file")
        logger.info(f"FilteringBenchmark._load_dataset: Loaded {len(self.trials)} trials")
        
        # Validate trial coverage
        missing_trials = self.get_missing_trials()
        if missing_trials:
            logger.warning(f"FilteringBenchmark._load_dataset: {len(missing_trials)} trials are missing from the dataset. Continuing with available trials.")
            logger.warning(f"Missing trials: {missing_trials}")
        else:
            logger.info("FilteringBenchmark._load_dataset: All required trials are available")
        
        # Log coverage statistics
        coverage_stats = self.get_trial_coverage_stats()
        logger.info(f"FilteringBenchmark._load_dataset: Trial coverage: {coverage_stats['coverage_percentage']:.1f}% ({coverage_stats['total_available']}/{coverage_stats['total_required']})")
        
        logger.info(f"FilteringBenchmark._load_dataset: Loaded {len(self.queries)} queries")
        logger.info(f"FilteringBenchmark._load_dataset: Loaded {len(self.relevance_judgments)} relevance judgments")
        logger.info(f"FilteringBenchmark._load_dataset: Loaded {len(self.trials)} trials")

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
                        logger.debug(f"FilteringBenchmark._download_trials: Response for {trial_id}: {response.status_code}")
                        
                        if response.status_code == 200:
                            data = response.json()
                            logger.debug(f"FilteringBenchmark._download_trials: Data for {trial_id}: {data}")
                            
                            if "protocolSection" in data:
                                # Extract trial data using the same structure as the spider
                                trial_data = self._extract_trial_data(data["protocolSection"])
                                trials_data.append(ClinicalTrial(trial_data))
                                
                                # Save individual trial file if requested
                                if save_individual:
                                    individual_file = self.dataset_path / f"trial_{trial_id}.json"
                                    with open(individual_file, 'w') as f:
                                        json.dump(trial_data, f, indent=2)
                            else:
                                logger.warning(f"FilteringBenchmark._download_trials: No protocol section found for {trial_id}")
                                
                        else:
                            logger.warning(f"FilteringBenchmark._download_trials: Failed to download {trial_id}, status: {response.status_code}")
                            
                    except requests.exceptions.SSLError as ssl_error:
                        logger.warning(f"FilteringBenchmark._download_trials: SSL error for {trial_id}, retrying without verification: {ssl_error}")
                        # Retry without SSL verification
                        response = requests.get(url, params=params, timeout=timeout, verify=False)
                        if response.status_code == 200:
                            data = response.json()
                            if "protocolSection" in data:
                                trial_data = self._extract_trial_data(data["protocolSection"])
                                trials_data.append(ClinicalTrial(trial_data))
                                
                                # Save individual trial file if requested
                                if save_individual:
                                    individual_file = self.dataset_path / f"trial_{trial_id}.json"
                                    with open(individual_file, 'w') as f:
                                        json.dump(trial_data, f, indent=2)
                            else:
                                logger.warning(f"FilteringBenchmark._download_trials: No protocol section found for {trial_id}")
                        else:
                            logger.warning(f"FilteringBenchmark._download_trials: Failed to download {trial_id} even without SSL verification, status: {response.status_code}")
                            
                    # Rate limiting - be respectful to the API
                    time.sleep(delay)  # Delay between requests
                    
                except Exception as e:
                    logger.error(f"FilteringBenchmark._download_trials: Error downloading {trial_id}: {e}")
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
                    logger.info(f"FilteringBenchmark.download_trials: Loaded {len(existing_trials)} existing trials")
                except Exception as e:
                    logger.warning(f"FilteringBenchmark.download_trials: Could not load existing trials: {e}")
                    existing_trials = []
            
            # Merge existing and new trials, avoiding duplicates
            existing_trial_ids = {trial.get('identification', {}).get('nct_id') for trial in existing_trials if trial.get('identification', {}).get('nct_id')}
            new_trials_only = [trial.to_dict() for trial in trials_data if trial.identification.nct_id not in existing_trial_ids]
            
            all_trials: List[Dict[str, Any]] = existing_trials + new_trials_only
            logger.info(f"FilteringBenchmark.download_trials: Saving {len(all_trials)} total trials (existing: {len(existing_trials)}, new unique: {len(new_trials_only)})")
            
            with open(output_path, 'w') as f:
                json.dump(all_trials, f, indent=2)
        else:
            raise RuntimeError("No trials were successfully downloaded")
    
    def _extract_trial_data(self, protocol: Dict[str, Any]) -> Dict[str, Any]:
        """Extract trial data from protocol section (same structure as spider)."""
        def safe_get(d: Any, *keys: str, default: Any = None) -> Any:
            """Safely get nested dictionary values"""
            for key in keys:
                if not isinstance(d, dict):
                    return default
                d = d.get(str(key), default)  # type: ignore
                if d is None:
                    return default
            return d
        
        return {
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
    
    def retry_download_missing_trials(self, batch_size: int = 1000, delay: float = 0.1, timeout: int = 30, save_individual: bool = False, individual_format: str = "json") -> int:
        """
        Retry downloading trials that were previously missing.
        
        Args:
            batch_size: Number of trials to download in a single batch
            delay: Delay in seconds between individual trial downloads
            timeout: Timeout in seconds for individual trial downloads
            save_individual: Whether to save individual trial files
            
        Returns:
            Number of newly downloaded trials
        """
        missing_trials = self.get_missing_trials()
        if not missing_trials:
            logger.info("FilteringBenchmark.retry_download_missing_trials: No missing trials to download")
            return 0
        
        logger.info(f"FilteringBenchmark.retry_download_missing_trials: Attempting to download {len(missing_trials)} missing trials in batches of {batch_size}")
        
        # Download missing trials in batches
        all_missing = list(missing_trials)
        for i in range(0, len(all_missing), batch_size):
            batch = all_missing[i:i + batch_size]
            logger.info(f"FilteringBenchmark.retry_download_missing_trials: Downloading batch {i//batch_size + 1}/{(len(all_missing) + batch_size - 1)//batch_size} ({len(batch)} trials)")
            self.download_trials(set(batch), delay=delay, timeout=timeout, save_individual=save_individual, individual_format=individual_format)
        
        # Reload trials to include newly downloaded ones
        trials_file = self.dataset_path / "retrieved_trials.json"
        if trials_file.exists():
            new_trials = self._load_trials(trials_file)
            # Update trials dict with new trials
            self.trials.update(new_trials)
            
            # Check new coverage
            new_missing = self.get_missing_trials()
            newly_downloaded = len(missing_trials) - len(new_missing)
            
            logger.info(f"FilteringBenchmark.retry_download_missing_trials: Successfully downloaded {newly_downloaded} new trials")
            logger.info(f"FilteringBenchmark.retry_download_missing_trials: Remaining missing trials: {len(new_missing)}")
            
            return newly_downloaded
        else:
            logger.warning("FilteringBenchmark.retry_download_missing_trials: No trials file found after download attempt")
            return 0
    
    def _load_trials(self, trials_file: Path) -> Dict[str, Dict[str, Any]]:
        """Load trial data from file."""
        try:
            with open(trials_file, 'r') as f:
                trials_data: List[Dict[str, Any]] = json.load(f)
            
            # Convert to dictionary with NCT ID as key for faster lookup
            trials_dict: Dict[str, Dict[str, Any]] = {}
            for trial in trials_data:
                nct_id = trial.get('identification', {}).get('nct_id')
                if nct_id:
                    trials_dict[nct_id] = trial
            
            logger.info(f"FilteringBenchmark._load_trials: Successfully loaded {len(trials_dict)} trials")
            return trials_dict
            
        except Exception as e:
            logger.error(f"FilteringBenchmark._load_trials: Error loading trials: {e}")
            return {}
    
    def get_trial_data(self, trial_id: str) -> Optional[Dict[str, Any]]:
        """
        Get trial data by NCT ID.
        
        Args:
            trial_id: NCT ID of the trial
            
        Returns:
            Trial data dictionary or None if not found
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
        available_trials = set(self.trials.keys())
        required_trials = set(judgment.trial_id for judgment in self.relevance_judgments)
        missing_trials = required_trials - available_trials
        
        total_required = len(required_trials)
        total_available = len(available_trials)
        total_missing = len(missing_trials)
        coverage_percentage = (total_available / total_required * 100) if total_required > 0 else 0.0
        
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
        
        print("\n" + "="*60)
        print("TRIAL COVERAGE SUMMARY")
        print("="*60)
        print(f"Total trials required: {coverage_stats['total_required']}")
        print(f"Total trials available: {coverage_stats['total_available']}")
        print(f"Total trials missing: {coverage_stats['total_missing']}")
        print(f"Coverage: {coverage_stats['coverage_percentage']:.1f}%")
        
        if coverage_stats['missing_trials']:
            if verbose:
                print(f"\nMissing trials ({len(coverage_stats['missing_trials'])}):")
                for i, trial_id in enumerate(coverage_stats['missing_trials'], 1):
                    print(f"  {i:2d}. {trial_id}")
            else:
                print(f"\nMissing trials: {', '.join(coverage_stats['missing_trials'])}")
            
            print(f"\nTo improve coverage, you can:")
            print(f"  1. Run with --retry-download flag to attempt re-download")
            print(f"  2. Check if trials are available on ClinicalTrials.gov")
            print(f"  3. Verify network connectivity and API access")
        
        print("="*60 + "\n")
    
    def validate_trial_coverage(self) -> bool:
        """
        Validate that all trials referenced in relevance judgments are available in trials data.
        
        Returns:
            True if all trials are available, False otherwise
        """
        missing_trials = self.get_missing_trials()
        if missing_trials:
            logger.warning(f"FilteringBenchmark.validate_trial_coverage: {len(missing_trials)} trials are missing: {missing_trials[:10]}{'...' if len(missing_trials) > 10 else ''}")
            return False
        else:
            logger.info("FilteringBenchmark.validate_trial_coverage: All required trials are available")
            return True
    
    def get_ground_truth_trials(self, query_id: str) -> List[str]:
        """
        Get ground truth relevant trials for a query.
        
        Args:
            query_id: Query identifier
            
        Returns:
            List of NCT IDs of relevant trials (only those available in the dataset)
        """
        relevant_trials: List[str] = []
        available_trials = set(self.trials.keys())
        
        for judgment in self.relevance_judgments:
            if judgment.query_id == query_id and judgment.is_relevant():
                # Only include trials that are actually available in the dataset
                if judgment.trial_id in available_trials:
                    relevant_trials.append(judgment.trial_id)
                else:
                    logger.debug(f"Trial {judgment.trial_id} is not available in dataset, skipping")
        
        if not relevant_trials:
            logger.warning(f"Query ID '{query_id}' not found in relevance judgments or has no relevant trials available in dataset.")
        
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
    
    def evaluate_filtering_performance(self, query: Query) -> Dict[str, Any]:
        """
        Evaluate filtering performance for a single query.
        
        Args:
            query: Query object
            
        Returns:
            Dictionary with evaluation metrics
        """
        query_id = query.query_id
        
        logger.info(f"FilteringBenchmark.evaluate_filtering_performance: Evaluating query: {query_id}")
        
        # Get ground truth relevant trials
        ground_truth_trials = self.get_ground_truth_trials(query_id)
        logger.info(f"FilteringBenchmark.evaluate_filtering_performance: Ground truth relevant trials: {len(ground_truth_trials)}")
        
        # Check if there are any ground truth trials available for this query
        if not ground_truth_trials:
            logger.warning(f"FilteringBenchmark.evaluate_filtering_performance: No ground truth trials available for query {query_id}, skipping evaluation")
            return {
                'query_id': query_id,
                'ground_truth_count': 0,
                'retrieved_count': 0,
                'predicted_eligible_count': 0,
                'true_positives': 0,
                'false_positives': 0,
                'false_negatives': 0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'accuracy': 0.0,
                'processing_time': 0.0,
                'api_cost': 0.0,
                'error': 'No ground truth trials available',
                'skipped': True
            }
        
        try:
            # TODO: Implement filtering
            return {
                'query_id': query_id,
                'ground_truth_count': len(ground_truth_trials),
                'retrieved_count': 0,
                'predicted_eligible_count': 0,
                'true_positives': 0,
                'false_positives': 0,
                'false_negatives': 0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'accuracy': 0.0,
                'processing_time': 0.0,
                'api_cost': 0.0,
                'error': None,
                'skipped': False
            }
            
        except Exception as e:
            logger.error(f"FilteringBenchmark.evaluate_filtering_performance: Error processing query {query_id}: {str(e)}")
            return {
                'query_id': query_id,
                'ground_truth_count': len(ground_truth_trials),
                'retrieved_count': 0,
                'predicted_eligible_count': 0,
                'true_positives': 0,
                'false_positives': 0,
                'false_negatives': 0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'accuracy': 0.0,
                'error': str(e),
                'skipped': False
            }
    
    def run_benchmark(self, max_queries: Optional[int] = None) -> Dict[str, Any]:
        """
        Run the complete benchmark.
        
        Args:
            max_queries: Maximum number of queries to process (for testing)
            
        Returns:
            Dictionary with overall benchmark results
        """
        logger.info("FilteringBenchmark.run_benchmark: Starting filtering performance benchmark...")
        
        queries_to_process = self.queries[:max_queries] if max_queries else self.queries
        
        results: List[Dict[str, Any]] = []
        total_processing_time = 0.0
        total_api_cost = 0.0
        
        # Create progress bar for query processing
        with tqdm(total=len(queries_to_process), desc="Processing queries", unit="query") as pbar:
            for i, query in enumerate(queries_to_process, 1):
                # Update progress bar description with current query ID
                pbar.set_description(f"Processing {query.query_id}")
                
                result = self.evaluate_filtering_performance(query)
                results.append(result)
                
                total_processing_time += result['processing_time']
                total_api_cost += result['api_cost']
                
                # Update progress bar
                pbar.update(1)
                
                # Log progress every 10 queries
                if i % 10 == 0:
                    logger.info(f"FilteringBenchmark.run_benchmark: Processed {i} queries. Total time: {total_processing_time:.2f}s, Total cost: ${total_api_cost:.4f}")
        
        # Calculate aggregate metrics
        successful_results = [r for r in results if r['error'] is None and not r.get('skipped', False)]
        skipped_results = [r for r in results if r.get('skipped', False)]
        failed_results = [r for r in results if r['error'] is not None and not r.get('skipped', False)]
        
        if successful_results:
            avg_precision = sum(r['precision'] for r in successful_results) / len(successful_results)
            avg_recall = sum(r['recall'] for r in successful_results) / len(successful_results)
            avg_f1 = sum(r['f1_score'] for r in successful_results) / len(successful_results)
            avg_accuracy = sum(r['accuracy'] for r in successful_results) / len(successful_results)
        else:
            avg_precision = avg_recall = avg_f1 = avg_accuracy = 0.0
        
        # Get trial coverage statistics
        coverage_stats = self.get_trial_coverage_stats()
        
        benchmark_results = {
            'timestamp': datetime.now().isoformat(),
            'dataset_path': str(self.dataset_path),
            'trial_coverage': coverage_stats,
            'total_queries': len(queries_to_process),
            'successful_queries': len(successful_results),
            'skipped_queries': len(skipped_results),
            'failed_queries': len(failed_results),
            'total_processing_time': total_processing_time,
            'total_api_cost': total_api_cost,
            'average_processing_time_per_query': total_processing_time / len(queries_to_process) if queries_to_process else 0.0,
            'average_api_cost_per_query': total_api_cost / len(queries_to_process) if queries_to_process else 0.0,
            'metrics': {
                'average_precision': avg_precision,
                'average_recall': avg_recall,
                'average_f1_score': avg_f1,
                'average_accuracy': avg_accuracy
            },
            'detailed_results': results
        }
        
        logger.info("FilteringBenchmark.run_benchmark: Benchmark completed!")
        logger.info(f"FilteringBenchmark.run_benchmark: Total queries processed: {len(queries_to_process)}")
        logger.info(f"FilteringBenchmark.run_benchmark: Successful queries: {len(successful_results)}")
        logger.info(f"FilteringBenchmark.run_benchmark: Skipped queries: {len(skipped_results)}")
        logger.info(f"FilteringBenchmark.run_benchmark: Failed queries: {len(failed_results)}")
        logger.info(f"FilteringBenchmark.run_benchmark: Trial coverage: {coverage_stats['coverage_percentage']:.1f}% ({coverage_stats['total_available']}/{coverage_stats['total_required']})")
        logger.info(f"FilteringBenchmark.run_benchmark: Total processing time: {total_processing_time:.2f}s")
        logger.info(f"FilteringBenchmark.run_benchmark: Total API cost: ${total_api_cost:.4f}")
        logger.info(f"FilteringBenchmark.run_benchmark: Average precision: {avg_precision:.4f}")
        logger.info(f"FilteringBenchmark.run_benchmark: Average recall: {avg_recall:.4f}")
        logger.info(f"FilteringBenchmark.run_benchmark: Average F1-score: {avg_f1:.4f}")
        
        return benchmark_results


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
        help="OpenAI API key (alternatively, set OPENAI_API_KEY environment variable)"
    )
    parser.add_argument(
        "--cache-size",
        type=int,
        default=100000,
        help="Size of GPT response cache"
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        help="Maximum number of queries to process (for testing)"
    )
    parser.add_argument(
        "--retry-download",
        action="store_true",
        help="Retry downloading missing trials before running benchmark"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of retry attempts for downloading missing trials"
    )
    parser.add_argument(
        "--retry-delay",
        type=int,
        default=5,
        help="Delay in seconds between retry attempts for downloading missing trials"
    )
    parser.add_argument(
        "--max-missing-trials",
        type=int,
        help="Maximum number of missing trials allowed before continuing with benchmark"
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
    
    args = parser.parse_args()
    
    # Show additional help if requested
    if args.show_help:
        print("\n" + "="*80)
        print("TRIAL COVERAGE HELP")
        print("="*80)
        print("This script evaluates clinical trial filtering performance on the TREC 2021 dataset.")
        print("It requires downloading clinical trial data from ClinicalTrials.gov.")
        print("\nTRIAL COVERAGE:")
        print("- The script checks if all trials referenced in the dataset are available")
        print("- Missing trials can occur due to network issues, API limits, or trial unavailability")
        print("- Use --coverage-only to check status without running the full benchmark")
        print("- Use --retry-download to attempt re-downloading missing trials")
        print("\nRECOMMENDED WORKFLOW:")
        print("1. First run: python script.py --coverage-only")
        print("2. If trials are missing: python script.py --retry-download")
        print("3. Check coverage again: python script.py --coverage-only")
        print("4. Run benchmark: python script.py")
        print("\nTROUBLESHOOTING:")
        print("- Check network connectivity and ClinicalTrials.gov access")
        print("- Verify API rate limits and retry settings")
        print("- Some trials may be permanently unavailable or withdrawn")
        print("="*80 + "\n")
        return
    
    # Get API key (optional for current implementation)
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    
    # Setup logging with specified level
    global log_file
    log_file = setup_logging("benchmark_filtering", args.log_level)
    
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
    benchmark = FilteringBenchmark(args.dataset_path, api_key, args.cache_size)
    
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
        logger.info(f"FilteringBenchmark.main: Coverage statistics saved to: {coverage_file}")
    
    # Optionally retry downloading missing trials
    if args.retry_download:
        logger.info(f"FilteringBenchmark.main: Retrying download of missing trials (max {args.max_retries} attempts)...")
        
        for attempt in range(1, args.max_retries + 1):
            logger.info(f"FilteringBenchmark.main: Download attempt {attempt}/{args.max_retries}")
            newly_downloaded = benchmark.retry_download_missing_trials(args.batch_size, args.download_delay, args.download_timeout, args.save_individual_trials, args.individual_trial_format)
            
            if newly_downloaded > 0:
                logger.info(f"FilteringBenchmark.main: Successfully downloaded {newly_downloaded} additional trials on attempt {attempt}")
                # Show updated coverage summary
                benchmark.print_coverage_summary(args.verbose_coverage)
                
                # Check if we have full coverage
                coverage_stats = benchmark.get_trial_coverage_stats()
                if coverage_stats['total_missing'] == 0:
                    logger.info("FilteringBenchmark.main: Full trial coverage achieved!")
                    break
            else:
                logger.info(f"FilteringBenchmark.main: No additional trials were downloaded on attempt {attempt}")
                if attempt < args.max_retries:
                    logger.info(f"FilteringBenchmark.main: Waiting {args.retry_delay} seconds before next attempt...")
                    time.sleep(args.retry_delay)
                else:
                    logger.warning("FilteringBenchmark.main: Maximum retry attempts reached")
    
    # If only coverage check is requested, exit here
    if args.coverage_only:
        logger.info("FilteringBenchmark.main: Coverage check completed. Exiting.")
        return
    
    # Check if missing trials exceed the threshold
    if args.max_missing_trials is not None:
        coverage_stats = benchmark.get_trial_coverage_stats()
        if coverage_stats['total_missing'] > args.max_missing_trials:
            logger.error(f"FilteringBenchmark.main: Too many missing trials ({coverage_stats['total_missing']}) exceed threshold ({args.max_missing_trials})")
            logger.error("FilteringBenchmark.main: Consider using --retry-download or check trial availability")
            sys.exit(1)
        else:
            logger.info(f"FilteringBenchmark.main: Missing trials ({coverage_stats['total_missing']}) within acceptable threshold ({args.max_missing_trials})")
    
    results = benchmark.run_benchmark(args.max_queries)
    
    # Save results
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"FilteringBenchmark.main: Benchmark results saved to: {output_path}")
    logger.info(f"FilteringBenchmark.main: Log file: {log_file}")


if __name__ == "__main__":
    main() 