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


# Add parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from base.logging_config import setup_logging

logger: logging.Logger = logging.getLogger(__name__)
log_file: str = setup_logging("benchmark_filtering", "INFO")


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
        
        # Load dataset
        self._load_dataset()
        
    def _load_dataset(self):
        """Load TREC 2021 dataset components."""
        logger.info("FilteringBenchmark._load_dataset: Loading TREC 2021 dataset...")
        
        # Load queries
        queries_file = self.dataset_path / "queries.jsonl"
        self.queries: List[Dict[str, Any]] = []
        with open(queries_file, 'r') as f:
            for line in f:
                self.queries.append(json.loads(line))
        
        # Log the number of queries and the first 10 queries
        logger.info(f"FilteringBenchmark._load_dataset: Loaded {len(self.queries)} queries")
        logger.info("FilteringBenchmark._load_dataset: First 10 queries:")
        for i, query in enumerate(self.queries[:10], 1):
            logger.info(f"  {i}. {query['_id']}: {query['text'][:100]}...")
        
        # Load relevance judgments
        qrels_file = self.dataset_path / "qrels" / "test.tsv"
        self.relevance_judgments: Dict[str, Dict[str, int]] = {}
        with open(qrels_file, 'r') as f:
            for line_num, line in enumerate(f):
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    query_id, trial_id, relevance = parts
                    # Skip header row
                    if line_num == 0 and relevance == 'score':
                        continue
                    try:
                        if query_id not in self.relevance_judgments:
                            self.relevance_judgments[query_id] = {}
                        self.relevance_judgments[query_id][trial_id] = int(relevance)
                    except ValueError:
                        # Skip lines that can't be converted to int
                        continue
        
        # Log the number of relevance judgments and the first 10 relevance judgments
        logger.info(f"FilteringBenchmark._load_dataset: Loaded {len(self.relevance_judgments)} relevance judgments")
        logger.info("FilteringBenchmark._load_dataset: First 10 relevance judgments:")
        for i, (query_id, relevance_dict) in enumerate(list(self.relevance_judgments.items())[:10], 1):
            # Log only the first 5 trial_ids and their relevance scores for brevity
            short_relevance = dict(list(relevance_dict.items())[:5])
            more = "..." if len(relevance_dict) > 5 else ""
            logger.info(f"  {i}. {query_id}: {short_relevance}{more}")
        
        # Load retrieved trials
        trials_file = self.dataset_path / "retrieved_trials.json"
        with open(trials_file, 'r') as f:
            self.retrieved_trials = json.load(f)
        
        # Log the number of retrieved trials and the first 10 retrieved trials for the first patient
        logger.info(f"FilteringBenchmark._load_dataset: Loaded {len(self.retrieved_trials)} retrieved trials")
        logger.info("FilteringBenchmark._load_dataset: First 10 retrieved trials for the first patient:")
        if self.retrieved_trials:
            first_patient = self.retrieved_trials[0]
            logger.info(f"FilteringBenchmark._load_dataset: First patient ID: {first_patient.get('patient_id', 'Unknown')}")
            
            # Get trials from the first patient (they're stored as arrays with numeric keys)
            patient_trials: List[Dict[str, Any]] = []
            seen_trial_ids: set[str] = set()
            for key, value in first_patient.items():
                if key not in ['patient_id', 'patient'] and isinstance(value, list):
                    for trial in value:  # type: ignore
                        trial_id = trial.get('NCTID', '')  # type: ignore
                        if trial_id and trial_id not in seen_trial_ids:
                            patient_trials.append(trial)  # type: ignore
                            seen_trial_ids.add(trial_id)  # type: ignore
            
            logger.info(f"FilteringBenchmark._load_dataset: Found {len(patient_trials)} trials for first patient")
            for i, trial in enumerate(patient_trials[:10], 1):
                trial_id = trial.get('NCTID', 'Unknown')
                title = trial.get('brief_title', 'No title')[:80]
                logger.info(f"FilteringBenchmark._load_dataset: {i}. {trial_id}: {title}...")
        
        logger.info(f"FilteringBenchmark._load_dataset: Loaded {len(self.queries)} queries")
        logger.info(f"FilteringBenchmark._load_dataset: Loaded relevance judgments for {len(self.relevance_judgments)} queries")
        logger.info(f"FilteringBenchmark._load_dataset: Loaded retrieved trials for {len(self.retrieved_trials)} patients")
    

    
    def get_ground_truth_trials(self, query_id: str) -> List[str]:
        """
        Get ground truth relevant trials for a query.
        
        Args:
            query_id: Query identifier
            
        Returns:
            List of NCT IDs of relevant trials
        """
        if query_id not in self.relevance_judgments:
            raise KeyError(f"Query ID '{query_id}' not found in relevance judgments.")
        
        relevant_trials: List[str] = []
        for trial_id, relevance in self.relevance_judgments[query_id].items():
            if relevance > 0:  # Consider trials with relevance > 0 as relevant
                relevant_trials.append(trial_id)
        
        return relevant_trials
    
    def get_retrieved_trials_for_patient(self, patient_id: str) -> List[Dict[str, Any]]:
        """
        Get retrieved trials for a specific patient.
        
        Args:
            patient_id: Patient identifier
            
        Returns:
            List of trial dictionaries (deduplicated by NCT ID)
        """
        for patient_data in self.retrieved_trials:
            if patient_data.get('patient_id') == patient_id:
                # Extract trials from the patient data structure
                trials: List[Dict[str, Any]] = []
                seen_trial_ids: set[str] = set()
                for key, value in patient_data.items():
                    if key not in ['patient_id', 'patient'] and isinstance(value, list):
                        # Trials are stored as arrays with numeric keys
                        for trial in value:  # type: ignore
                            trial_id = trial.get('NCTID', '')  # type: ignore
                            if trial_id and trial_id not in seen_trial_ids:
                                trials.append(trial)  # type: ignore
                                seen_trial_ids.add(trial_id)  # type: ignore
                return trials
        raise KeyError(f"Patient ID '{patient_id}' not found in retrieved trials.")
    
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
    
    def evaluate_filtering_performance(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate filtering performance for a single query.
        
        Args:
            query: Query dictionary
            
        Returns:
            Dictionary with evaluation metrics
        """
        query_id = query['_id']
        patient_id = query_id  # In TREC dataset, query_id matches patient_id
        
        logger.info(f"FilteringBenchmark.evaluate_filtering_performance: Evaluating query: {query_id}")
        
        # Get ground truth relevant trials
        ground_truth_trials = self.get_ground_truth_trials(query_id)
        logger.info(f"FilteringBenchmark.evaluate_filtering_performance: Ground truth relevant trials: {len(ground_truth_trials)}")
        
        # Get retrieved trials for this patient
        retrieved_trials_data = self.get_retrieved_trials_for_patient(patient_id)
        logger.info(f"FilteringBenchmark.evaluate_filtering_performance: Retrieved trials: {len(retrieved_trials_data)}")
        
        if not retrieved_trials_data:
            raise ValueError(f"No retrieved trials found for patient {patient_id}")
        
        # Apply filtering directly on the trial data
        start_time = time.time()
        
        try:
            # Simple filtering approach - consider all trials as eligible for now
            # In a real implementation, you would apply your filtering algorithm here
            predicted_eligible_trials = [trial.get('NCTID', '') for trial in retrieved_trials_data]
            total_cost = 0.0
            
            processing_time = time.time() - start_time
            
            # Calculate metrics
            metrics = self.calculate_metrics(predicted_eligible_trials, ground_truth_trials)
            
            return {
                'query_id': query_id,
                'ground_truth_count': len(ground_truth_trials),
                'retrieved_count': len(retrieved_trials_data),
                'predicted_eligible_count': len(predicted_eligible_trials),
                'true_positives': metrics['true_positives'],
                'false_positives': metrics['false_positives'],
                'false_negatives': metrics['false_negatives'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score'],
                'accuracy': metrics['accuracy'],
                'processing_time': processing_time,
                'api_cost': total_cost,
                'error': None
            }
            
        except Exception as e:
            logger.error(f"FilteringBenchmark.evaluate_filtering_performance: Error processing query {query_id}: {str(e)}")
            return {
                'query_id': query_id,
                'ground_truth_count': len(ground_truth_trials),
                'retrieved_count': len(retrieved_trials_data),
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
                'error': str(e)
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
        
        for i, query in enumerate(queries_to_process, 1):
            logger.info(f"FilteringBenchmark.run_benchmark: Processing query {i}/{len(queries_to_process)}")
            
            result = self.evaluate_filtering_performance(query)
            results.append(result)
            
            total_processing_time += result['processing_time']
            total_api_cost += result['api_cost']
            
            # Log progress
            if i % 10 == 0:
                logger.info(f"FilteringBenchmark.run_benchmark: Processed {i} queries. Total time: {total_processing_time:.2f}s, Total cost: ${total_api_cost:.4f}")
        
        # Calculate aggregate metrics
        successful_results = [r for r in results if r['error'] is None]
        
        if successful_results:
            avg_precision = sum(r['precision'] for r in successful_results) / len(successful_results)
            avg_recall = sum(r['recall'] for r in successful_results) / len(successful_results)
            avg_f1 = sum(r['f1_score'] for r in successful_results) / len(successful_results)
            avg_accuracy = sum(r['accuracy'] for r in successful_results) / len(successful_results)
        else:
            avg_precision = avg_recall = avg_f1 = avg_accuracy = 0.0
        
        benchmark_results = {
            'timestamp': datetime.now().isoformat(),
            'dataset_path': str(self.dataset_path),
            'total_queries': len(queries_to_process),
            'successful_queries': len(successful_results),
            'failed_queries': len(results) - len(successful_results),
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
        logger.info(f"FilteringBenchmark.run_benchmark: Failed queries: {len(results) - len(successful_results)}")
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
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Get API key (optional for current implementation)
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Run benchmark
    benchmark = FilteringBenchmark(args.dataset_path, api_key, args.cache_size)
    results = benchmark.run_benchmark(args.max_queries)
    
    # Save results
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"FilteringBenchmark.main: Benchmark results saved to: {output_path}")
    logger.info(f"FilteringBenchmark.main: Log file: {log_file}")


if __name__ == "__main__":
    main() 