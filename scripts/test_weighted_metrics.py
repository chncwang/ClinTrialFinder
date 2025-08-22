#!/usr/bin/env python3
"""
Test script to demonstrate the new weighted metrics functionality.

This script shows how the weighted metrics handle different types of misclassifications
with varying severity based on relevance scores (0, 1, 2).
"""

import sys
from pathlib import Path
from typing import List, Dict

# Add parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.benchmark_filtering_performance import TrialEvaluationResult

def create_mock_trial_results():
    """Create mock trial evaluation results for testing."""

    # Mock trial results with different scenarios
    mock_results = [
        # Correct predictions
        TrialEvaluationResult(
            trial_id="trial_001",
            trial_title="Test Trial 1",
            predicted_eligible=True,
            ground_truth_relevant=True,
            suitability_probability=0.9,
            reason="Correctly identified as relevant",
            api_cost=0.01,
            original_relevance_score=2  # High relevance
        ),
        TrialEvaluationResult(
            trial_id="trial_002",
            trial_title="Test Trial 2",
            predicted_eligible=True,
            ground_truth_relevant=True,
            suitability_probability=0.8,
            reason="Correctly identified as relevant",
            api_cost=0.01,
            original_relevance_score=1  # Medium relevance
        ),
        TrialEvaluationResult(
            trial_id="trial_003",
            trial_title="Test Trial 3",
            predicted_eligible=False,
            ground_truth_relevant=False,
            suitability_probability=0.1,
            reason="Correctly identified as not relevant",
            api_cost=0.01,
            original_relevance_score=0  # Not relevant
        ),

        # False negatives with different severity
        TrialEvaluationResult(
            trial_id="trial_004",
            trial_title="Test Trial 4",
            predicted_eligible=False,
            ground_truth_relevant=True,
            suitability_probability=0.3,
            reason="Failed to identify high-relevance trial",
            api_cost=0.01,
            original_relevance_score=2  # High relevance - most severe error
        ),
        TrialEvaluationResult(
            trial_id="trial_005",
            trial_title="Test Trial 5",
            predicted_eligible=False,
            ground_truth_relevant=True,
            suitability_probability=0.4,
            reason="Failed to identify medium-relevance trial",
            api_cost=0.01,
            original_relevance_score=1  # Medium relevance - less severe error
        ),

        # False positive
        TrialEvaluationResult(
            trial_id="trial_006",
            trial_title="Test Trial 6",
            predicted_eligible=True,
            ground_truth_relevant=False,
            suitability_probability=0.7,
            reason="Incorrectly identified as relevant",
            api_cost=0.01,
            original_relevance_score=0  # Not relevant
        ),
    ]

    return mock_results

def test_weighted_metrics():
    """Test the weighted metrics calculation."""

    print("Testing Weighted Metrics for Clinical Trial Filtering")
    print("=" * 60)

    # Create mock trial results
    trial_results = create_mock_trial_results()

    print(f"Created {len(trial_results)} mock trial evaluation results:")
    for i, result in enumerate(trial_results, 1):
        relevance = result.original_relevance_score
        predicted = "Eligible" if result.predicted_eligible else "Not Eligible"
        actual = "Relevant" if result.ground_truth_relevant else "Not Relevant"
        error_type = "ERROR" if result.predicted_eligible != result.ground_truth_relevant else "CORRECT"

        print(f"  {i}. Trial {result.trial_id}: Predicted={predicted}, Actual={actual}, "
              f"Relevance Score={relevance}, Status={error_type}")

    print("\n" + "=" * 60)

    # Create a benchmark instance to use its weighted metrics calculation
    # Note: This is a simplified version for testing
    class MockBenchmark:
        def calculate_weighted_metrics(self, trial_results: List[TrialEvaluationResult]) -> Dict[str, float]:
            """Mock implementation of weighted metrics calculation."""
            if not trial_results:
                return {}

            # Initialize counters for weighted metrics
            weighted_tp: float = 0.0
            weighted_fp: float = 0.0
            weighted_fn: float = 0.0

            # For graded metrics
            squared_errors: List[float] = []
            absolute_errors: List[float] = []

            # For cost-sensitive metrics
            total_cost: float = 0.0
            max_possible_cost: float = 0.0

            for result in trial_results:
                if result.original_relevance_score is None:
                    continue

                actual_score: int = result.original_relevance_score
                predicted_score: int = 1 if result.predicted_eligible else 0

                # Calculate error magnitude
                error: int = abs(predicted_score - actual_score)
                absolute_errors.append(float(error))
                squared_errors.append(float(error ** 2))

                # Weighted metrics: higher penalty for misclassifying high-relevance trials
                if result.predicted_eligible and result.ground_truth_relevant:
                    # True Positive: weight based on actual relevance
                    weight = actual_score if actual_score > 0 else 1.0
                    weighted_tp += weight
                elif result.predicted_eligible and not result.ground_truth_relevant:
                    # False Positive: standard weight
                    weighted_fp += 1.0
                elif not result.predicted_eligible and result.ground_truth_relevant:
                    # False Negative: weight based on actual relevance (higher penalty for score 2)
                    weight = actual_score if actual_score > 0 else 1.0
                    weighted_fn += weight

                # Cost-sensitive metrics
                if actual_score == 0:
                    if predicted_score == 0:  # Correct
                        cost = 0
                    else:  # False positive
                        cost = 1
                elif actual_score == 1:
                    if predicted_score == 0:  # False negative on score 1
                        cost = 1
                    elif predicted_score == 1:  # Correct
                        cost = 0
                    else:  # Over-prediction
                        cost = 1
                else:  # actual_score == 2
                    if predicted_score == 0:  # False negative on score 2 (most severe)
                        cost = 2
                    elif predicted_score == 1:  # Under-prediction
                        cost = 1
                    else:  # Correct
                        cost = 0

                total_cost += cost
                max_possible_cost += 2  # Maximum cost per trial

            # Calculate weighted metrics
            weighted_precision = weighted_tp / (weighted_tp + weighted_fp) if (weighted_tp + weighted_fp) > 0 else 0.0
            weighted_recall = weighted_tp / (weighted_tp + weighted_fn) if (weighted_tp + weighted_fn) > 0 else 0.0
            weighted_f1 = 2 * (weighted_precision * weighted_recall) / (weighted_precision + weighted_recall) if (weighted_precision + weighted_recall) > 0 else 0.0

            # Calculate weighted accuracy (proportion of correctly classified weighted trials)
            total_weighted_trials = weighted_tp + weighted_fp + weighted_fn
            weighted_accuracy = weighted_tp / total_weighted_trials if total_weighted_trials > 0 else 0.0

            # Calculate graded metrics
            mean_absolute_error = sum(absolute_errors) / len(absolute_errors) if absolute_errors else 0.0
            root_mean_square_error = (sum(squared_errors) / len(squared_errors)) ** 0.5 if squared_errors else 0.0

            # Calculate cost-sensitive accuracy
            cost_sensitive_accuracy = 1.0 - (total_cost / max_possible_cost) if max_possible_cost > 0 else 0.0

            return {
                'weighted_precision': weighted_precision,
                'weighted_recall': weighted_recall,
                'weighted_f1': weighted_f1,
                'weighted_accuracy': weighted_accuracy,
                'mean_absolute_error': mean_absolute_error,
                'root_mean_square_error': root_mean_square_error,
                'cost_sensitive_accuracy': cost_sensitive_accuracy,
                'total_cost': total_cost,
                'max_possible_cost': max_possible_cost
            }

    # Calculate weighted metrics
    benchmark = MockBenchmark()
    weighted_metrics = benchmark.calculate_weighted_metrics(trial_results)

    print("WEIGHTED METRICS RESULTS:")
    print("=" * 60)
    print(f"Weighted Precision: {weighted_metrics['weighted_precision']:.4f}")
    print(f"Weighted Recall: {weighted_metrics['weighted_recall']:.4f}")
    print(f"Weighted F1-Score: {weighted_metrics['weighted_f1']:.4f}")
    print(f"Weighted Accuracy: {weighted_metrics['weighted_accuracy']:.4f}")
    print(f"Mean Absolute Error: {weighted_metrics['mean_absolute_error']:.4f}")
    print(f"Root Mean Square Error: {weighted_metrics['root_mean_square_error']:.4f}")
    print(f"Cost-Sensitive Accuracy: {weighted_metrics['cost_sensitive_accuracy']:.4f}")
    print(f"Total Cost: {weighted_metrics['total_cost']:.1f}")
    print(f"Maximum Possible Cost: {weighted_metrics['max_possible_cost']:.1f}")

    print("\n" + "=" * 60)
    print("INTERPRETATION:")
    print("=" * 60)
    print("• Weighted metrics assign higher penalties for misclassifying high-relevance trials (score 2)")
    print("• False negative on score 2 trial costs 2 points vs 1 point for score 1 trial")
    print("• Mean Absolute Error shows average prediction error magnitude")
    print("• Cost-sensitive accuracy considers the severity of different error types")
    print("• Lower MAE and higher cost-sensitive accuracy indicate better performance")

if __name__ == "__main__":
    test_weighted_metrics()
