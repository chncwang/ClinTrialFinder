#!/usr/bin/env python3
"""
Simple test script to verify trial downloading functionality.

This script tests downloading just a few trials to verify the API integration works.
"""

import sys
import json
from pathlib import Path

# Add parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.benchmark_filtering_performance import FilteringBenchmark
from base.gpt_client import GPTClient
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_small_download():
    """Test downloading a small number of trials."""
    print("Testing small trial download...")

        # Create a minimal benchmark instance for testing
    class MinimalBenchmark(FilteringBenchmark):
        def __init__(self, dataset_path: str):
            # Create a mock GPT client for testing
            gpt_client = GPTClient(api_key="test_key", cache_size=1000, strict_cache_mode=False)

            # Call parent constructor
            super().__init__(dataset_path, gpt_client)

            # Override trials to start empty
            self.trials = {}

                        # Load just a few patients and relevance judgments for testing
            self._load_minimal_dataset()

        def _load_minimal_dataset(self):
            """Load minimal dataset for testing."""
            # Load just one patient
            queries_file = self.dataset_path / "queries.jsonl"
            self.patients = []
            with open(queries_file, 'r') as f:
                # Just load the first patient
                line = f.readline()
                if line:
                    patient_data = json.loads(line)
                    from scripts.benchmark_filtering_performance import Patient
                    self.patients.append(Patient.from_dict(patient_data))

            # Load just a few relevance judgments for the first patient
            qrels_file = self.dataset_path / "qrels" / "test.tsv"
            self.relevance_judgments = []
            with open(qrels_file, 'r') as f:
                first_patient_id = self.patients[0].patient_id
                count = 0
                for line in f:
                    if count >= 5:  # Just get 5 trials
                        break
                    if line.startswith(first_patient_id):
                        from scripts.benchmark_filtering_performance import RelevanceJudgment
                        judgment = RelevanceJudgment.from_tsv_line(line)
                        if judgment is not None:
                            self.relevance_judgments.append(judgment)
                            count += 1

            print(f"Loaded {len(self.patients)} patients and {len(self.relevance_judgments)} relevance judgments")

    try:
        # Initialize minimal benchmark
        benchmark = MinimalBenchmark("dataset/trec_2021")

        # Extract trial IDs
        trial_ids = set(judgment.trial_id for judgment in benchmark.relevance_judgments)
        print(f"Found {len(trial_ids)} trial IDs: {list(trial_ids)}")

        # Test downloading these trials
        print("Testing trial download...")
        benchmark.download_trials(trial_ids)

        print("✅ Download test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Error during download test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("=" * 60)
    print("Testing Small Trial Download")
    print("=" * 60)

    success = test_small_download()

    print("\n" + "=" * 60)
    if success:
        print("✅ Download test passed!")
    else:
        print("❌ Download test failed!")
    print("=" * 60)

    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
