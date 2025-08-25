#!/usr/bin/env python3
"""
Test script for the modified benchmark script with trial downloading capability.

This script tests whether the FilteringBenchmark class can properly load the dataset
and download trials when the trial data file is missing.
"""

import sys
from pathlib import Path

# Add parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.benchmark_filtering_performance import FilteringBenchmark
from base.gpt_client import GPTClient
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_benchmark_loading():
    """Test loading the benchmark dataset."""
    print("Testing benchmark dataset loading...")

    # Path to the TREC 2021 dataset
    dataset_path = "dataset/trec_2021"

    if not Path(dataset_path).exists():
        print(f"Error: Dataset path {dataset_path} does not exist")
        return False

    try:
        # Create a mock GPT client for testing
        gpt_client = GPTClient(api_key="test_key", cache_size=1000, strict_cache_mode=False)

        # Initialize the benchmark (this will trigger trial downloading if needed)
        print("Initializing FilteringBenchmark...")
        benchmark = FilteringBenchmark(dataset_path, gpt_client)

        print(f"✅ Successfully loaded benchmark dataset")
        print(f"   - Patients: {len(benchmark.patients)}")
        print(f"   - Relevance judgments: {len(benchmark.relevance_judgments)}")
        print(f"   - Trials: {len(benchmark.trials)}")

        # Check trial coverage
        missing_trials = benchmark.get_missing_trials()
        if missing_trials:
            print(f"   - Missing trials: {len(missing_trials)}")
            print(f"     First 5 missing: {missing_trials[:5]}")
        else:
            print(f"   - All required trials are available")

        # Test getting trial data for a few trials
        if benchmark.trials:
            sample_trial_id = list(benchmark.trials.keys())[0]
            trial_data = benchmark.trials[sample_trial_id]
            if trial_data:
                print(f"   - Sample trial {sample_trial_id}: {trial_data.identification.brief_title[:50] if trial_data.identification.brief_title else 'N/A'}...")
            else:
                print(f"   - Warning: Could not retrieve data for trial {sample_trial_id}")

        return True

    except Exception as e:
        print(f"❌ Error loading benchmark dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("=" * 60)
    print("Testing FilteringBenchmark with Trial Downloading")
    print("=" * 60)

    success = test_benchmark_loading()

    print("\n" + "=" * 60)
    if success:
        print("✅ All tests passed!")
    else:
        print("❌ Tests failed!")
    print("=" * 60)

    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
