#!/usr/bin/env python3
"""
Quick test script for the 10 false negative patient-trial pairs to verify prompt improvements.
Tests specific pairs directly for maximum efficiency.
"""

import subprocess
import sys
from pathlib import Path

# The 10 false negative patient-trial pairs
FN_PAIRS = [
    ("trec-202112", "NCT02924363"),  # Mitral valve - "lost to follow-up"
    ("trec-202117", "NCT00688168"),  # Multiple myeloma - auto-HSCT
    ("trec-202124", "NCT01815697"),  # BPH - WBC 135K
    ("trec-202145", "NCT01123369"),  # Sickle cell - recent crisis
    ("trec-202151", "NCT04686318"),  # Pyelonephritis - pregnant
    ("trec-202152", "NCT02823899"),  # Cholera - active diarrhea
    ("trec-20216", "NCT01570634"),   # C. diff - required pressors
    ("trec-202160", "NCT02505919"),  # BPH - urinary retention
    ("trec-202165", "NCT01184703"),  # Cardiomyopathy - cardiac meds
    ("trec-20217", "NCT01882855"),   # Hepatic enceph - recent GI bleed
]

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_fn_pairs.py <OPENAI_API_KEY>")
        sys.exit(1)

    api_key = sys.argv[1]

    print("=" * 80)
    print("Testing 10 False Negative Patient-Trial Pairs with Improved Prompt")
    print("=" * 80)
    print()

    results = {}

    for i, (patient_id, trial_id) in enumerate(FN_PAIRS, 1):
        print(f"\n[{i}/10] Testing {patient_id} - {trial_id}")
        print("-" * 80)

        # Run benchmark for this specific patient-trial pair
        cmd = [
            "python", "-m", "benchmark.benchmark_filtering_performance",
            "--api-key", api_key,
            "--patient-id", patient_id,
            "--trial-id", trial_id,
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,  # 2 minute timeout per pair
            )

            if result.returncode == 0:
                # Parse output to check if the trial was marked as eligible
                output = result.stdout
                if trial_id in output:
                    # Look for eligibility indicators in the output
                    if '"predicted_eligible": true' in output or 'eligible=True' in output:
                        results[patient_id] = "✅ FIXED"
                        print(f"✅ {patient_id}: Trial {trial_id} now marked as eligible")
                    elif '"predicted_eligible": false' in output or 'eligible=False' in output:
                        results[patient_id] = "❌ STILL FN"
                        print(f"❌ {patient_id}: Trial {trial_id} still marked as ineligible")
                    else:
                        results[patient_id] = "⚠️ UNCLEAR"
                        print(f"⚠️ {patient_id}: Could not determine eligibility from output")
                else:
                    results[patient_id] = "⚠️ NOT FOUND"
                    print(f"⚠️ {patient_id}: Trial {trial_id} not in results")
            else:
                results[patient_id] = "❌ ERROR"
                print(f"❌ {patient_id}: Benchmark failed with error")
                print(f"Error: {result.stderr[:200]}")

        except subprocess.TimeoutExpired:
            results[patient_id] = "⏱️ TIMEOUT"
            print(f"⏱️ {patient_id}: Benchmark timed out")
        except Exception as e:
            results[patient_id] = "❌ ERROR"
            print(f"❌ {patient_id}: Exception - {str(e)[:100]}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    fixed = sum(1 for r in results.values() if r == "✅ FIXED")
    still_fn = sum(1 for r in results.values() if r == "❌ STILL FN")
    errors = sum(1 for r in results.values() if "ERROR" in r or "TIMEOUT" in r or "UNCLEAR" in r)

    for patient_id, trial_id in FN_PAIRS:
        status = results.get(patient_id, "⚠️ NOT RUN")
        print(f"{status} {patient_id} - {trial_id}")

    print()
    print(f"Fixed: {fixed}/10")
    print(f"Still FN: {still_fn}/10")
    print(f"Errors/Timeouts/Unclear: {errors}/10")
    print()

    if fixed >= 6:
        print("🎉 SUCCESS: Fixed at least 6 false negatives!")
    elif fixed >= 4:
        print("✅ GOOD: Fixed 4-5 false negatives, within expected range")
    else:
        print("⚠️ NEEDS WORK: Fixed fewer than expected false negatives")

if __name__ == "__main__":
    main()
