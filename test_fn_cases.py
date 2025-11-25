#!/usr/bin/env python3
"""
Quick test script for the 10 false negative cases to verify prompt improvements.
"""

import subprocess
import sys
from pathlib import Path

# The 10 false negative patient IDs
FN_PATIENT_IDS = [
    "trec-202112",  # Mitral valve - "lost to follow-up"
    "trec-202117",  # Multiple myeloma - auto-HSCT
    "trec-202124",  # BPH - WBC 135K
    "trec-202145",  # Sickle cell - recent crisis
    "trec-202151",  # Pyelonephritis - pregnant
    "trec-202152",  # Cholera - active diarrhea
    "trec-20216",   # C. diff - required pressors
    "trec-202160",  # BPH - urinary retention
    "trec-202165",  # Cardiomyopathy - cardiac meds
    "trec-20217",   # Hepatic enceph - recent GI bleed
]

# The specific trial IDs for each patient (for reference)
FN_TRIAL_IDS = {
    "trec-202112": "NCT02924363",
    "trec-202117": "NCT00688168",
    "trec-202124": "NCT01815697",
    "trec-202145": "NCT01123369",
    "trec-202151": "NCT04686318",
    "trec-202152": "NCT02823899",
    "trec-20216": "NCT01570634",
    "trec-202160": "NCT02505919",
    "trec-202165": "NCT01184703",
    "trec-20217": "NCT01882855",
}

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_fn_cases.py <OPENAI_API_KEY>")
        sys.exit(1)

    api_key = sys.argv[1]

    print("=" * 80)
    print("Testing 10 False Negative Cases with Improved Prompt")
    print("=" * 80)
    print()

    results = {}

    for i, patient_id in enumerate(FN_PATIENT_IDS, 1):
        trial_id = FN_TRIAL_IDS[patient_id]
        print(f"\n[{i}/10] Testing {patient_id} - {trial_id}")
        print("-" * 80)

        # Run benchmark for this specific patient
        cmd = [
            "python", "-m", "benchmark.benchmark_filtering_performance",
            "--api-key", api_key,
            "--patient-id", patient_id,
            "--max-trials", "500",  # Include enough trials
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout per patient
            )

            if result.returncode == 0:
                # Parse output to check if the specific trial was matched
                output = result.stdout
                if trial_id in output:
                    if "suitability_probability: 1.0" in output or "eligible" in output.lower():
                        results[patient_id] = "✅ FIXED"
                        print(f"✅ {patient_id}: Trial {trial_id} now marked as eligible")
                    else:
                        results[patient_id] = "❌ STILL FN"
                        print(f"❌ {patient_id}: Trial {trial_id} still marked as ineligible")
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
    errors = sum(1 for r in results.values() if "ERROR" in r or "TIMEOUT" in r)

    for patient_id, status in results.items():
        trial_id = FN_TRIAL_IDS[patient_id]
        print(f"{status} {patient_id} - {trial_id}")

    print()
    print(f"Fixed: {fixed}/10")
    print(f"Still FN: {still_fn}/10")
    print(f"Errors/Timeouts: {errors}/10")
    print()

    if fixed >= 6:
        print("🎉 SUCCESS: Fixed at least 6 false negatives!")
    elif fixed >= 4:
        print("✅ GOOD: Fixed 4-5 false negatives, within expected range")
    else:
        print("⚠️ NEEDS WORK: Fixed fewer than expected false negatives")

if __name__ == "__main__":
    main()
