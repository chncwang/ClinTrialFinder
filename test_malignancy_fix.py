#!/usr/bin/env python3
"""
Test script to verify the malignancy criterion interpretation fix.

This script tests NCT06336902 (Botensilimab + Balstilimab) against the Oct 18 patient
to ensure it's no longer incorrectly excluded.
"""

from base.gpt_client import GPTClient
from base.trial_expert import GPTTrialFilter
from base.clinical_trial import ClinicalTrial, Identification, Eligibility

# Patient conditions from Oct 18 search
PATIENT_CONDITIONS = [
    '56-year-old female',
    'Colorectal cancer with liver and para-aortic lymph node metastases',
    'Hepatectomy performed 3 months ago to remove liver lesions and para-aortic lymph nodes',
    'Residual liver metastases with tumor residual rates of 10% and 20% in different liver segments post-surgery',
    'Para-aortic lymph node metastasis (1/2 nodes positive), no capsule involvement',
    'No hepatic hilar lymph node metastasis (0/1 nodes)',
    'No vascular tumor thrombus observed',
    'No cancer at liver tissue margins post-resection',
    'KRAS p.G13R mutation',
    'TP53 p.R175H and p.C238Y mutations',
    'Reduced copy number of STK11',
    'SMAD4 p.E337* mutation',
    'APC p.G1412* mutation',
    'KMT2A c.6320-1G>A mutation',
    'Increased copy number of FGF23, CCND2, and RAD52',
    'TMB 5.03 muts/Mb (top 47% among colorectal cancers)',
    'Immunohistochemistry: KI67 80% (high proliferation index)',
    'EGFR 1+',
    'MLH1, MSH2, MSH6, PMS2 positive (proficient mismatch repair)',
    'PD-L1 (22C3) CPS = 20 (high expression)',
    'BRAF and HER2 negative'
]

# NCT06336902 trial information
TRIAL_TITLE = "Botensilimab Plus Balstilimab and Fasting Mimicking Diet Plus Vitamin C for Patients with KRAS-Mutant Metastatic Colorectal Cancer"

INCLUSION_CRITERIA = """Inclusion Criteria:

* Histologically or cytologically confirmed microsatellite stable (MSS) metastatic colorectal adenocarcinoma with any KRAS mutation (as determined by a Clinical Laboratory Improvement Act [CLIA]-certified lab), including metastases to liver, lung, etc.
* Disease progression, intolerance or contraindication to a fluoropyrimidine, oxaliplatin, irinotecan
* ≥ 18 years of age
* Performance status Eastern Cooperative Oncology Group (ECOG) 0-1
* Estimated life expectancy ≥ 3 months
* Body mass index (BMI) ≥ 18.5
* Absolute neutrophil count ≥ 1,500/mcL
* Hemoglobin ≥ 8.0 g/dL
* Platelets ≥ 75,000/mcL
* Total bilirubin ≤ 1.5 x upper limit of normal (ULN) (for patients with Gilbert syndrome ≤ 3.0 x ULN)
* Aspartate aminotransferase (AST) / alanine aminotransferase (ALT) ≤ 3 x ULN
* Creatinine ≤ 1.5 x ULN
* Measurable disease as defined by RECIST 1.1
* No history of prior or current malignancy that requires active treatment

Exclusion Criteria:

* Patients with a current diagnosis of diabetes mellitus are not eligible for this study."""


def create_mock_trial():
    """Create a mock trial object for NCT06336902."""
    identification = Identification(
        nct_id="NCT06336902",
        url="https://clinicaltrials.gov/study/NCT06336902",
        brief_title=TRIAL_TITLE,
        official_title=TRIAL_TITLE
    )

    eligibility = Eligibility(
        criteria=INCLUSION_CRITERIA,
        gender="All",
        min_age="18 Years",
        max_age=None,
        healthy_volunteers="No"
    )

    trial = ClinicalTrial(
        identification=identification,
        eligibility=eligibility
    )

    return trial


def main():
    """Run the test."""
    print("=" * 80)
    print("Testing Malignancy Criterion Interpretation Fix")
    print("=" * 80)
    print(f"\nTrial: {TRIAL_TITLE}")
    print(f"NCT ID: NCT06336902")
    print(f"\nPatient: 56F with KRAS-mutant MSS metastatic colorectal cancer")
    print(f"Number of conditions: {len(PATIENT_CONDITIONS)}")
    print("\n" + "=" * 80)

    # Initialize GPT client and filter
    gpt_client = GPTClient()
    trial_filter = GPTTrialFilter(gpt_client)

    # Create mock trial
    trial = create_mock_trial()

    # Evaluate trial
    print("\nEvaluating trial eligibility...")
    is_eligible, cost, failure_reason = trial_filter.evaluate_trial(
        trial, PATIENT_CONDITIONS, refresh_cache=True
    )

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"\nEligible: {is_eligible}")
    print(f"API Cost: ${cost:.4f}")

    if failure_reason:
        print(f"\nFailure Type: {failure_reason.type}")
        print(f"Failure Message: {failure_reason.message}")
        if failure_reason.failed_condition:
            print(f"Failed Condition: {failure_reason.failed_condition}")
        if failure_reason.failed_criterion:
            print(f"Failed Criterion: {failure_reason.failed_criterion}")
        if failure_reason.failure_details:
            print(f"Failure Details: {failure_reason.failure_details}")

    print("\n" + "=" * 80)
    print("TEST RESULT")
    print("=" * 80)

    if is_eligible:
        print("\n✅ TEST PASSED: Trial is now correctly included!")
        print("The malignancy criterion fix is working as expected.")
        return 0
    else:
        print("\n❌ TEST FAILED: Trial is still excluded!")
        print("The fix may need adjustment.")
        if failure_reason and "malignancy" in str(failure_reason).lower():
            print("\nThe trial is still being excluded due to malignancy criterion.")
            print("The prompt engineering fix may need to be stronger or more explicit.")
        return 1


if __name__ == "__main__":
    exit(main())
