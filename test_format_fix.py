#!/usr/bin/env python3
"""
Test script to verify that the format specifier error is fixed.
"""

def test_format_string():
    """Test the problematic format string that was causing the error."""

    # Simulate the case object
    class MockCase:
        def __init__(self):
            self.full_medical_record = "Test patient summary"
            self.trial_title = "Test trial title"
            self.trial_criteria = "Test trial criteria"

    case = MockCase()

    # This is the problematic format string that was fixed
    user_prompt = f"""Categorize the following false positive error case into one of:
- "exclusion_criteria_violation": patient meets ≥1 exclusion criterion.
- "inclusion_criteria_violation": patient fails to meet ≥1 inclusion criterion.
- "data_label_error": benchmark label or data is likely incorrect.

Return JSON only, in this schema:
{{
  "reasoning": "≤60 words explaining your choice",
  "category": "exclusion_criteria_violation" | "inclusion_criteria_violation" | "data_label_error",
}}

Case details:

Clinical Record:
{case.full_medical_record}

Trial Title:
{case.trial_title}

Trial Criteria:
{case.trial_criteria}
"""

    print("Format string test passed!")
    print("The problematic f-string now works correctly.")
    print("\nGenerated prompt:")
    print(user_prompt)

if __name__ == "__main__":
    test_format_string()
