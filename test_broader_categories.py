#!/usr/bin/env python3
"""Quick test to verify broader disease category extraction works correctly."""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from base.gpt_client import GPTClient
from base.disease_expert import extract_disease_from_record, get_parent_disease_categories

def test_broader_categories():
    """Test that all broader categories are returned."""

    # Get API key from environment
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    # Initialize GPT client
    gpt_client = GPTClient(api_key=api_key)

    # Read clinical record
    clinical_record_path = "/Users/chncwang/Projects/NPC-record/clinical-record-20251129.txt"
    with open(clinical_record_path, 'r') as f:
        clinical_record = f.read()

    print("="*80)
    print("TESTING BROADER DISEASE CATEGORY EXTRACTION")
    print("="*80)
    print()

    # Test 1: Extract disease (without broader categories)
    print("Test 1: Standard disease extraction (avoid_specific_disease=False)")
    print("-" * 80)
    disease, cost1 = extract_disease_from_record(clinical_record, gpt_client, avoid_specific_disease=False)
    print(f"✅ Extracted disease: {disease}")
    print(f"   Type: {type(disease)}")
    print(f"   Cost: ${cost1:.4f}")
    print()

    # Test 2: Get broader categories directly
    print("Test 2: Get parent disease categories")
    print("-" * 80)
    categories, cost2 = get_parent_disease_categories(disease, gpt_client)
    print(f"✅ Broader categories: {categories}")
    print(f"   Count: {len(categories)}")
    print(f"   Type: {type(categories)}")
    print(f"   Cost: ${cost2:.4f}")
    print()

    # Test 3: Extract disease WITH broader categories (NEW BEHAVIOR)
    print("Test 3: Extract disease with broader categories (avoid_specific_disease=True)")
    print("-" * 80)
    result, cost3 = extract_disease_from_record(clinical_record, gpt_client, avoid_specific_disease=True)
    print(f"✅ Result: {result}")
    print(f"   Type: {type(result)}")
    print(f"   Count: {len(result) if isinstance(result, list) else 'N/A'}")
    print(f"   Cost: ${cost3:.4f}")
    print()

    # Verify the fix
    print("="*80)
    print("VERIFICATION")
    print("="*80)

    if isinstance(result, list):
        print(f"✅ PASS: Returns list (not single string)")
        print(f"✅ PASS: Contains {len(result)} categories")

        if len(result) >= 2:
            print(f"✅ PASS: Multiple categories returned (not just first one)")
        else:
            print(f"❌ FAIL: Only {len(result)} category returned")

        print()
        print("Categories that will be searched:")
        for i, category in enumerate(result, 1):
            print(f"  {i}. {category}")

        # Check for expected categories
        print()
        expected_in_results = ["head and neck", "solid tumor"]
        for expected in expected_in_results:
            found = any(expected in cat.lower() for cat in result)
            status = "✅" if found else "❌"
            print(f"{status} Expected category '{expected}': {'FOUND' if found else 'NOT FOUND'}")

    else:
        print(f"❌ FAIL: Returns {type(result)} instead of list")

    print()
    print("="*80)
    print(f"TOTAL API COST: ${cost1 + cost2 + cost3:.4f}")
    print("="*80)

    return result

if __name__ == "__main__":
    try:
        result = test_broader_categories()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
