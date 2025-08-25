#!/usr/bin/env python3
"""
Test script to demonstrate cost calculation functionality in the GPT client.
"""

import os
import sys
from pathlib import Path

# Add the base directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from base.gpt_client import GPTClient
from base.pricing import AITokenPricing

def test_cost_calculation():
    """Test the cost calculation functionality."""

    # Check if API key is available
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key to test the cost calculation")
        return

    try:
        # Initialize GPT client
        print("🔧 Initializing GPT client...")
        gpt_client = GPTClient(api_key=api_key)
        print("✅ GPT client initialized successfully")

        # Test prompts
        test_prompts = [
            {
                "name": "Short prompt",
                "system": "You are a helpful assistant.",
                "user": "Hello, how are you?",
                "model": "gpt-4o-mini"
            },
            {
                "name": "Medium prompt",
                "system": "You are a clinical trial expert. Analyze the following case.",
                "user": "Patient has diabetes and is 65 years old. Trial requires age 18-70 and diabetes diagnosis.",
                "model": "gpt-4o"
            },
            {
                "name": "Long prompt",
                "system": "You are a medical researcher specializing in oncology clinical trials. Please provide a detailed analysis.",
                "user": "Patient presents with stage III breast cancer, ER+, PR+, HER2-. Previous treatment: 4 cycles of AC chemotherapy. Current medications: tamoxifen, calcium supplements. Comorbidities: hypertension (controlled), type 2 diabetes (controlled). Lab values: WBC 4.2, Hgb 12.1, Plt 180, Creatinine 0.9, AST 25, ALT 28. Trial criteria: Stage II-III breast cancer, ER+ or PR+, HER2-, no prior chemotherapy, adequate organ function.",
                "model": "gpt-4.1-mini"
            }
        ]

        print("\n🧮 Testing cost calculations...")
        print("=" * 80)

        total_cost = 0.0

        for i, prompt_info in enumerate(test_prompts, 1):
            print(f"\n📝 Test {i}: {prompt_info['name']}")
            print(f"   Model: {prompt_info['model']}")
            print(f"   System prompt length: {len(prompt_info['system'])} chars")
            print(f"   User prompt length: {len(prompt_info['user'])} chars")

            try:
                # Make the API call
                response, cost = gpt_client.call_gpt(
                    prompt=prompt_info['user'],
                    system_role=prompt_info['system'],
                    model=prompt_info['model']
                )

                # Calculate expected cost manually
                expected_cost = AITokenPricing.calculate_cost(
                    prompt_info['system'] + prompt_info['user'],
                    response,
                    prompt_info['model']
                )

                print(f"   Response length: {len(response)} chars")
                print(f"   Actual cost: ${cost:.6f}")
                print(f"   Expected cost: ${expected_cost:.6f}")
                print(f"   Cost match: {'✅' if abs(cost - expected_cost) < 1e-6 else '❌'}")

                total_cost += cost

            except Exception as e:
                print(f"   ❌ Error: {e}")

        print("\n" + "=" * 80)
        print(f"💰 Total cost for all tests: ${total_cost:.6f}")

        # Test pricing utility directly
        print("\n🔍 Testing pricing utility directly...")
        test_text = "This is a test text for token estimation."
        estimated_tokens = AITokenPricing.estimate_tokens(test_text)
        print(f"Text: '{test_text}'")
        print(f"Estimated tokens: {estimated_tokens:.1f}")
        print(f"Character to token ratio: {AITokenPricing.CHAR_TO_TOKEN_RATIO}")

        # Show available models and their costs
        print("\n📊 Available models and costs (per 1K tokens):")
        for model, (input_cost, output_cost) in AITokenPricing.MODEL_COSTS.items():
            print(f"   {model}:")
            print(f"     Input: ${input_cost:.6f}")
            print(f"     Output: ${output_cost:.6f}")

        print("\n✅ Cost calculation test completed successfully!")

    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_cost_calculation()
