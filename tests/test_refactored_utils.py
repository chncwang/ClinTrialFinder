#!/usr/bin/env python3
"""
Tests for refactored utility functions in base/utils.py
"""
import os
import sys
import tempfile
import json
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from base.utils import get_api_key, create_gpt_client, load_json_list_file, save_json_list_file


def test_get_api_key_from_arg():
    """Test get_api_key with explicit argument"""
    api_key = get_api_key("test-key-123")
    assert api_key == "test-key-123", "API key from argument should be returned"
    print("✓ test_get_api_key_from_arg passed")


def test_get_api_key_from_env():
    """Test get_api_key from environment variable"""
    # Save existing value
    existing_key = os.environ.get("TEST_API_KEY")
    
    try:
        # Set test environment variable
        os.environ["TEST_API_KEY"] = "env-key-456"
        api_key = get_api_key(None, "TEST_API_KEY")
        assert api_key == "env-key-456", "API key from environment should be returned"
        print("✓ test_get_api_key_from_env passed")
    finally:
        # Restore original value
        if existing_key:
            os.environ["TEST_API_KEY"] = existing_key
        elif "TEST_API_KEY" in os.environ:
            del os.environ["TEST_API_KEY"]


def test_get_api_key_missing():
    """Test get_api_key raises error when key is missing"""
    try:
        get_api_key(None, "NONEXISTENT_KEY")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "not found" in str(e).lower(), "Error message should mention key not found"
        print("✓ test_get_api_key_missing passed")


def test_create_gpt_client():
    """Test create_gpt_client factory function"""
    # This test just verifies the function creates a client without errors
    # We're using a fake key since we're not actually calling the API
    client = create_gpt_client(
        api_key="test-key",
        cache_size=1000,
        temperature=0.5,
        max_retries=2
    )
    assert client is not None, "GPT client should be created"
    print("✓ test_create_gpt_client passed")


def test_load_save_json_list_file():
    """Test load_json_list_file and save_json_list_file"""
    test_data = [
        {"id": 1, "name": "test1"},
        {"id": 2, "name": "test2"}
    ]
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name
    
    try:
        # Test save
        save_json_list_file(test_data, temp_path, "test data")
        
        # Test load
        loaded_data = load_json_list_file(temp_path)
        assert loaded_data == test_data, "Loaded data should match saved data"
        print("✓ test_load_save_json_list_file passed")
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def main():
    """Run all tests"""
    print("Running utility function tests...")
    print()
    
    test_get_api_key_from_arg()
    test_get_api_key_from_env()
    test_get_api_key_missing()
    test_create_gpt_client()
    test_load_save_json_list_file()
    
    print()
    print("All tests passed! ✓")


if __name__ == "__main__":
    main()
