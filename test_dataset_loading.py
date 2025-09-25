#!/usr/bin/env python3
"""
Test script to verify dataset loading functionality works correctly.
This script tests the fixed dataset loading with the NonMatchingSplitsSizesError handling.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mca_ai.config import load_config
from mca_ai.data_loader import load_dataset_any

def test_dataset_loading():
    """Test the dataset loading functionality."""
    print("Testing dataset loading functionality...")
    
    try:
        # Load configuration
        config_path = "configs/default.yaml"
        cfg = load_config(config_path)
        print(f"✓ Configuration loaded from {config_path}")
        
        # Test dataset loading
        print("Loading dataset...")
        ds = load_dataset_any(cfg)
        print(f"✓ Dataset loaded successfully!")
        
        # Print dataset info
        print(f"Dataset splits: {list(ds.keys())}")
        for split_name, split_data in ds.items():
            print(f"  {split_name}: {len(split_data)} examples")
            if len(split_data) > 0:
                print(f"    Sample text: {split_data[0]['text'][:100]}...")
        
        print("\n✓ All tests passed! Dataset loading is working correctly.")
        return True
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_dataset_loading()
    sys.exit(0 if success else 1)
