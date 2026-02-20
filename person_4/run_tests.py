#!/usr/bin/env python3
"""
Test runner script
Run with: python run_tests.py
"""

import sys
import pytest

if __name__ == "__main__":
    print("=" * 70)
    print("Content Integrity Platform - Running Tests")
    print("=" * 70)
    print()
    
    # Run pytest with verbose output
    exit_code = pytest.main([
        "-v",                    # Verbose
        "--tb=short",           # Short traceback format
        "--color=yes",          # Colored output
        "tests/",               # Test directory
    ])
    
    print()
    print("=" * 70)
    if exit_code == 0:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed. Check output above.")
    print("=" * 70)
    
    sys.exit(exit_code)
