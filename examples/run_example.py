#!/usr/bin/env python3
"""
Example: run the GoDec decomposition on a sample file.

Usage:
    python examples/run_example.py
    python examples/run_example.py path/to/your/file.tif
"""

import sys
from godec import DecomposeGoDec

if __name__ == "__main__":
    filename = sys.argv[1] if len(sys.argv) > 1 else "data/Escalator-1000f-8b.tif"
    DecomposeGoDec(filename)
