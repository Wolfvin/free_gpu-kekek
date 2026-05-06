#!/usr/bin/env python3.13
"""Free GPU Trainer — Quick launcher."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tui import run_app

if __name__ == "__main__":
    run_app()
