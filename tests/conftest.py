# tests/conftest.py
# ruff: noqa: E402
# Use non-interactive backend for CI/headless environments (must be before pyplot import)
import matplotlib

matplotlib.use("Agg")

import sys
from pathlib import Path

# Add the project root to sys.path so tests can import custom_noise without installation
sys.path.insert(0, str(Path(__file__).parent.parent))
