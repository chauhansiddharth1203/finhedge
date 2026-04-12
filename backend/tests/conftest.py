"""conftest.py – Shared pytest configuration."""
import sys
from pathlib import Path

# Ensure the project root is on sys.path for all tests
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
