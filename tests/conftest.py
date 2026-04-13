"""Shared pytest fixtures."""
import sys
from pathlib import Path

# Ensure the repo root is on the path even when running pytest from a
# subdirectory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
