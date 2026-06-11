import sys
from pathlib import Path

# Make the repo root importable regardless of how pytest is invoked
# (CI runs `pytest tests/`, which does not add the CWD to sys.path).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
