"""Ensures the repository root is importable regardless of how pytest is invoked
(`pytest`, `python -m pytest`, or from a different working directory), so that
`import tools` / `import api` resolve consistently across environments.
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
