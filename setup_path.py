import os
import sys


def add_project_root_to_path() -> None:
    """Add the project root (parent of this file) to PYTHONPATH if missing."""
    root = os.path.dirname(os.path.abspath(__file__))
    if root not in sys.path:
        sys.path.insert(0, root)

