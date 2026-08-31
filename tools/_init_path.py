import sys
from pathlib import Path


def add_repo_root_to_path():
    repo_root = str(Path(__file__).resolve().parents[1])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
