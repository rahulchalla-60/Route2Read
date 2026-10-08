"""
Route2Read: Dynamic Path & Environment Configuration.
Resolves repository root dynamically across local, Colab, and cloud environments.
"""

import os
from pathlib import Path


def get_project_root() -> Path:
    """
    Dynamically resolves project root directory.
    Checks:
    1. ROUTE2READ_ROOT environment variable (if set).
    2. Google Drive path (/content/drive/MyDrive/Route2Read) if running in Colab and directory exists.
    3. Anchor file relative to this module (parent of parent of src/utils/paths.py).
    """
    env_root = os.environ.get("ROUTE2READ_ROOT")
    if env_root and os.path.exists(env_root):
        return Path(env_root).resolve()

    colab_drive_path = Path("/content/drive/MyDrive/Route2Read")
    if colab_drive_path.exists():
        return colab_drive_path.resolve()

    # Fallback to local repo root relative to this file: src/utils/paths.py -> root
    curr = Path(__file__).resolve()
    for parent in [curr.parent.parent.parent, curr.parent.parent]:
        if (parent / "README.md").exists() or (parent / ".git").exists():
            return parent

    return Path.cwd().resolve()


PROJECT_ROOT = get_project_root()
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
ROUTING_DIR = RESULTS_DIR / "routing_logs"
EVAL_DIR = RESULTS_DIR / "eval_metrics"
MODELS_DIR = PROJECT_ROOT / "models"


def ensure_dirs():
    """Ensures all standard project artifact directories exist."""
    for d in [DATA_DIR, RESULTS_DIR, FIGURES_DIR, TABLES_DIR, ROUTING_DIR, EVAL_DIR, MODELS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
