"""Import helpers for running training scripts from the repo checkout."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def ensure_crisisworld_package() -> None:
    """Register the repository root as the ``CrisisWorld`` package if needed."""
    if "CrisisWorld" in sys.modules:
        return

    repo_root = Path(__file__).resolve().parents[2]
    init_py = repo_root / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        "CrisisWorld",
        init_py,
        submodule_search_locations=[str(repo_root)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to bootstrap CrisisWorld from {init_py}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["CrisisWorld"] = module
    spec.loader.exec_module(module)
