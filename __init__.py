"""CrisisWorld — outbreak-control environment for budgeted deliberation."""

from pathlib import Path

from .client import CrisisWorldClient
from .models import ActionUnion, CrisisState, Observation


def get_package_root() -> Path:
    """Return the root directory of the installed CrisisWorld package."""
    return Path(__file__).resolve().parent


__all__ = ["ActionUnion", "CrisisState", "CrisisWorldClient", "Observation", "get_package_root"]
