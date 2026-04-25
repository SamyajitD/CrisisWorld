"""Atomic trace I/O — save and load EpisodeTrace as JSON."""

from __future__ import annotations

from pathlib import Path

from schemas.episode import EpisodeTrace


def save_trace(trace: EpisodeTrace, path: Path) -> Path:
    """Write trace as JSON. Atomic via tmp+replace. Creates parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    try:
        tmp.write_text(trace.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return path


def load_trace(path: Path) -> EpisodeTrace:
    """Read and validate a JSON trace file."""
    raw = Path(path).read_text(encoding="utf-8")
    return EpisodeTrace.model_validate_json(raw)
