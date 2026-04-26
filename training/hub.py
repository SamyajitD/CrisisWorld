"""Hub integration — push/pull datasets, adapters, and results to HuggingFace Hub."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)


def _get_token() -> str | None:
    """Get HF token from file or env var."""
    import os

    token_file = Path(__file__).resolve().parent.parent / "configs" / ".hf_token"
    if token_file.exists():
        return token_file.read_text(encoding="utf-8").strip()
    return os.environ.get("HF_TOKEN")


def push_dataset(local_path: Path, repo_id: str, filename: str = "data.jsonl") -> str:
    """Push a JSONL dataset file to a HuggingFace Hub dataset repo.

    Returns the URL of the uploaded file.
    """
    from huggingface_hub import HfApi

    token = _get_token()
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    url = api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=filename,
        repo_id=repo_id,
        repo_type="dataset",
    )
    _log.info("Pushed dataset to %s/%s", repo_id, filename)
    return url


def push_results(result_data: dict[str, Any], repo_id: str, filename: str = "results.json") -> str:
    """Push a results JSON to a HuggingFace Hub model/dataset repo."""
    from huggingface_hub import HfApi

    token = _get_token()
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)

    tmp = Path("/tmp") / filename
    tmp.write_text(json.dumps(result_data, indent=2, default=str), encoding="utf-8")
    url = api.upload_file(
        path_or_fileobj=str(tmp),
        path_in_repo=filename,
        repo_id=repo_id,
        repo_type="dataset",
    )
    _log.info("Pushed results to %s/%s", repo_id, filename)
    return url


def push_adapter(adapter_dir: Path, repo_id: str) -> str:
    """Push a PEFT adapter directory to a HuggingFace Hub model repo."""
    from huggingface_hub import HfApi

    token = _get_token()
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="model", exist_ok=True)
    url = api.upload_folder(
        folder_path=str(adapter_dir),
        repo_id=repo_id,
        repo_type="model",
    )
    _log.info("Pushed adapter to %s", repo_id)
    return url


def pull_dataset(repo_id: str, filename: str = "data.jsonl", local_dir: Path = Path("data")) -> Path:
    """Download a dataset file from HuggingFace Hub."""
    from huggingface_hub import hf_hub_download

    token = _get_token()
    local_dir.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        local_dir=str(local_dir),
        token=token,
    )
    _log.info("Pulled dataset from %s/%s to %s", repo_id, filename, path)
    return Path(path)
