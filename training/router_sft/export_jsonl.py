"""Convert raw router rollouts into prompt/completion JSONL files."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from .labels import normalize_sft_target
from .prompts import ROUTER_SYSTEM_PROMPT, build_router_user_prompt


def export_router_sft(
    raw_path: Path,
    output_dir: Path,
    *,
    val_fraction: float,
    split_seed: int,
) -> dict[str, Any]:
    """Read raw examples, split by episode, and export train/val JSONL."""
    rows = _load_jsonl(raw_path)
    episodes = _group_by_episode(rows)
    train_rows, val_rows = _split_episodes(
        episodes,
        val_fraction=val_fraction,
        split_seed=split_seed,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "router_sft_train.jsonl"
    val_path = output_dir / "router_sft_val.jsonl"
    cls_train_path = output_dir / "router_cls_train.jsonl"
    cls_val_path = output_dir / "router_cls_val.jsonl"

    _write_sft_jsonl(train_path, train_rows)
    _write_sft_jsonl(val_path, val_rows)
    _write_cls_jsonl(cls_train_path, train_rows)
    _write_cls_jsonl(cls_val_path, val_rows)

    return {
        "raw_rows": len(rows),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "train_episodes": len({row["episode_id"] for row in train_rows}),
        "val_episodes": len({row["episode_id"] for row in val_rows}),
        "output_dir": str(output_dir),
    }


def build_sft_record(row: dict[str, Any]) -> dict[str, Any]:
    """Convert one raw router row into a chat-style SFT example."""
    payload = row["input_payload"]
    target = normalize_sft_target(row["teacher_decision"])
    assistant = json.dumps(target, sort_keys=True)
    return {
        "messages": [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": build_router_user_prompt(payload)},
            {"role": "assistant", "content": assistant},
        ],
        "metadata": {
            "episode_id": row["episode_id"],
            "seed": row["seed"],
            "outer_turn": row["outer_turn"],
            "inner_iteration": row["inner_iteration"],
            "route_label": row["route_label"],
            "return_to_go": row["return_to_go"],
            "episode_total_reward": row["episode_total_reward"],
            "termination_reason": row["termination_reason"],
        },
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _group_by_episode(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["episode_id"]), []).append(row)
    return grouped


def _split_episodes(
    grouped: dict[str, list[dict[str, Any]]],
    *,
    val_fraction: float,
    split_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    episode_ids = sorted(grouped)
    random.Random(split_seed).shuffle(episode_ids)

    val_count = int(round(len(episode_ids) * val_fraction))
    val_ids = set(episode_ids[:val_count])

    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    for episode_id in episode_ids:
        target = val_rows if episode_id in val_ids else train_rows
        target.extend(grouped[episode_id])
    return train_rows, val_rows


def _write_sft_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(build_sft_record(row), sort_keys=True) + "\n")


def _write_cls_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            record = {
                "text": build_router_user_prompt(row["input_payload"]),
                "label": row["route_label"],
                "metadata": {
                    "episode_id": row["episode_id"],
                    "outer_turn": row["outer_turn"],
                    "inner_iteration": row["inner_iteration"],
                },
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True, help="Raw rollout JSONL")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=7)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    summary = export_router_sft(
        raw_path=args.raw,
        output_dir=args.output_dir,
        val_fraction=args.val_fraction,
        split_seed=args.split_seed,
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
