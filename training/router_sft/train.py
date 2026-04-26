"""Standalone LoRA SFT entrypoint for the router dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _require_training_deps() -> dict[str, Any]:
    try:
        import datasets
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise SystemExit(
            "Missing training dependencies. Install "
            "training/router_sft/requirements.txt first."
        ) from exc

    return {
        "datasets": datasets,
        "torch": torch,
        "LoraConfig": LoraConfig,
        "get_peft_model": get_peft_model,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "DataCollatorForLanguageModeling": DataCollatorForLanguageModeling,
        "Trainer": Trainer,
        "TrainingArguments": TrainingArguments,
    }


def train_router_sft(args: argparse.Namespace) -> None:
    deps = _require_training_deps()
    datasets = deps["datasets"]
    torch = deps["torch"]
    LoraConfig = deps["LoraConfig"]
    get_peft_model = deps["get_peft_model"]
    AutoModelForCausalLM = deps["AutoModelForCausalLM"]
    AutoTokenizer = deps["AutoTokenizer"]
    DataCollatorForLanguageModeling = deps["DataCollatorForLanguageModeling"]
    Trainer = deps["Trainer"]
    TrainingArguments = deps["TrainingArguments"]

    dataset = datasets.load_dataset("json", data_files={"train": str(args.train_path)})
    eval_dataset = None
    if args.val_path is not None:
        eval_data = datasets.load_dataset(
            "json",
            data_files={"validation": str(args.val_path)},
        )
        eval_dataset = eval_data["validation"]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bf16_enabled = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    fp16_enabled = torch.cuda.is_available() and not bf16_enabled
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if bf16_enabled else torch.float32,
    )

    target_modules = _infer_target_modules(
        model,
        requested=[
            module.strip()
            for module in args.target_modules.split(",")
            if module.strip()
        ],
    )
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)

    def preprocess(record: dict[str, Any]) -> dict[str, Any]:
        text = _render_messages(record["messages"], tokenizer)
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=args.max_length,
        )
        tokenized["labels"] = list(tokenized["input_ids"])
        return tokenized

    train_dataset = dataset["train"].map(
        preprocess,
        remove_columns=dataset["train"].column_names,
    )
    tokenized_eval = None
    if eval_dataset is not None:
        tokenized_eval = eval_dataset.map(
            preprocess,
            remove_columns=eval_dataset.column_names,
        )

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        evaluation_strategy="epoch" if tokenized_eval is not None else "no",
        bf16=bf16_enabled,
        fp16=fp16_enabled,
        report_to=[],
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=tokenized_eval,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()
    trainer.save_model(str(args.output_dir / "adapter"))
    tokenizer.save_pretrained(str(args.output_dir / "adapter"))

    summary = {
        "model_name": args.model_name,
        "train_rows": len(train_dataset),
        "eval_rows": len(tokenized_eval) if tokenized_eval is not None else 0,
        "output_dir": str(args.output_dir),
    }
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _render_messages(messages: list[dict[str, str]], tokenizer: Any) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    rendered: list[str] = []
    for message in messages:
        rendered.append(f"{message['role'].upper()}: {message['content']}")
    return "\n\n".join(rendered)


def _infer_target_modules(model: Any, requested: list[str]) -> list[str]:
    module_names = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    target_modules = [name for name in requested if name in module_names]
    if not target_modules:
        raise SystemExit(
            "None of the requested LoRA target modules were found in the model. "
            "Pass --target-modules with names that exist in the chosen base model."
        )
    return target_modules


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--val-path", type=Path)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_router_sft(args)


if __name__ == "__main__":
    main()
