"""Gradio Space control plane for CrisisWorld + Cortex experiments.

Tabs: Configure | Launch | Browse | Compare
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import gradio as gr


# ---------------------------------------------------------------------------
# Tab 1: Configure — generate a run manifest
# ---------------------------------------------------------------------------

def generate_manifest(
    run_type: str,
    base_model: str,
    dataset_repo: str,
    num_epochs: int,
    lora_r: int,
    learning_rate: float,
    eval_turns: int,
    seeds: str,
) -> str:
    """Generate a run manifest JSON from the form inputs."""
    seed_list = [int(s.strip()) for s in seeds.split(",") if s.strip()]
    manifest = {
        "run_name": f"run-{uuid.uuid4().hex[:6]}",
        "run_type": run_type,
        "dataset_repo": dataset_repo,
        "base_models": {"default": base_model},
        "adapter_output_repo": "",
        "sft_config": {
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "lora_r": lora_r,
            "lora_alpha": lora_r * 2,
            "batch_size": 4,
            "max_seq_length": 2048,
        },
        "eval_config": {
            "max_turns": eval_turns,
            "eval_seeds": seed_list,
            "conditions": ["flat-fat", "cortex-full", "cortex-llm"],
        },
        "seeds": seed_list,
        "quantization": "4bit",
    }
    return json.dumps(manifest, indent=2)


# ---------------------------------------------------------------------------
# Tab 2: Launch — show Colab link
# ---------------------------------------------------------------------------

COLAB_BASE = "https://colab.research.google.com/github"


def get_colab_link(manifest_json: str) -> str:
    """Generate a Colab launch link with the manifest."""
    return (
        "Copy the manifest JSON above, then open the Colab notebook:\n\n"
        "1. Open notebooks/crisis_world_sft.ipynb in Colab\n"
        "2. Paste the manifest in the 'Load Manifest' cell\n"
        "3. Run all cells\n\n"
        "The notebook will train, evaluate, and push results to the Hub."
    )


# ---------------------------------------------------------------------------
# Tab 3: Browse — show past runs
# ---------------------------------------------------------------------------

def browse_runs(results_repo: str) -> str:
    """List runs from a Hub results repo."""
    if not results_repo:
        return "Enter a results repo ID to browse runs."
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        files = api.list_repo_files(results_repo, repo_type="dataset")
        result_files = [f for f in files if f.endswith(".json")]
        if not result_files:
            return "No results found in repo."
        return "\n".join(f"- {f}" for f in result_files)
    except Exception as exc:
        return f"Error browsing repo: {exc}"


# ---------------------------------------------------------------------------
# Tab 4: Compare — render comparison tables
# ---------------------------------------------------------------------------

def compare_runs(results_text: str) -> str:
    """Render a comparison from results JSON."""
    if not results_text.strip():
        return "Paste results JSON to compare."
    try:
        data = json.loads(results_text)
        if "comparison_table" in data:
            return data["comparison_table"]
        return json.dumps(data, indent=2)
    except Exception as exc:
        return f"Error parsing results: {exc}"


# ---------------------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------------------

def create_app() -> gr.Blocks:
    with gr.Blocks(title="CrisisWorld + Cortex") as app:
        gr.Markdown("# CrisisWorld + Cortex — Experiment Control Plane")

        with gr.Tab("Configure"):
            gr.Markdown("### Generate a Run Manifest")
            with gr.Row():
                run_type = gr.Dropdown(
                    choices=["router_sft", "single_policy_sft", "cortex_eval", "single_eval"],
                    value="router_sft", label="Run Type",
                )
                base_model = gr.Dropdown(
                    choices=[
                        "meta-llama/Llama-3.1-8B-Instruct",
                        "meta-llama/Meta-Llama-3-8B-Instruct",
                        "Qwen/Qwen2.5-7B-Instruct",
                    ],
                    value="meta-llama/Llama-3.1-8B-Instruct",
                    label="Base Model",
                )
            with gr.Row():
                dataset_repo = gr.Textbox(label="Dataset Repo", placeholder="your-username/crisis-world-sft-data")
                seeds = gr.Textbox(label="Seeds (comma-separated)", value="42, 43, 44")
            with gr.Row():
                num_epochs = gr.Slider(1, 10, value=3, step=1, label="Epochs")
                lora_r = gr.Slider(4, 64, value=16, step=4, label="LoRA rank")
                learning_rate = gr.Number(value=2e-4, label="Learning Rate")
                eval_turns = gr.Slider(5, 50, value=20, step=5, label="Eval Turns")

            manifest_output = gr.Code(language="json", label="Generated Manifest")
            gen_btn = gr.Button("Generate Manifest", variant="primary")
            gen_btn.click(
                generate_manifest,
                inputs=[run_type, base_model, dataset_repo, num_epochs, lora_r, learning_rate, eval_turns, seeds],
                outputs=manifest_output,
            )

        with gr.Tab("Launch"):
            gr.Markdown("### Launch Training in Colab")
            manifest_input = gr.Code(language="json", label="Manifest JSON")
            launch_output = gr.Markdown()
            launch_btn = gr.Button("Get Launch Instructions")
            launch_btn.click(get_colab_link, inputs=manifest_input, outputs=launch_output)

        with gr.Tab("Browse"):
            gr.Markdown("### Browse Past Runs")
            repo_input = gr.Textbox(label="Results Repo ID", placeholder="your-username/crisis-world-results")
            runs_output = gr.Markdown()
            browse_btn = gr.Button("Browse")
            browse_btn.click(browse_runs, inputs=repo_input, outputs=runs_output)

        with gr.Tab("Compare"):
            gr.Markdown("### Compare Run Results")
            results_input = gr.Code(language="json", label="Results JSON")
            compare_output = gr.Markdown()
            compare_btn = gr.Button("Compare")
            compare_btn.click(compare_runs, inputs=results_input, outputs=compare_output)

    return app


app = create_app()

if __name__ == "__main__":
    app.launch()
