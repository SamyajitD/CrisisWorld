# %% [markdown]
# # Router SFT Showcase Plan
#
# Goal: create a Python script that can be opened as a notebook and used to
# demonstrate a small supervised fine-tuning run with Unsloth on Llama 3.1 8B,
# using this repository's router SFT data. The run is intentionally short and
# is only meant to show that the model can fit the task enough for visible
# movement in training curves and lightweight episode metrics.
#
# This is a plan file, not the final runnable notebook.

# %% [markdown]
# ## Why this path
#
# - Use `data/router_sft.jsonl`, not `data/policy_sft.jsonl`.
# - `router_sft` already contains rich chat prompts and episode metadata.
# - `policy_sft` in this repo is too shallow for a convincing learning demo.
# - To show episode rewards after SFT, keep `perception`, `world_modeler`,
#   `planner`, and `critic` heuristic, and replace only the Executive role with a
#   notebook-local LLM wrapper around the trained adapter.

# %% [markdown]
# ## Runtime assumptions
#
# - Notebook runtime: Colab, Kaggle, or Jupyter with a CUDA GPU.
# - Target GPU: T4/L4/A10/A100 class GPU.
# - Python: 3.11 preferred to match this repo.
# - Hugging Face login is required if a gated Llama model is used.
# - The run is capped to a small subset and a small number of optimizer steps.
#
# Expected outcome:
# - downward training loss
# - valid JSON rate on held-out prompts
# - route-label agreement on held-out prompts
# - small before/after movement on episode reward and outbreak duration

# %% [markdown]
# ## Notebook structure
#
# The final `.py` should use `# %%` cells so it is directly convertible to
# Jupyter.

# %% [markdown]
# ### Cell 1: User config
#
# Define all notebook knobs in one place:
#
# - `REPO_URL`
# - `REPO_DIR = "MetaFinals"`
# - `BASE_MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"`
# - `MAX_SEQ_LENGTH = 1024 or 1536`
# - `TRAIN_ROWS = 256 to 1024`
# - `VAL_ROWS = 64 to 128`
# - `MAX_STEPS = 30 to 80`
# - `EPISODE_SEEDS = [101, 102, 103]`
# - `OUTPUT_DIR = "outputs/notebook_router_sft_demo"`
# - `USE_WANDB = False`
#
# Keep the defaults small enough that the notebook finishes in a reasonable
# time and clearly behaves like a showcase, not a full training run.

# %% [markdown]
# ### Cell 2: Clone or update the repo
#
# In a hosted notebook you do not have the current local directory, so the
# notebook should explicitly pull the repo:
#
# 1. If `MetaFinals/` does not exist: `git clone REPO_URL MetaFinals`
# 2. Else: `git -C MetaFinals pull --ff-only`
# 3. `cd MetaFinals`
#
# This cell should print the current commit SHA for reproducibility.

# %% [markdown]
# ### Cell 3: Install project + training dependencies
#
# Install:
#
# - `pip install -e .`
# - `pip install -r training/router_sft/requirements.txt`
# - `pip install unsloth trl bitsandbytes matplotlib seaborn pandas`
#
# Optional:
#
# - `pip install wandb`
#
# Keep the notebook self-contained: it should not assume the package is already
# available in the runtime.

# %% [markdown]
# ### Cell 4: Imports, seeds, and output directories
#
# Create:
#
# - deterministic seeds for Python / NumPy / Torch
# - output directories for checkpoints, plots, and cached eval outputs
# - a simple `RunConfig` dataclass for notebook settings

# %% [markdown]
# ### Cell 5: Load and trim the router SFT dataset
#
# Steps:
#
# 1. Read `data/router_sft.jsonl`
# 2. Parse `messages` and `metadata`
# 3. Shuffle with a fixed seed
# 4. Keep only a small subset for the showcase
# 5. Build train/validation splits
# 6. Print route-label distribution and a few sample prompts
#
# Also derive two evaluation targets from metadata:
#
# - `route_label`
# - `episode_total_reward`
#
# These are useful for analysis even though SFT itself optimizes token loss.

# %% [markdown]
# ### Cell 6: Build evaluation helpers before training
#
# Add helpers for:
#
# - JSON extraction from model text
# - route-label normalization
# - exact-parse rate
# - route-label accuracy
# - optional action-present / role-present checks
#
# Run these helpers on a small held-out sample with the unfine-tuned base model
# so the notebook has a clean pre-SFT baseline.

# %% [markdown]
# ### Cell 7: Load Unsloth model and tokenizer
#
# Use Unsloth's fast loader with a 4-bit Llama 3.1 8B checkpoint.
#
# Plan:
#
# - `from unsloth import FastLanguageModel`
# - `model, tokenizer = FastLanguageModel.from_pretrained(...)`
# - set `load_in_4bit=True`
# - set `max_seq_length`
# - attach LoRA with `FastLanguageModel.get_peft_model(...)`
#
# Recommended LoRA targets:
#
# - `q_proj`
# - `k_proj`
# - `v_proj`
# - `o_proj`
# - `gate_proj`
# - `up_proj`
# - `down_proj`
#
# Keep rank modest, for example `r=16`.

# %% [markdown]
# ### Cell 8: Prepare conversational text for SFT
#
# Convert each row's `messages` into a single training string using the tokenizer
# chat template. The notebook should do this explicitly instead of depending on
# hidden defaults, so it is easier to debug and plot.
#
# Suggested fields:
#
# - `text`
# - `route_label`
# - `episode_total_reward`
#
# The SFT trainer only needs `text`; the other fields are kept for evaluation.

# %% [markdown]
# ### Cell 9: Configure a short SFT run
#
# Training settings should be deliberately small:
#
# - `per_device_train_batch_size = 2`
# - `gradient_accumulation_steps = 4`
# - `warmup_steps = 5`
# - `max_steps = 30 to 80`
# - `learning_rate = 2e-4`
# - `logging_steps = 1 or 2`
# - `eval_steps = 10`
# - `save_strategy = "no"` for the quickest demo, or `"steps"` for checkpoints
#
# Use `trl.SFTTrainer` on top of the Unsloth model so training logs are easy to
# inspect from `trainer.state.log_history`.

# %% [markdown]
# ### Cell 10: Train
#
# Run the short SFT job and save:
#
# - adapter weights
# - tokenizer
# - trainer metrics JSON
# - raw `log_history`
#
# The notebook should print:
#
# - final train loss
# - final eval loss if present
# - wall-clock time

# %% [markdown]
# ### Cell 11: Plot the training curves
#
# Plot from `trainer.state.log_history`:
#
# - train loss vs step
# - eval loss vs step
# - learning rate vs step
# - grad norm vs step if logged
#
# This cell is the main "model is learning" visual.

# %% [markdown]
# ### Cell 12: Offline validation before vs after SFT
#
# Evaluate the same held-out prompt subset with:
#
# - base model
# - trained adapter
#
# Report:
#
# - valid JSON rate
# - route-label accuracy
# - a small table of prompt / teacher / base / tuned outputs
#
# This is the fastest quality signal and should run before episode evaluation.

# %% [markdown]
# ### Cell 13: Build a local Executive role for environment episodes
#
# The repo's existing LLM path is built around Hugging Face Inference API, so
# the notebook needs a local wrapper.
#
# Add:
#
# - `LocalExecutiveProvider.complete(system, user) -> dict`
# - generation with the tuned Unsloth model
# - JSON extraction with fallback to heuristic `ExecutiveRole`
# - `NotebookExecutiveRole.invoke(RoleInput) -> ExecutiveDecision`
#
# Important constraint:
# - only the Executive role should be swapped to the local LLM
# - all other roles stay heuristic so the environment run remains stable

# %% [markdown]
# ### Cell 14: Run lightweight showcase episodes
#
# Build a notebook-local Cortex agent:
#
# - heuristic `PerceptionRole`
# - heuristic `WorldModelerRole`
# - heuristic `PlannerRole`
# - heuristic `CriticRole`
# - tuned `NotebookExecutiveRole`
#
# Compare against:
#
# - heuristic Executive baseline
# - optional base-model Executive before SFT
#
# Keep this tiny:
#
# - 3 to 5 seeds
# - short max-turns if configurable
#
# Collect:
#
# - `total_cumulative_reward`
# - `outbreak_duration`
# - `termination_reason`
# - decision mix, for example `act` vs `call`

# %% [markdown]
# ### Cell 15: Plot reward-oriented metrics
#
# Plot small before/after comparisons:
#
# - cumulative reward by seed
# - mean reward with error bars
# - outbreak duration by seed
# - decision distribution
#
# Important wording in the notebook:
# - these are environment rewards from short evaluation rollouts
# - they are not RL rewards used during SFT training

# %% [markdown]
# ### Cell 16: Final summary and caveats
#
# Close with a concise interpretation:
#
# - loss went down or not
# - JSON validity improved or not
# - route agreement improved or not
# - episode reward changed or not
#
# Caveats to state clearly:
#
# - this is a tiny showcase run
# - no claim of convergence
# - reward movement can be noisy with few seeds
# - SFT optimizes imitation loss, not reward directly

# %% [markdown]
# ## Suggested implementation notes
#
# - Keep all notebook-only helpers inside the notebook file. Do not patch the
#   repo unless you decide to promote the notebook path into a reusable module.
# - Save all artifacts under `outputs/notebook_router_sft_demo/`.
# - Prefer explicit prints over hidden side effects.
# - If memory is tight, reduce:
#   - `MAX_SEQ_LENGTH`
#   - `TRAIN_ROWS`
#   - `MAX_STEPS`
#   - LoRA rank

# %% [markdown]
# ## Deliverable expected after this plan
#
# A single file such as:
#
# `notebooks/router_sft_unsloth_showcase.py`
#
# with `# %%` cells, runnable top to bottom in a GPU notebook, producing:
#
# - installed local repo
# - short Unsloth SFT run on Llama 3.1 8B
# - training curve plots
# - held-out prompt metrics
# - short episode reward plots
