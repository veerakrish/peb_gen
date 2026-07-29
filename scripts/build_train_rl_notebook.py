"""Generates kaggle/train_rl_notebook.ipynb from the cell sources below.

Kept as a generator script (rather than hand-editing ipynb JSON) so the
exact same cell source strings can be sliced out and run locally as a
plain script for verification (see scripts/verify_train_rl_notebook.py) —
the notebook and its local smoke test can't silently drift apart.
"""

import json
import os
import uuid

MARKDOWN_CELLS = {}

CELLS = [
    (
        "markdown",
        """# Stage 2: PPO Placement Policy Training

Trains the PPO placement agent (`src/models/placement_rl.py`) on top of the
frozen Stage 1 Graph Transformer encoder (`checkpoints/encoder_best.pth`),
using `PCBPlacementEnv` (`src/environment/pcb_gym_env.py`).

**Before running on Kaggle:**
1. Upload this repo as a Kaggle Dataset (or clone it in a Kaggle Notebook
   with internet access) and set `REPO_ROOT` below to wherever it lands
   (typically `/kaggle/input/<dataset-name>` or `/kaggle/working/pcb_gen`
   after a `git clone`).
2. Upload `checkpoints/encoder_best.pth` (produced by
   `src/pipelines/train_encoder.py`) as a Kaggle Dataset and set
   `ENCODER_PATH` to its path under `/kaggle/input/...`.
3. Enable a GPU/multi-core CPU accelerator in the notebook's settings.
""",
    ),
    (
        "code",
        """# ==============================================================================
# CELL: Setup & Dependencies
# ==============================================================================
!pip install -q stable-baselines3 torch-geometric gymnasium shapely pyyaml tensorboard
""",
    ),
    (
        "code",
        """import os
import sys
import random

import torch
import yaml
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

# Point this at wherever the repo lives in this Kaggle session (see the
# markdown cell above). Defaults to a local checkout for running outside
# Kaggle (e.g. this project's own repo root).
REPO_ROOT = os.environ.get("PCB_GEN_REPO_ROOT", ".")
sys.path.insert(0, REPO_ROOT)

from data.synthetic_generator import generate_graph
from src.environment.pcb_gym_env import PCBPlacementEnv
from src.models.placement_rl import build_ppo_agent, load_frozen_encoder, compute_node_embeddings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using compute device: {device}")
""",
    ),
    (
        "code",
        """# ==============================================================================
# CELL: Locate the Stage 1 checkpoint and configs
# ==============================================================================
# Kaggle path if you uploaded checkpoints/encoder_best.pth as a Dataset named
# "pcb-encoder-checkpoint"; falls back to a local path so this same notebook
# runs unmodified outside Kaggle.
_KAGGLE_ENCODER_PATH = "/kaggle/input/pcb-encoder-checkpoint/encoder_best.pth"
_LOCAL_ENCODER_PATH = os.path.join(REPO_ROOT, "checkpoints", "encoder_best.pth")
ENCODER_PATH = _KAGGLE_ENCODER_PATH if os.path.exists(_KAGGLE_ENCODER_PATH) else _LOCAL_ENCODER_PATH

DEFAULT_CONFIG_PATH = os.path.join(REPO_ROOT, "config", "default_config.yaml")
DATASET_CONFIG_PATH = os.path.join(REPO_ROOT, "config", "dataset_config.yaml")

if not os.path.exists(ENCODER_PATH):
    raise FileNotFoundError(
        f"Stage 1 checkpoint not found at {ENCODER_PATH}. Run "
        "src/pipelines/train_encoder.py first (or upload encoder_best.pth "
        "as a Kaggle Dataset named 'pcb-encoder-checkpoint')."
    )
print(f"Using encoder checkpoint: {ENCODER_PATH}")

with open(DEFAULT_CONFIG_PATH) as f:
    default_config = yaml.safe_load(f)
with open(DATASET_CONFIG_PATH) as f:
    dataset_config = yaml.safe_load(f)
""",
    ),
    (
        "code",
        """# ==============================================================================
# CELL: Per-episode random netlist environment
# ==============================================================================
# PCBPlacementEnv places every component of *one fixed* netlist per episode
# (see src/environment/pcb_gym_env.py) — it doesn't generate boards itself.
# Training on a single fixed board would just memorize that one layout, so
# this wrapper regenerates a fresh synthetic drone netlist (and its Stage 1
# embeddings) on every reset(). Standard SB3 rollout collection calls
# reset() with no options once an episode ends, so this has to happen
# inside reset() itself rather than being threaded through
# reset(options={"netlist": ...}) by the caller.


class RandomNetlistPCBEnv(PCBPlacementEnv):
    def __init__(self, config_path, dataset_config, board_config, encoder, categories, pin_types, rng_seed=None):
        self._dataset_config = dataset_config
        self._board_config = board_config
        self._encoder = encoder
        self._categories = categories
        self._pin_types = pin_types
        self._gen_rng = random.Random(rng_seed)

        netlist = generate_graph(self._gen_rng, dataset_config, board_config)
        super().__init__(config_path=config_path, netlist=netlist)
        self.set_embeddings(compute_node_embeddings(encoder, netlist, categories, pin_types))

    def reset(self, *, seed=None, options=None):
        if options is None or "netlist" not in options:
            netlist = generate_graph(self._gen_rng, self._dataset_config, self._board_config)
            self.load_netlist(netlist)
            self.set_embeddings(
                compute_node_embeddings(self._encoder, netlist, self._categories, self._pin_types)
            )
        return super().reset(seed=seed, options=options)


def make_env(rank: int):
    def _init():
        # Each subprocess gets its own frozen-encoder copy rather than
        # sharing one loaded in the parent process — torch models loaded
        # on a specific device don't reliably survive being pickled across
        # SubprocVecEnv's process boundary.
        encoder = load_frozen_encoder(ENCODER_PATH, device="cpu")
        env = RandomNetlistPCBEnv(
            config_path=DEFAULT_CONFIG_PATH,
            dataset_config=dataset_config,
            board_config=default_config["board"],
            encoder=encoder,
            categories=default_config["encoder"]["node_categories"],
            pin_types=default_config["encoder"]["pin_types"],
            rng_seed=rank,
        )
        return Monitor(env)

    return _init
""",
    ),
    (
        "code",
        """# ==============================================================================
# CELL: Vectorized environments & PPO agent
# ==============================================================================
NUM_ENVS = 4  # Kaggle notebooks typically get 4 CPU cores
VecEnvCls = SubprocVecEnv if NUM_ENVS > 1 else DummyVecEnv
env = VecEnvCls([make_env(i) for i in range(NUM_ENVS)])

# The reward weights (overlap_area=100, hpwl=1, out_of_bounds=50, each
# multiplied by areas in mm^2) produce per-episode returns in the 1e5-1e6
# range. PPO's loss combines policy_loss + vf_coef*value_loss — with value
# targets that large, value_loss dwarfs the policy gradient term, so the
# optimizer step is almost entirely about the critic and the actor barely
# moves. Caught from a real run: explained_variance stuck at ~0 and the
# Gaussian policy's std frozen at exactly 1.0 for 100k+ steps even after
# fixing the action-space normalization (see PCBPlacementEnv). Normalizing
# only the reward (not observations — node_embeddings/masks/occupancy
# already have sensible ranges, and normalizing binary masks would be
# actively wrong) brings value_loss down to a well-conditioned scale.
env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)

rl_cfg = default_config["placement_rl"]
model, _encoder_unused = build_ppo_agent(
    env,
    encoder_checkpoint_path=None,  # each subprocess already loaded its own copy inside make_env
    learning_rate=rl_cfg["learning_rate"],
    n_steps=rl_cfg["n_steps"],
    batch_size=rl_cfg["batch_size"],
    n_epochs=rl_cfg["n_epochs"],
    gamma=rl_cfg["gamma"],
    gae_lambda=rl_cfg["gae_lambda"],
    clip_range=rl_cfg["clip_range"],
    ent_coef=rl_cfg["ent_coef"],
    verbose=1,
    tensorboard_log="/kaggle/working/tb_logs/",
)
print(model.policy)
""",
    ),
    (
        "code",
        """# ==============================================================================
# CELL: Train
# ==============================================================================
# Kaggle sessions can time out mid-run; checkpoint periodically so a
# restart can resume from the latest save instead of losing all progress.
os.makedirs("/kaggle/working/checkpoints", exist_ok=True)
checkpoint_callback = CheckpointCallback(
    save_freq=max(1, 10_000 // NUM_ENVS),
    save_path="/kaggle/working/checkpoints",
    name_prefix="ppo_pcb_placement",
)

TOTAL_TIMESTEPS = 100_000
print("Starting PPO Policy Training...")
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_callback)
""",
    ),
    (
        "code",
        """# ==============================================================================
# CELL: Save final policy
# ==============================================================================
final_path = "/kaggle/working/checkpoints/ppo_pcb_placement_best"
model.save(final_path)
# VecNormalize's running reward statistics — needed to resume training
# with the same normalization; not needed just to run inference with the
# saved policy, since only actions (not rewards) are used at deployment.
env.save("/kaggle/working/checkpoints/vecnormalize_stats.pkl")
print(f"Saved Stage 2 policy to {final_path}.zip")
""",
    ),
]


def build_notebook():
    cells = []
    for cell_type, source in CELLS:
        cell = {
            "id": uuid.uuid4().hex[:8],
            "cell_type": cell_type,
            "metadata": {},
            "source": source.splitlines(keepends=True),
        }
        if cell_type == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        cells.append(cell)

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    out_path = os.path.join(os.path.dirname(__file__), "..", "kaggle", "train_rl_notebook.ipynb")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(build_notebook(), f, indent=1)
    print(f"wrote {out_path}")
