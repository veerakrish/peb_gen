import json
import os

import yaml

from src.environment.pcb_gym_env import PCBPlacementEnv
from src.models.placement_rl import build_ppo_agent, compute_node_embeddings

CONFIG_PATH = "config/default_config.yaml"
SYNTHETIC_SAMPLE = os.path.join("data", "raw", "synthetic", "pcb_00000.json")
ENCODER_CHECKPOINT = os.path.join("checkpoints", "encoder_best.pth")


def test_ppo_agent_trains_with_frozen_encoder_embeddings():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    categories = cfg["encoder"]["node_categories"]
    pin_types = cfg["encoder"]["pin_types"]

    with open(SYNTHETIC_SAMPLE) as f:
        netlist = json.load(f)

    env = PCBPlacementEnv(config_path=CONFIG_PATH, netlist=netlist)
    model, encoder = build_ppo_agent(
        env,
        encoder_checkpoint_path=ENCODER_CHECKPOINT,
        n_steps=256,
        batch_size=64,
        verbose=0,
    )
    assert encoder is not None

    embeddings = compute_node_embeddings(encoder, netlist, categories, pin_types)
    assert len(embeddings) == len(netlist["components"])
    assert all(v.shape == (cfg["encoder"]["embedding_dim"],) for v in embeddings.values())
    env.set_embeddings(embeddings)

    model.learn(total_timesteps=1000)

    obs, _ = env.reset(seed=0)
    action, _ = model.predict(obs, deterministic=True)
    assert action.shape == (3,)
    assert env.action_space.contains(action.astype("float32"))
