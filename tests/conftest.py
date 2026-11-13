"""Shared fixtures. Tests that need prepared assets skip until `python -m supermimic_hoi.prepare` has run."""

import json
from pathlib import Path

import pytest
import torch

DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="session")
def data():
    if not (DATA / "weave600" / "manifest.json").is_file():
        pytest.skip("Prepared data not found; run python -m supermimic_hoi.prepare")
    return DATA


@pytest.fixture(scope="session")
def robot(data):
    from supermimic_hoi.robot import Robot

    return Robot(data / "robot")


@pytest.fixture(scope="session")
def two_clips(data):
    names = json.loads((data / "weave600" / "manifest.json").read_text())["clip_names"]
    return [next(n for n in names if "smallbox" in n), next(n for n in names if "woodchair" in n)]


@pytest.fixture
def tiny_config(data, two_clips, tmp_path):
    """Two clips, four CPU worlds and very small networks."""
    from supermimic_hoi.config import TrainConfig

    torch.manual_seed(0)
    return TrainConfig(
        dataset=str(data / "weave600"), robot_assets=str(data / "robot"), object_assets=str(data / "objects"),
        clips=two_clips, num_envs=4, device="cpu", backend="cpu", randomize=False, cuda_graph=False, compile=False,
        amp="none", iterations=2, warmup_steps=12, buffer_steps=64, wm_batch=16, pi_batch=8, horizon=6, wm_updates=2,
        hidden=[64, 64], policy_hidden=[64, 64], reward_head_hidden=[32], critic_hidden=[32], log_every=1,
        save_every=100, log_dir=str(tmp_path / "run"))  # fmt: skip
