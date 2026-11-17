"""Shared fixtures. Every test that takes ``assets`` (directly or through ``robot``, ``two_clips`` or
``tiny_config``) runs once per robot and skips a robot whose data ``python -m supermimic_hoi.prepare``
has not built yet."""

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import torch

DATA = Path(__file__).resolve().parents[1] / "data"


@dataclass(frozen=True)
class Assets:
    robot: Path
    objects: Path
    dataset: Path
    clip_objects: tuple[str, str]  # the two test clips are the first clips with these objects
    capacity: dict = field(default_factory=dict)  # contact and constraint capacity, where the default is too small


ASSETS = {
    "g1_inspire": Assets(DATA / "robot", DATA / "objects", DATA / "weave600", ("smallbox", "woodchair")),
}  # fmt: skip


@pytest.fixture(scope="session", params=list(ASSETS))
def assets(request):
    found = ASSETS[request.param]
    if not (found.dataset / "manifest.json").is_file():
        pytest.skip(f"Prepared {request.param} data not found; run python -m supermimic_hoi.prepare")
    return found


@pytest.fixture(scope="session")
def robot(assets):
    from supermimic_hoi.robots import load_robot

    return load_robot(assets.robot)


@pytest.fixture(scope="session")
def two_clips(assets):
    names = json.loads((assets.dataset / "manifest.json").read_text())["clip_names"]
    return [next(n for n in names if obj in n) for obj in assets.clip_objects]


@pytest.fixture
def tiny_config(assets, two_clips, tmp_path):
    """Two clips, four CPU worlds and very small networks."""
    from supermimic_hoi.config import TrainConfig

    torch.manual_seed(0)
    return TrainConfig(
        dataset=str(assets.dataset), robot_assets=str(assets.robot), object_assets=str(assets.objects),
        clips=two_clips, num_envs=4, device="cpu", backend="cpu", randomize=False, cuda_graph=False, compile=False,
        amp="none", iterations=2, warmup_steps=12, buffer_steps=64, wm_batch=16, pi_batch=8, horizon=6, wm_updates=2,
        hidden=[64, 64], policy_hidden=[64, 64], reward_head_hidden=[32], critic_hidden=[32], log_every=1,
        save_every=100, log_dir=str(tmp_path / "run"), **assets.capacity)  # fmt: skip
