"""The supported robots. An asset directory holds exactly one of them, named by its MJCF file."""

from pathlib import Path

from ..robot import Robot
from .g1_inspire import G1Inspire

ROBOTS = (G1Inspire,)


def load_robot(asset_dir: str | Path) -> Robot:
    found = [robot for robot in ROBOTS if (Path(asset_dir) / robot.mjcf).is_file()]
    if len(found) != 1:
        raise FileNotFoundError(f"{asset_dir} must hold one of {[robot.mjcf for robot in ROBOTS]}; run `python -m supermimic_hoi.prepare`")
    return found[0](asset_dir)
