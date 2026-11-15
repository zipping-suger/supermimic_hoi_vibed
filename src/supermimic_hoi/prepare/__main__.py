"""Build ``data/`` from a Weave checkout: robot MJCF, object collision assets, sampled clips.

    python -m supermimic_hoi.prepare --weave /path/to/Weave
"""

import argparse
from pathlib import Path

from ..robot import Robot
from .dataset import prepare_dataset
from .objects import prepare_objects
from .robot import prepare_robot


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weave", required=True, type=Path, help="Weave checkout with data/<split>/*.npz downloaded")
    p.add_argument("--output", type=Path, default=Path("data"))
    p.add_argument("--split", default="test", help="Folder under the checkout's data/: test (references) or rollout/test")
    p.add_argument("--clips", type=int, default=600)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", help="Dataset directory name (default: weave<clips>)")
    a = p.parse_args()

    prepare_robot(a.weave, a.output / "robot")
    robot = Robot(a.output / "robot")
    print(f"Robot: {robot.num_joints} joints, {robot.num_actions} actions, {robot.model.body_mass.sum():.2f} kg")
    manifest = prepare_dataset(a.weave, a.output / (a.name or f"weave{a.clips}"), robot, a.split, a.clips, a.seed)
    print(f"Dataset: {manifest['clips']} clips, {manifest['frames']} frames ({manifest['hours']:.2f} h), "
          f"FK error {manifest['fk_max_error_m'] * 1000:.3f} mm, "
          f"largest joint-limit excess {manifest['joint_limit_max_excess_rad']:.4f} rad")  # fmt: skip
    print(f"Per object: {manifest['per_object']}; rejected as corrupted: {manifest['rejected_clips']}")
    prepare_objects(a.weave, a.output / "objects", list(manifest["per_object"]))


if __name__ == "__main__":
    main()
