"""Train a policy.

    python -m supermimic_hoi.train --config configs/weave600.json
"""

import argparse

import torch

from .config import TrainConfig
from .runner import Runner


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="JSON file of TrainConfig fields")
    for name in ("dataset", "robot_assets", "object_assets", "device", "backend", "log_dir", "resume", "swanlab_mode"):
        p.add_argument("--" + name.replace("_", "-"))
    for name in ("num_envs", "iterations", "seed"):
        p.add_argument("--" + name.replace("_", "-"), type=int)
    p.add_argument("--clips", nargs="+", help="Train on these clips only")
    p.add_argument("--threads", type=int, default=4)
    a = vars(p.parse_args())
    torch.set_num_threads(a.pop("threads"))
    Runner(TrainConfig.from_json(a.pop("config"), **a)).train()


if __name__ == "__main__":
    main()
