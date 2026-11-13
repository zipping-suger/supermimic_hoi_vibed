"""Optional SwanLab logging. The SDK is imported only when tracking is enabled."""

from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path


class Tracker:
    def __init__(self, cfg):
        self.cfg, self.run = cfg, None

    def __enter__(self):
        if self.cfg.swanlab_mode != "disabled":
            import swanlab

            log_dir = Path(self.cfg.log_dir)
            self.run = swanlab.init(project=self.cfg.swanlab_project, experiment_name=self.cfg.swanlab_name or log_dir.name,
                                    config=asdict(self.cfg), mode=self.cfg.swanlab_mode, logdir=str(log_dir / "swanlog"))  # fmt: skip
        return self

    def log(self, metrics: dict, step: int):
        if self.run is not None:
            self.run.log({k: v for k, v in metrics.items() if math.isfinite(v)}, step=step)

    def __exit__(self, exc_type, exc, traceback):
        if self.run is not None:
            self.run.finish(state="success" if exc_type is None else "crashed")
        return False
