"""Audit reference resets on the training physics.

Every admissible reset frame of every clip is restored and followed for a few control steps with
the reference-PD action (zero body residual, reference finger targets). This checks that reset
states are reproduced exactly and that none of them is numerically violent; it is not a tracking
benchmark, since open-loop PD is not expected to follow a clip.

    python -m supermimic_hoi.audit --config configs/weave600.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from .config import TrainConfig
from .env import make_task
from .mdp import FAILURE_REASONS
from .state import pack


@torch.no_grad()
def audit(cfg: TrainConfig, steps: int = 5, all_frames: bool = False) -> tuple[dict, dict]:
    """Returns (summary, per-frame tensors). ``all_frames`` also tests frames the penetration screen rejects."""
    cfg = replace(cfg, randomize=False, cuda_graph=False, compile=False)
    cfg.num_envs = len(cfg.clips) or len(json.loads((Path(cfg.dataset) / "manifest.json").read_text())["clip_names"])
    task = make_task(cfg)
    env, m, physics = task.env, task.motion, task.env.physics
    F = m.num_frames
    tested = torch.zeros(F, dtype=torch.bool, device=cfg.device)
    failed, rejected = torch.zeros_like(tested), torch.zeros_like(tested)
    reasons = torch.zeros(F, len(FAILURE_REASONS), dtype=torch.bool, device=cfg.device)
    speed = torch.zeros(F, 3, device=cfg.device)  # peak root, object and joint speed
    reset_error = 0.0
    for offset in range(int(m.clip_len.max()) - 1):
        frames = m.clip_start + offset
        active = frames < m.clip_end
        frames = torch.where(active, frames, m.clip_start)
        if not all_frames:
            active &= env.reset_map[frames] == frames
        if not bool(active.any()):
            continue
        env.reset(torch.ones_like(active), frames)
        error = (pack(physics.state()) - pack(m.ref_state(frames))).abs().amax(-1)
        reset_error = max(reset_error, float(error[active].max()))
        alive = active.clone()
        for _ in range(steps):
            row = env.step(task.controller.reference_action(m.q[m.next_frame(env.f)]))
            physics.check()
            s1 = physics.state()
            peak = torch.stack((s1[2].norm(dim=-1), s1[8].norm(dim=-1), s1[5].abs().amax(-1)), -1)
            speed[frames] = torch.where(alive[:, None], torch.maximum(speed[frames], peak), speed[frames])
            rejected[frames] |= alive & ~row["valid"]
            failed[frames] |= alive & row["term"]
            reasons[frames] |= row["reasons"] & alive[:, None]
            alive &= ~row["cut"]
        tested[frames] |= active
    n = int(tested.sum())
    summary = dict(
        clips=len(m.names), frames=F, tested_frames=n, steps_per_reset=steps, all_frames=all_frames,
        reset_guard=env.reset_summary, reset_max_error=reset_error,
        failed_frames=int((failed & tested).sum()), rejected_frames=int((rejected & tested).sum()),
        failures={name: int((reasons[:, i] & tested).sum()) for i, name in enumerate(FAILURE_REASONS)},
        peak_root_speed=float(speed[tested, 0].max()), peak_object_speed=float(speed[tested, 1].max()),
        peak_joint_speed=float(speed[tested, 2].max()))  # fmt: skip
    arrays = dict(tested=tested, failed=failed, rejected=rejected, reasons=reasons, speed=speed, depth=env.reset_depths)
    return summary, {k: v.cpu() for k, v in arrays.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config")
    p.add_argument("--device")
    p.add_argument("--backend")
    p.add_argument("--clips", nargs="+")
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--all-frames", action="store_true", help="Also test frames the penetration screen rejects")
    p.add_argument("--output", default="results/audit.json")
    a = p.parse_args()
    cfg = TrainConfig.from_json(a.config, device=a.device, backend=a.backend, clips=a.clips)
    summary, arrays = audit(cfg, a.steps, a.all_frames)
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    torch.save(arrays, out.with_suffix(".pt"))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
