"""Deterministic evaluation of a checkpoint's EMA policy: every clip once, from its first admissible frame.

    python -m supermimic_hoi.evaluate logs/weave600/last.pt --output results/evaluation.json

A clip counts as completed when the policy reaches its last frame without a termination. Contact
is not a termination condition, so the report also gives the fraction of intended hand-steps with
measured object contact. No exploration noise, observation noise or domain randomization is applied.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .config import TrainConfig
from .mdp import FAILURE_REASONS
from .rotations import angle_squared
from .runner import Runner
from .state import pack


def load_runner(checkpoint: str, **overrides) -> Runner:
    """Rebuild the checkpoint's networks on one world per clip, without randomization or capture."""
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)["config"]
    saved.update({k: v for k, v in overrides.items() if v is not None})
    saved.update(randomize=False, compile=False, cuda_graph=False, resume="", swanlab_mode="disabled")
    cfg = TrainConfig(**saved)
    if cfg.device == "cpu":
        cfg.backend = "cpu"
    cfg.num_envs = len(cfg.clips) or len(json.loads((Path(cfg.dataset) / "manifest.json").read_text())["clip_names"])
    runner = Runner(cfg)
    runner.load(checkpoint, weights_only=True)
    return runner


@torch.no_grad()
def evaluate(runner: Runner, record=()) -> tuple[dict, dict]:
    """Returns (report, trajectories). ``record``: clip names whose packed states are kept per step."""
    env, m, mdp, cfg = runner.env, runner.motion, runner.mdp, runner.cfg
    N, device = env.N, cfg.device
    everyone = torch.ones(N, dtype=torch.bool, device=device)
    start = env.reset_map[m.clip_start]
    env.reset(everyone, start)
    active = everyone.clone()
    steps = torch.zeros(N, dtype=torch.long, device=device)
    sums = torch.zeros(N, 3, device=device)  # object position, object rotation, tracked-body position error
    contact = torch.zeros(N, 2, device=device)  # (intended hand-steps with contact, intended hand-steps)
    reasons = torch.zeros(N, len(FAILURE_REASONS), dtype=torch.bool, device=device)
    completed = torch.zeros(N, dtype=torch.bool, device=device)
    recorded = {name: [pack(env.physics.state())[m.names.index(name)].cpu()] for name in record}
    while bool(active.any()):
        row = env.step(runner.ema(env.history, m.goal(env.f, cfg.future)))
        env.physics.check()
        s, f = env.physics.state(), env.f
        body = mdp.fk(s)[0][:, : m.num_tracked]
        errors = torch.stack(((s[6] - m.object_p[f]).norm(dim=-1), angle_squared(s[7], m.object_R[f]).sqrt(),
                              (body - m.body_p[f][:, : m.num_tracked]).norm(dim=-1).mean(-1)), -1)  # fmt: skip
        intent = m.hand_intent(f)
        sums += torch.where(active[:, None], errors, 0.0)
        contact += torch.where(active[:, None], torch.stack(((row["contact"] * intent).sum(-1), intent.sum(-1)), -1), 0.0)
        steps += active
        for name, states in recorded.items():
            if active[m.names.index(name)]:
                states.append(pack(s)[m.names.index(name)].cpu())
        reasons |= row["reasons"] & active[:, None]
        completed |= active & row["clip_end"] & ~row["term"]
        active &= ~(row["term"] | row["clip_end"])
        # Finished worlds idle at their start frame so they cannot disturb the batch numerically.
        env.reset(~active, start)

    reference_steps = (m.clip_end - start).cpu()
    steps, sums, contact, reasons, completed = steps.cpu(), sums.cpu(), contact.cpu(), reasons.cpu(), completed.cpu()
    results = []
    for i, name in enumerate(m.names):
        n = max(int(steps[i]), 1)
        results.append(dict(
            clip=name, object=m.object_names[int(m.clip_object[i])], completed=bool(completed[i]), steps=int(steps[i]),
            reference_steps=int(reference_steps[i]), start_frame=int(start[i] - m.clip_start[i]),
            failure_reasons=[r for r, flag in zip(FAILURE_REASONS, reasons[i].tolist()) if flag],
            object_position_error_m=float(sums[i, 0]) / n, object_rotation_error_rad=float(sums[i, 1]) / n,
            body_position_error_m=float(sums[i, 2]) / n,
            intended_contact_coverage=float(contact[i, 0] / contact[i, 1]) if contact[i, 1] > 0 else None))  # fmt: skip
    progress = (steps / reference_steps).numpy()
    coverage = [r["intended_contact_coverage"] for r in results if r["intended_contact_coverage"] is not None]
    per_object = {}
    for r in results:
        per_object.setdefault(r["object"], []).append(r["completed"])
    report = dict(iteration=runner.iteration, clips=len(results), success_rate=float(completed.float().mean()),
                  mean_progress=float(progress.mean()), mean_intended_contact_coverage=float(np.mean(coverage)) if coverage else None,
                  success_rate_per_object={k: float(np.mean(v)) for k, v in sorted(per_object.items())}, results=results)  # fmt: skip
    return report, {name: torch.stack(states).numpy() for name, states in recorded.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("--device")
    for name in ("dataset", "robot-assets", "object-assets"):
        p.add_argument("--" + name, help="Override the path stored in the checkpoint")
    p.add_argument("--clips", nargs="+", help="Evaluate only these clips")
    p.add_argument("--record", nargs="+", default=[], help="Save the state trajectory of these clips")
    p.add_argument("--output", default="results/evaluation.json")
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    runner = load_runner(a.checkpoint, device=a.device, dataset=a.dataset, robot_assets=a.robot_assets,
                         object_assets=a.object_assets, clips=a.clips)  # fmt: skip
    report, trajectories = evaluate(runner, a.record)
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    for name, states in trajectories.items():
        np.savez_compressed(out.with_name(f"{out.stem}_{name}.npz"), state=states, clip=name,
                            start_frame=next(r["start_frame"] for r in report["results"] if r["clip"] == name))  # fmt: skip
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    main()
