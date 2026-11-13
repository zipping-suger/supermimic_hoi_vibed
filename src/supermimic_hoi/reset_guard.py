"""Screen reference frames for interpenetration before they are used as reset states.

Retargeted references overlap the object and the ground in places; resetting the simulator into
deep overlap produces an explosive separation. Reference poses and collision geometry are fixed,
so the overlap of every frame is measured once with CPU MuJoCo. Collection then only indexes a
device table that maps each proposed frame to the nearest admissible frame of the same clip.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import mujoco
import numpy as np
import torch

from .sim import Indexing, encode, make_scene


@torch.no_grad()
def reference_depths(motion, robot, registry, folder, threads: int = 16) -> torch.Tensor:
    """Deepest overlap in metres among all enabled collision pairs, per reference frame."""
    models = {}
    for k in sorted(set(motion.clip_object.tolist())):
        models[k] = make_scene(robot, registry, folder, [k], 1, "cpu").compile()

    def clip_depths(clip: int) -> np.ndarray:
        model = models[int(motion.clip_object[clip])]
        data, idx = mujoco.MjData(model), Indexing(model, robot)
        frames = torch.arange(int(motion.clip_start[clip]), int(motion.clip_end[clip]) + 1, device=motion.device)
        qpos = torch.from_numpy(np.tile(model.qpos0, (len(frames), 1)))
        encode(tuple(x.cpu().double() for x in motion.ref_state(frames)), qpos, torch.zeros(len(frames), model.nv, dtype=torch.float64), idx)
        depths = np.zeros(len(frames))
        for row, q in enumerate(qpos.numpy()):
            data.qpos[:] = q
            mujoco.mj_kinematics(model, data)
            mujoco.mj_collision(model, data)
            if data.ncon:
                depths[row] = max(0.0, -float(data.contact.dist[: data.ncon].min()))
        return depths

    with ThreadPoolExecutor(threads) as pool:
        return torch.from_numpy(np.concatenate(list(pool.map(clip_depths, range(len(motion.names))))))


def nearest_admissible_frame(depths, clip_start, clip_end, tolerance: float) -> torch.Tensor:
    """Map every frame to the nearest frame of its clip that overlaps at most ``tolerance``.

    The last frame of a clip is never a reset state (it has no successor). Admissible frames map
    to themselves; a clip without any admissible frame is an error, not a silent fallback.
    """
    if not bool(torch.isfinite(depths).all()) or bool((depths < 0).any()):
        raise ValueError("Penetration depths must be finite and nonnegative")
    result = torch.empty(len(depths), dtype=torch.long)
    for start, end in zip(clip_start.tolist(), clip_end.tolist(), strict=True):
        valid = (depths[start:end] <= tolerance).nonzero().flatten() + start
        if not len(valid):
            raise ValueError(f"The clip starting at frame {start} has no admissible reset frame")
        frames = torch.arange(start, end + 1)
        right = torch.searchsorted(valid, frames).clamp(max=len(valid) - 1)
        left = (right - 1).clamp_min(0)
        nearer_left = (frames - valid[left]).abs() <= (frames - valid[right]).abs()
        result[start : end + 1] = torch.where(nearer_left, valid[left], valid[right])
    return result


def build_reset_guard(motion, robot, registry, folder, tolerance: float):
    """(frame map, per-frame depths, summary) for the selected clips; tensors on the motion device."""
    depths = reference_depths(motion, robot, registry, folder)
    mapping = nearest_admissible_frame(depths, motion.clip_start.cpu(), motion.clip_end.cpu(), tolerance)
    nonterminal = torch.ones(len(depths), dtype=torch.bool)
    nonterminal[motion.clip_end.cpu()] = False
    admitted = (depths <= tolerance) & nonterminal
    per_clip = torch.zeros(len(motion.names), dtype=torch.long).index_add_(0, motion.clip_id.cpu(), admitted.long())
    summary = dict(tolerance_m=tolerance, frames=len(depths), admitted_frames=int(admitted.sum()),
                   max_depth_m=float(depths.max()), min_admitted_per_clip=int(per_clip.min()),
                   median_admitted_per_clip=float(per_clip.float().median()))  # fmt: skip
    return mapping.to(motion.device), depths.float().to(motion.device), summary
