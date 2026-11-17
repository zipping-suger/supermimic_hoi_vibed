"""Screen reference frames for interpenetration before they are used as reset states.

Retargeted references overlap the object and the ground in places, and resetting the simulator
into deep overlap produces an explosive separation. How deep is harmless depends on what
overlaps: a finger inside the object is pushed out against a 1 Nm drive, and a foot below the
floor lifts the robot gently, while a torso or an object corner is expelled hard. The overlap of
every frame is therefore measured once with CPU MuJoCo, per kind of collision pair, and compared
with that kind's tolerance. Collection then only indexes a device table that maps each proposed
frame to the nearest admissible frame of the same clip.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import mujoco
import numpy as np
import torch

from .sim import Indexing, encode, make_scene

# Kinds of collision pair, in the column order of the depth table.
OVERLAP_KINDS = ("hand_object", "robot_ground", "other")


def overlap_kind(model: mujoco.MjModel, robot) -> np.ndarray:
    """(ngeom, ngeom) column of OVERLAP_KINDS for every geom pair of a compiled scene."""
    body = [model.body(b).name for b in model.geom_bodyid]
    ground = model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE
    on_robot = np.array([name.startswith("robot/") for name in body])
    hand = np.array([robot.hand_link(name.removeprefix("robot/"))[0] >= 0 for name in body]) & on_robot
    obj = np.array([name.startswith("object/") for name in body])
    kind = np.full((model.ngeom, model.ngeom), OVERLAP_KINDS.index("other"))
    kind[np.ix_(hand, obj)] = OVERLAP_KINDS.index("hand_object")
    kind[np.ix_(on_robot, ground)] = OVERLAP_KINDS.index("robot_ground")
    return np.minimum(kind, kind.T)  # either geom order; "other" is the last column


@torch.no_grad()
def reference_depths(motion, robot, registry, folder, threads: int = 16) -> torch.Tensor:
    """(frames, kinds): deepest overlap in metres of every reference frame, per kind of collision pair."""
    models = {}
    for k in sorted(set(motion.clip_object.tolist())):
        model = make_scene(robot, registry, folder, [k], 1, "cpu").compile()
        models[k] = (model, overlap_kind(model, robot))

    def clip_depths(clip: int) -> np.ndarray:
        model, kind = models[int(motion.clip_object[clip])]
        data, idx = mujoco.MjData(model), Indexing(model, robot)
        frames = torch.arange(int(motion.clip_start[clip]), int(motion.clip_end[clip]) + 1, device=motion.device)
        qpos = torch.from_numpy(np.tile(model.qpos0, (len(frames), 1)))
        encode(tuple(x.cpu().double() for x in motion.ref_state(frames)), qpos, torch.zeros(len(frames), model.nv, dtype=torch.float64), idx)
        depths = np.zeros((len(frames), len(OVERLAP_KINDS)))
        for row, q in enumerate(qpos.numpy()):
            data.qpos[:] = q
            mujoco.mj_kinematics(model, data)
            mujoco.mj_collision(model, data)
            contact = data.contact[: data.ncon]
            np.maximum.at(depths[row], kind[contact.geom[:, 0], contact.geom[:, 1]], -contact.dist)
        return depths

    with ThreadPoolExecutor(threads) as pool:
        return torch.from_numpy(np.concatenate(list(pool.map(clip_depths, range(len(motion.names))))))


def nearest_admissible_frame(admissible, clip_start, clip_end) -> torch.Tensor:
    """Map every frame to the nearest admissible frame of its clip.

    The last frame of a clip is never a reset state (it has no successor). Admissible frames map
    to themselves; a clip without any admissible frame is an error, not a silent fallback.
    """
    result = torch.empty(len(admissible), dtype=torch.long)
    for start, end in zip(clip_start.tolist(), clip_end.tolist(), strict=True):
        valid = admissible[start:end].nonzero().flatten() + start
        if not len(valid):
            raise ValueError(f"The clip starting at frame {start} has no admissible reset frame")
        frames = torch.arange(start, end + 1)
        right = torch.searchsorted(valid, frames).clamp(max=len(valid) - 1)
        left = (right - 1).clamp_min(0)
        nearer_left = (frames - valid[left]).abs() <= (frames - valid[right]).abs()
        result[start : end + 1] = torch.where(nearer_left, valid[left], valid[right])
    return result


def build_reset_guard(motion, robot, registry, folder, tolerances: dict[str, float]):
    """(frame map, per-frame depths (frames, kinds), summary); tensors on the motion device.

    ``tolerances``: deepest admitted overlap in metres for every kind in OVERLAP_KINDS.
    """
    depths = reference_depths(motion, robot, registry, folder)
    if not bool(torch.isfinite(depths).all()) or bool((depths < 0).any()):
        raise ValueError("Penetration depths must be finite and nonnegative")
    limit = torch.tensor([tolerances[kind] for kind in OVERLAP_KINDS], dtype=depths.dtype)
    admissible = (depths <= limit).all(-1)
    mapping = nearest_admissible_frame(admissible, motion.clip_start.cpu(), motion.clip_end.cpu())
    admissible[motion.clip_end.cpu()] = False
    per_clip = torch.zeros(len(motion.names), dtype=torch.long).index_add_(0, motion.clip_id.cpu(), admissible.long())
    in_contact = (motion.hand_intent(torch.arange(motion.num_frames, device=motion.device)).sum(-1) > 0).cpu()
    summary = dict(tolerance_m=dict(tolerances), frames=len(depths), admitted_frames=int(admissible.sum()),
                   contact_frames=int(in_contact.sum()), admitted_contact_frames=int((admissible & in_contact).sum()),
                   max_depth_m=dict(zip(OVERLAP_KINDS, depths.amax(0).tolist())),
                   min_admitted_per_clip=int(per_clip.min()), median_admitted_per_clip=float(per_clip.float().median()))  # fmt: skip
    return mapping.to(motion.device), depths.float().to(motion.device), summary
