"""Sample Weave clips into one compact archive in this repository's joint order.

A split is a folder of Weave archives under ``data/``: the kinematic references (``test``) or the
trajectories Weave's policy executed in simulation for the same clips (``rollout/test``). Both
have the same fields; a rollout is named after its reference clip with ROLLOUT_SUFFIX appended.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

# Isaac Lab's breadth-first body order of the G1 + Inspire articulation (Weave env_cfg.py). The
# archives store joints in the same order: joint i moves body i + 1.
ISAAC_BODIES = (
    "pelvis", "left_hip_pitch_link", "right_hip_pitch_link", "waist_yaw_link", "left_hip_roll_link",
    "right_hip_roll_link", "waist_roll_link", "left_hip_yaw_link", "right_hip_yaw_link", "torso_link",
    "left_knee_link", "right_knee_link", "left_shoulder_pitch_link", "right_shoulder_pitch_link",
    "left_ankle_pitch_link", "right_ankle_pitch_link", "left_shoulder_roll_link", "right_shoulder_roll_link",
    "left_ankle_roll_link", "right_ankle_roll_link", "left_shoulder_yaw_link", "right_shoulder_yaw_link",
    "left_elbow_link", "right_elbow_link", "left_wrist_roll_link", "right_wrist_roll_link",
    "left_wrist_pitch_link", "right_wrist_pitch_link", "left_wrist_yaw_link", "right_wrist_yaw_link",
    "L_index_proximal", "L_middle_proximal", "L_pinky_proximal", "L_ring_proximal", "L_thumb_proximal_base",
    "R_index_proximal", "R_middle_proximal", "R_pinky_proximal", "R_ring_proximal", "R_thumb_proximal_base",
    "L_index_intermediate", "L_middle_intermediate", "L_pinky_intermediate", "L_ring_intermediate",
    "L_thumb_proximal", "R_index_intermediate", "R_middle_intermediate", "R_pinky_intermediate",
    "R_ring_intermediate", "R_thumb_proximal", "L_thumb_intermediate", "R_thumb_intermediate",
    "L_thumb_distal", "R_thumb_distal",
)  # fmt: skip


ROLLOUT_SUFFIX = "__policy_rollout"

# A body joint further than this outside its limit marks a corrupted reference (an angle that
# wrapped through pi), not an overshoot that the loader's projection onto the limits can absorb.
MAX_BODY_LIMIT_EXCESS = 0.2


def select_clips(names, valid, quota: int, rng) -> list[int]:
    """Choose ``quota`` valid clips of one object, covering as many base interactions as possible.

    Clip names are ``<subject>_<object>_<id>_v<k>``: the variants of one base clip share the
    interaction and differ in the approach. Each round takes one unused variant of every base
    clip in a shuffled order, so no interaction repeats before all of them have been used.
    """
    variants: dict[str, list[int]] = {}
    for index, clip in enumerate(names):
        if valid[index]:
            variants.setdefault(re.sub(r"_v\d+$", "", str(clip)), []).append(index)
    if quota > sum(map(len, variants.values())):
        raise ValueError(f"Only {sum(map(len, variants.values()))} valid clips for a quota of {quota}")
    pools = {base: list(rng.permutation(ids)) for base, ids in sorted(variants.items())}
    chosen = []
    while len(chosen) < quota:
        for base in rng.permutation(sorted(pools)):
            if pools[base] and len(chosen) < quota:
                chosen.append(int(pools[base].pop()))
    return sorted(chosen)


def prepare_dataset(weave: Path, output: Path, robot, split: str = "test", count: int = 600, seed: int = 0) -> dict:
    """Write ``motion.npz`` and ``manifest.json`` with an object-balanced, seeded clip sample.

    Stored per frame: pelvis pose, joint positions in ``robot.joint_names`` order, object pose and
    Weave's per-body contact labels (references: +1 contact, 0 neutral, -1 separated; rollouts: 1
    where the measured contact force exceeds 1 N, else 0). Velocities and body poses are recomputed
    at load time, so the archive stays small and self-consistent. Clips keep their reference names.
    """
    import torch

    from ..kinematics import Kinematics
    from ..rotations import quat_to_mat

    files = sorted((Path(weave) / "data" / split).glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No Weave archives under {Path(weave) / 'data' / split}")
    quota = {i: count // len(files) for i in range(len(files))}
    for i in np.random.default_rng(seed).permutation(len(files))[: count % len(files)]:
        quota[int(i)] += 1

    model = robot.model
    isaac_joints = [model.joint(model.body_jntadr[model.body(b).id]).name for b in ISAAC_BODIES[1:]]
    order = [isaac_joints.index(n) for n in robot.joint_names]
    kin = Kinematics(robot, bodies=ISAAC_BODIES)
    lo, hi = robot.joint_limits.T
    out = {k: [] for k in ("root_pos", "root_quat", "joint_pos", "object_pos", "object_quat", "contact_label")}
    clip_names, clip_objects, lengths, sources, per_object, rejected = [], [], [], {}, {}, []
    fk_error, limit_excess = 0.0, 0.0
    for index, file in enumerate(files):
        with np.load(file, allow_pickle=False) as data:
            objects = set(data["object_names"].tolist())
            if len(objects) != 1 or int(data["fps"][0]) != 50:
                raise ValueError(f"{file.name}: expected one object at 50 Hz")
            name, clips = objects.pop(), np.array([str(clip).removesuffix(ROLLOUT_SUFFIX) for clip in data["motion_names"]])
            starts = np.concatenate(([0], np.cumsum(data["motion_lengths"])))
            arrays = {k: data[k] for k in ("body_pos_w", "body_quat_w", "object_pos_w", "object_quat_w", "contact_label")}
            joint_pos = data["joint_pos"][:, order]
        excess = np.maximum(lo - joint_pos, joint_pos - hi)
        body_excess = np.maximum.reduceat(excess[:, robot.body_idx].max(-1), starts[:-1])
        valid = body_excess <= MAX_BODY_LIMIT_EXCESS
        rejected += [str(clip) for clip in clips[~valid]]
        chosen = select_clips(clips, valid, quota[index], np.random.default_rng([seed, index]))
        sources[file.name] = hashlib.sha256(file.read_bytes()).hexdigest()
        per_object[name] = len(chosen)
        for clip in chosen:
            rows = slice(int(starts[clip]), int(starts[clip + 1]))
            q = joint_pos[rows]
            root_pos, root_quat = arrays["body_pos_w"][rows, 0], arrays["body_quat_w"][rows, 0]
            # The stored body poses must be reproduced by this repository's model and joint order.
            zero = torch.zeros(len(q), 3)
            fk = kin.forward(torch.from_numpy(root_pos), quat_to_mat(torch.from_numpy(root_quat)), zero, zero,
                             torch.from_numpy(q), torch.zeros(len(q), robot.num_joints))[0]  # fmt: skip
            fk_error = max(fk_error, float((fk - torch.from_numpy(arrays["body_pos_w"][rows])).norm(dim=-1).max()))
            limit_excess = max(limit_excess, float(excess[rows].max()))
            out["root_pos"].append(root_pos), out["root_quat"].append(root_quat), out["joint_pos"].append(q)
            out["object_pos"].append(arrays["object_pos_w"][rows])
            out["object_quat"].append(arrays["object_quat_w"][rows])
            out["contact_label"].append(arrays["contact_label"][rows].astype(np.int8))
            clip_names.append(str(clips[clip])), clip_objects.append(name), lengths.append(len(q))
    if fk_error > 2e-3:
        raise ValueError(f"FK disagrees with the recorded body positions by {fk_error:.4f} m")

    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "motion.npz", **{k: np.concatenate(v) for k, v in out.items()},
        clip_names=np.array(clip_names), clip_objects=np.array(clip_objects), clip_lengths=np.array(lengths),
        joint_names=np.array(robot.joint_names), label_bodies=np.array(ISAAC_BODIES), fps=np.array(50))  # fmt: skip
    manifest = dict(
        format_version=1, source="Weave motions (appolyn/Weave)", split=split, seed=seed,
        clips=len(clip_names), frames=int(sum(lengths)), hours=sum(lengths) / 50 / 3600,
        per_object=per_object, source_sha256=sources, rejected_clips=rejected,
        fk_max_error_m=fk_error, joint_limit_max_excess_rad=max(limit_excess, 0.0),
        motion_sha256=hashlib.sha256((output / "motion.npz").read_bytes()).hexdigest(), clip_names=clip_names)  # fmt: skip
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
