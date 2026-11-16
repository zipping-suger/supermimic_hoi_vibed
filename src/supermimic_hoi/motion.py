"""Reference library: synchronized robot/object clips at 50 Hz with FK of the tracked bodies."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .kinematics import Kinematics
from .robot import FINGERTIPS, FOOT_BODIES, TRACKED_BODIES, Robot, finger_channel
from .rotations import angular_velocity, central_difference, mat_to_6d, mm3, mv3, quat_to_mat, yaw_mat

CONTROL_DT = 0.02
ROTATION_LENGTH = 0.15  # metres that one radian of object rotation counts for in an object pose step


def pose_step(p0, R0, p1, R1):
    """(..., 6) change of an object pose over one control step: translation, then the world-frame
    rotation scaled to a length. The rotation is the antisymmetric part of R1 R0^T (axis times the
    sine of the angle), which is smooth at rest and within 1 % of the rotation vector for a step."""
    r = mm3(R1, R0.transpose(-1, -2))
    turn = torch.stack((r[..., 2, 1] - r[..., 1, 2], r[..., 0, 2] - r[..., 2, 0], r[..., 1, 0] - r[..., 0, 1]), -1) * 0.5
    return torch.cat((p1 - p0, ROTATION_LENGTH * turn), -1)


class Motion:
    """All selected clips concatenated along the frame axis.

    Robot and object share one world frame and are never re-anchored independently. Joint
    references are projected onto the joint limits before velocities and FK are derived, so every
    reference state is a feasible reset. Contact labels are reference intent, not measurements:
    a finger is intended to touch while any of its links is labelled in contact.
    """

    def __init__(self, dataset: str | Path, robot: Robot, object_names: list[str], device: str = "cpu", clips=()):
        dataset = Path(dataset)
        manifest = json.loads((dataset / "manifest.json").read_text())
        with np.load(dataset / "motion.npz", allow_pickle=False) as data:
            if int(data["fps"]) != round(1 / CONTROL_DT) or tuple(data["joint_names"]) != robot.joint_names:
                raise ValueError("Dataset must be 50 Hz in the robot's joint order; rerun supermimic_hoi.prepare")
            all_names = data["clip_names"].tolist()
            names = list(clips) if clips else all_names
            if len(set(names)) != len(names) or set(names) - set(all_names):
                raise ValueError("Unknown or duplicate clips")
            starts = np.concatenate(([0], np.cumsum(data["clip_lengths"])))
            ids = [all_names.index(n) for n in names]
            rows = np.concatenate([np.arange(starts[i], starts[i + 1]) for i in ids])
            lengths = data["clip_lengths"][ids].tolist()
            objects = data["clip_objects"][ids].tolist()
            arrays = {k: torch.from_numpy(data[k][rows]) for k in
                      ("root_pos", "root_quat", "joint_pos", "object_pos", "object_quat", "contact_label")}  # fmt: skip
            label_bodies = data["label_bodies"].tolist()
        if any(not torch.isfinite(x).all() for x in arrays.values()) or min(lengths) < 2:
            raise ValueError("Reference clips must be finite and at least two frames long")

        self.device, self.names, self.object_names = device, names, list(object_names)
        self.kin = Kinematics(robot, device)
        self.clip_len = torch.tensor(lengths, device=device)
        self.clip_start = torch.cumsum(self.clip_len, 0) - self.clip_len
        self.clip_end = self.clip_start + self.clip_len - 1  # last frame, inclusive
        self.clip_id = torch.repeat_interleave(torch.arange(len(names), device=device), self.clip_len)
        self.clip_object = torch.tensor([self.object_names.index(o) for o in objects], device=device)
        self.num_frames = int(self.clip_len.sum())

        q_raw = arrays["joint_pos"]
        q = q_raw.clamp(self.kin.q_lo.cpu(), self.kin.q_hi.cpu())
        self.projection = dict(frames_changed=int(((q - q_raw).abs() > 1e-6).any(-1).sum()),
                               max_correction_rad=float((q - q_raw).abs().max()))  # fmt: skip
        root_R, object_R = quat_to_mat(arrays["root_quat"]), quat_to_mat(arrays["object_quat"])
        per_clip = lambda fn, x: torch.cat([fn(part, CONTROL_DT) for part in x.split(lengths)])  # noqa: E731
        to = lambda x: x.to(device)  # noqa: E731
        self.root_p, self.root_R, self.q = to(arrays["root_pos"]), to(root_R), to(q)
        self.root_v, self.qd = to(per_clip(central_difference, arrays["root_pos"])), to(per_clip(central_difference, q))
        self.root_w = to(per_clip(angular_velocity, root_R))
        self.object_p, self.object_R = to(arrays["object_pos"]), to(object_R)
        self.object_v = to(per_clip(central_difference, arrays["object_pos"]))
        self.object_w = to(per_clip(angular_velocity, object_R))
        # Reference object step into every frame; zero into a clip's first frame.
        before = (torch.arange(self.num_frames, device=device) - 1).clamp_min(self.clip_start[self.clip_id])
        self.object_step = pose_step(self.object_p[before], self.object_R[before], self.object_p, self.object_R)

        # Tracked bodies come first, then the fingertips; rotations and velocities are only
        # needed for the tracked bodies.
        self.num_tracked = len(TRACKED_BODIES)
        self.tips = list(range(self.num_tracked, self.num_tracked + len(FINGERTIPS)))
        self.feet = [TRACKED_BODIES.index(n) for n in FOOT_BODIES]
        parts = [[], [], [], []]
        with torch.no_grad():
            for chunk in torch.arange(self.num_frames, device=device).split(65536):
                out = self.kin.forward(*self.ref_robot_state(chunk))
                parts[0].append(out[0])
                for store, value in zip(parts[1:], out[1:]):
                    store.append(value[:, : self.num_tracked])
        self.body_p, self.body_R, self.body_v, self.body_w = (torch.cat(x) for x in parts)

        Rt = self.root_R.transpose(-1, -2)
        self.root_v_local, self.root_w_local = mv3(Rt, self.root_v), mv3(Rt, self.root_w)
        channels = torch.tensor([finger_channel(b) for b in label_bodies])
        touching = arrays["contact_label"] > 0
        self.finger_intent = to(torch.stack([touching[:, channels == c].any(-1) for c in range(10)], -1).float())
        self.onehot = torch.eye(len(self.object_names), device=device)
        digest = hashlib.sha256(manifest["motion_sha256"].encode())
        digest.update(json.dumps(names).encode() + robot.joint_limits.tobytes())
        self.signature = digest.hexdigest()

    def ref_robot_state(self, f):
        return self.root_p[f], self.root_R[f], self.root_v[f], self.root_w[f], self.q[f], self.qd[f]

    def ref_state(self, f):
        return (*self.ref_robot_state(f), self.object_p[f], self.object_R[f], self.object_v[f], self.object_w[f])

    def next_frame(self, f):
        return torch.minimum(f + 1, self.clip_end[self.clip_id[f]])

    def key(self, f):
        """One-hot object identity of the clip that frame ``f`` belongs to."""
        return self.onehot[self.clip_object[self.clip_id[f]]]

    def hand_intent(self, f):
        """(..., 2): whether the reference has any finger of the left / right hand in contact."""
        return self.finger_intent[f].unflatten(-1, (2, 5)).amax(-1)

    def goal_dim(self, K: int) -> int:
        return K * (15 + 2 * self.q.shape[-1] + 15 + 10)

    def goal(self, f, K: int):
        """Reference-only goal: K future frames of robot and object relative to the current reference root."""
        end = self.clip_end[self.clip_id[f]]
        future = torch.minimum(f.unsqueeze(-1) + torch.arange(1, K + 1, device=f.device), end.unsqueeze(-1))
        Ht = yaw_mat(self.root_R[f]).transpose(-1, -2).unsqueeze(1)
        features = torch.cat((
            mv3(Ht, self.root_p[future] - self.root_p[f].unsqueeze(1)), mat_to_6d(mm3(Ht, self.root_R[future])),
            self.root_v_local[future], self.root_w_local[future], self.q[future], self.qd[future] * 0.1,
            mv3(Ht, self.object_p[future] - self.root_p[f].unsqueeze(1)), mat_to_6d(mm3(Ht, self.object_R[future])),
            mv3(Ht, self.object_v[future]), mv3(Ht, self.object_w[future]), self.finger_intent[future]), -1)  # fmt: skip
        return features.flatten(-2)

    def pregrasp_frames(self, lead: int = 10, min_steps: int = 10):
        """Per clip, the frames ``lead`` steps before each sustained hand-contact phase begins.

        Returns a padded (clips, max_count) frame table and the per-clip counts; a clip without a
        sustained phase gets its first frame.
        """
        intent = self.hand_intent(torch.arange(self.num_frames, device=self.device)).cpu().numpy() > 0
        choices = []
        for first, last in zip(self.clip_start.tolist(), self.clip_end.tolist()):
            frames = set()
            for side in range(2):
                edges = np.diff(np.concatenate(([False], intent[first : last + 1, side], [False])).astype(int))
                for begin, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
                    if stop - begin >= min_steps:
                        frames.add(max(first, first + int(begin) - lead))
            choices.append(sorted(frames) or [first])
        table = torch.zeros(len(choices), max(map(len, choices)), dtype=torch.long, device=self.device)
        for i, frames in enumerate(choices):
            table[i, : len(frames)] = torch.tensor(frames, device=self.device)
        return table, torch.tensor([len(c) for c in choices], device=self.device)
