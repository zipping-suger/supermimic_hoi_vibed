"""Differentiable forward kinematics of the robot in torch, built from its MJCF.

The same FK gives tracked-body poses and velocities for the reference motion, for simulator
states and for world-model predictions, so the analytic reward is identical on all three.
"""

from __future__ import annotations

import mujoco
import numpy as np
import torch

from .robot import Robot
from .rotations import axis_angle_to_mat, mm3, mv3, quat_to_mat


class Kinematics:
    def __init__(self, robot: Robot, device: str = "cpu", bodies: tuple[str, ...] | None = None):
        """``bodies``: output bodies in order (default: ``robot.tracked_bodies``)."""
        m = robot.model
        joint_of = {n: i for i, n in enumerate(robot.joint_names)}
        names, parent, joint, axes, anchors, offsets_p, offsets_q = [], [], [], [], [], [], []
        local = {}
        for b in range(1, m.nbody):
            if m.body_jntnum[b] > 1:
                raise ValueError(f"FK requires at most one joint per body: {m.body(b).name}")
            j = m.body_jntadr[b] if m.body_jntnum[b] else -1
            hinge = j >= 0 and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
            if j >= 0 and not hinge and m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE:
                raise ValueError(f"FK supports hinge joints only: {m.body(b).name}")
            if j >= 0 and not hinge:
                parent.append(-1)  # the free root
            else:
                parent.append(local[m.body_parentid[b]])
            local[b] = len(names)
            names.append(m.body(b).name)
            joint.append(joint_of[m.joint(j).name] if hinge else -1)
            axes.append(m.jnt_axis[j].copy() if hinge else np.zeros(3))
            anchors.append(m.jnt_pos[j].copy() if hinge else np.zeros(3))
            offsets_p.append(m.body_pos[b].copy())
            offsets_q.append(m.body_quat[b].copy())
        as_tensor = lambda x: torch.tensor(np.array(x), dtype=torch.float32, device=device)  # noqa: E731
        self.robot, self.device = robot, device
        self.body_names, self.parent, self.joint = names, parent, joint
        self.axes, self.anchors, self.off_p = as_tensor(axes), as_tensor(anchors), as_tensor(offsets_p)
        self.off_R = quat_to_mat(as_tensor(offsets_q))
        self.output = [names.index(n) for n in (robot.tracked_bodies if bodies is None else bodies)]
        needed = set()
        for body in self.output:
            while body >= 0:
                needed.add(body)
                body = parent[body]
        self.order = tuple(i for i in range(len(names)) if i in needed)
        limits = torch.tensor(robot.joint_limits, device=device)
        self.q_lo, self.q_hi = limits[:, 0], limits[:, 1]

    def forward(self, p, R, v, w, q, qd):
        """World positions, rotations, linear (link-origin) and angular velocities of the output bodies.

        p, R, v, w: root position (..., 3), rotation (..., 3, 3), world linear/angular velocity.
        q, qd: joint positions/velocities (..., J) in ``robot.joint_names`` order.
        """
        P, Rs, V, W = {}, {}, {}, {}
        for i in self.order:
            par = self.parent[i]
            if par < 0:
                P[i], Rs[i], V[i], W[i] = p, R, v, w
                continue
            j = self.joint[i]
            Rp = Rs[par]
            pi = P[par] + mv3(Rp, self.off_p[i])
            Ri = mm3(Rp, self.off_R[i])
            wi = W[par]
            vi = V[par] + torch.cross(W[par], pi - P[par], dim=-1)
            if j >= 0:
                frame = Ri
                Ri = mm3(frame, axis_angle_to_mat(self.axes[i], q[..., j]))
                anchor = pi + mv3(frame, self.anchors[i])
                pi = anchor - mv3(Ri, self.anchors[i])
                spin = mv3(frame, self.axes[i]) * qd[..., j : j + 1]
                wi = wi + spin
                # Linear velocity of the body origin: parent motion about the anchor plus the joint's spin.
                vi = V[par] + torch.cross(W[par], pi - P[par], dim=-1) + torch.cross(spin, pi - anchor, dim=-1)
            P[i], Rs[i], V[i], W[i] = pi, Ri, vi, wi
        out = self.output
        return (torch.stack([P[i] for i in out], -2), torch.stack([Rs[i] for i in out], -3),
                torch.stack([V[i] for i in out], -2), torch.stack([W[i] for i in out], -2))  # fmt: skip
