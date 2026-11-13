"""Robot domain randomization and the privileged simulation context.

The context seen by the world model, contact-reward head and critic is the object's physical
parameters followed by the normalized robot parameters xi = [friction, encoder offsets, torso CoM
offset]. The actor never sees it. Friction and CoM are drawn once per world, encoder offsets once
per episode.
"""

from __future__ import annotations

import mujoco
import numpy as np
import torch

from .robot import COM_BODY, COM_RANGE, FRICTION_RANGE, JOINT_OFFSET, Robot

OBJECT_CONTEXT = 13  # mass, CoM, inertia, friction (see prepare.objects.CONTEXT_FIELDS)


class ContextLayout:
    def __init__(self, robot: Robot, device: str):
        J = robot.num_joints
        self.friction, self.joint_offset, self.com = slice(0, 1), slice(1, 1 + J), slice(1 + J, 4 + J)
        self.dim = 4 + J
        lo, hi = FRICTION_RANGE
        self.center = torch.tensor([(lo + hi) / 2] + [0.0] * (J + 3), device=device)
        self.scale = torch.tensor([(hi - lo) / 2] + [JOINT_OFFSET] * J + [0.05] * 3, device=device)

    def normalize(self, xi):
        return (xi - self.center) / self.scale

    def offsets(self, context):
        """Encoder offsets in radians from a full context row."""
        xi = context[..., OBJECT_CONTEXT:] * self.scale + self.center
        return xi[..., self.joint_offset]


class Randomizer:
    """Samples xi and applies its simulator part. Disabled: nominal friction 1, no offsets."""

    def __init__(self, robot: Robot, physics, layout: ContextLayout, enabled: bool):
        self.layout, self.enabled, self.physics = layout, enabled, physics
        N, device = physics.N, physics.device
        self.xi = torch.zeros(N, layout.dim, device=device)
        self.xi[:, layout.friction] = 1.0
        self.joint_offset = torch.full((robot.num_joints,), JOINT_OFFSET, device=device)
        if hasattr(physics, "sim"):
            self._apply(robot, physics.sim, N, device)
        elif enabled:
            raise ValueError("Randomization needs the MuJoCo-Warp backend")

    def _apply(self, robot, sim, N, device):
        model = sim.mj_model
        body_names = [model.body(b).name for b in model.geom_bodyid]
        robot_geoms = torch.tensor([i for i, n in enumerate(body_names) if n.startswith("robot/")], device=device)
        ground_geoms = torch.tensor(np.flatnonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE), device=device)
        if not len(ground_geoms):
            raise ValueError("The scene has no ground plane")
        sim.expand_model_fields(("geom_friction", "body_ipos") if self.enabled else ("geom_friction",))
        if self.enabled:
            self.xi[:, self.layout.friction] = torch.empty(N, 1, device=device).uniform_(*FRICTION_RANGE)
            com = torch.stack([torch.empty(N, device=device).uniform_(lo, hi) for lo, hi in COM_RANGE], -1)
            self.xi[:, self.layout.com] = com
            torso = model.body("robot/" + COM_BODY).id
            default = sim.get_default_field("body_ipos")
            default = default[:, torso] if default.ndim == 3 else default[torso]
            sim.model.body_ipos[:, torso] = default + com
        # MuJoCo takes the larger friction of a pair, so the ground sits at the bottom of the range
        # and the robot's own friction decides every robot-ground contact.
        sim.model.geom_friction[:, ground_geoms, 0] = FRICTION_RANGE[0] if self.enabled else 1.0
        sim.model.geom_friction[:, robot_geoms, 0] = self.xi[:, self.layout.friction]

    def resample(self, mask):
        """New encoder offsets for the worlds that start an episode."""
        if self.enabled:
            new = (2 * torch.rand_like(self.xi[:, self.layout.joint_offset]) - 1) * self.joint_offset
            self.xi[:, self.layout.joint_offset] = torch.where(mask[:, None], new, self.xi[:, self.layout.joint_offset])

    def context(self):
        return torch.cat((self.physics.object_context, self.layout.normalize(self.xi)), -1)
