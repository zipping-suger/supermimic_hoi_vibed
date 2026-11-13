"""Policy actions -> joint PD targets.

Body joints follow the next reference pose plus a scaled residual. Finger drivers take absolute
commands in [-1, 1] mapped onto their joint limits and low-pass filtered; the mimic joints follow
their driver's filtered target. Action order is body[29] then finger drivers[12].
"""

from __future__ import annotations

import torch
from torch import nn

from .robot import Robot


class Controller(nn.Module):
    def __init__(self, robot: Robot, device: str = "cpu", dt: float = 0.02, body_action_clip: float = 20.0):
        super().__init__()
        index = lambda x: torch.tensor(x, device=device, dtype=torch.long)  # noqa: E731
        value = lambda x: torch.tensor(x, device=device, dtype=torch.float32)  # noqa: E731
        self.register_buffer("body", index(robot.body_idx))
        self.register_buffer("driver", index(robot.driver_idx))
        self.register_buffer("mimic", index(robot.mimic_idx))
        self.register_buffer("mimic_driver", index(robot.mimic_driver))
        self.register_buffer("mimic_ratio", value(robot.mimic_ratio))
        self.register_buffer("limits", value(robot.joint_limits))
        self.register_buffer("default", value(robot.default_joint_pos))
        self.register_buffer("center", value(robot.control_center))
        self.register_buffer("scale", value(robot.control_scale))
        self.num_body, self.num_hand = len(robot.body_idx), len(robot.driver_idx)
        # First-order low-pass on the finger targets (0.9 per 1/60 s, expressed at the control rate).
        # Weave applies finger targets unfiltered; the filter keeps exploration noise off the fingers.
        self.alpha = 1 - 0.9 ** (60 * dt)
        self.body_action_clip = body_action_clip

    def body_command(self, action, qref):
        """Full normalized body command: the reference skip plus the residual action."""
        body = self.body
        return (qref[..., body] - self.default[body]) / self.scale[body] + action[..., : self.num_body]

    def bound_action(self, action, qref_next):
        """Clip the full body command (reference skip included) and return the bounded residual."""
        skip = (qref_next[..., self.body] - self.default[self.body]) / self.scale[self.body]
        residual = action[..., : self.num_body]
        full = residual + skip
        bounded = torch.where(full.abs() <= self.body_action_clip, residual,
                              full.clamp(-self.body_action_clip, self.body_action_clip) - skip)  # fmt: skip
        return torch.cat((bounded, action[..., self.num_body :]), -1)

    def seed(self, q):
        """Filter state that holds the fingers at joint positions ``q``."""
        lo, hi = self.limits[self.driver].unbind(-1)
        return q[..., self.driver].clamp(lo, hi)

    def reference_action(self, qref_next):
        """Zero body residual and the finger command whose unfiltered target is the reference."""
        lo, hi = self.limits[self.driver].unbind(-1)
        hand = 2 * (qref_next[..., self.driver] - lo) / (hi - lo) - 1
        return torch.cat((torch.zeros_like(qref_next[..., self.body]), hand.clamp(-1, 1)), -1)

    def forward(self, action, qref_next, hand_targets, offset):
        """PD targets (..., J) and the new finger filter state (..., 12).

        ``action`` must already be bounded. ``offset`` is the per-joint encoder offset: the policy
        observes q - offset, so every target moves by +offset.
        """
        body = qref_next[..., self.body] + self.scale[self.body] * action[..., : self.num_body]
        lo, hi = self.limits[self.driver].unbind(-1)
        raw = lo + (action[..., self.num_body :].clamp(-1, 1) + 1) * 0.5 * (hi - lo)
        hand = (self.alpha * raw + (1 - self.alpha) * hand_targets).clamp(lo, hi)
        mimic_lo, mimic_hi = self.limits[self.mimic].unbind(-1)
        mimic = (hand[..., self.mimic_driver] * self.mimic_ratio).clamp(mimic_lo, mimic_hi)
        target = torch.zeros_like(qref_next)
        target = target.index_copy(-1, self.body, body).index_copy(-1, self.driver, hand).index_copy(-1, self.mimic, mimic)
        return target + offset, hand

    def effective(self, target):
        """Normalized physical PD target, the world model's action input."""
        return (target - self.center) / self.scale
