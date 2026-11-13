"""Analytic observation, reward and termination heads.

Everything here is tensor arithmetic on a state tuple and a reference frame, so the same code
labels real simulator transitions and differentiates through imagined ones. Contact enters only
as numbers supplied by the caller: measured forces in simulation, the contact-reward head's
finger scores in imagination. Contact is rewarded, never a termination condition.
"""

from __future__ import annotations

import torch

from . import robot as R
from .motion import CONTROL_DT
from .rotations import angle_squared, mat_to_6d, mm3, mv3, yaw_mat

FAILURE_REASONS = ("root_position", "root_tilt", "object_position", "object_tilt")


def chain_product(value, dim):
    """Product over a short dimension as a chain of multiplications (torch.prod's backward reads a
    zero count on the host, which a CUDA graph cannot capture)."""
    factors = value.unbind(dim)
    result = factors[0]
    for factor in factors[1:]:
        result = result * factor
    return result


def force_scores(force, threshold: float, saturation: float):
    """Per-finger contact score in [0, 1] from the mean normal force over a control step."""
    return torch.where(force > threshold, (force / saturation).clamp(0, 1), 0.0)


def required_hand_minimum(per_hand, intent):
    """Worst score among the hands the reference requires; 0 when it requires none."""
    required = torch.where(intent > 0, per_hand, 1.0)
    return torch.minimum(required[..., 0], required[..., 1]) * (intent.sum(-1) > 0)


def contact_score(finger_scores, finger_intent):
    """Mean score of each hand's intended fingers, then the worse of the required hands, so a good
    grasp with one hand cannot stand in for a missing one."""
    intent = finger_intent.unflatten(-1, (2, 5))
    per_hand = (finger_scores.unflatten(-1, (2, 5)) * intent).sum(-1) / intent.sum(-1).clamp_min(1)
    return required_hand_minimum(per_hand, intent.amax(-1))


class HOIMDP:
    def __init__(self, motion, controller, layout, cfg, noise: bool):
        self.motion, self.controller, self.layout, self.cfg, self.noisy = motion, controller, layout, cfg, noise
        self.kin = motion.kin
        device = motion.device
        J, A = controller.limits.shape[0], controller.num_body + controller.num_hand
        self.num_tracked = motion.num_tracked
        self.tips = torch.tensor(motion.tips, device=device)
        self.feet = torch.tensor(motion.feet, device=device)
        margin = 0.05 * (self.kin.q_hi - self.kin.q_lo)
        self.soft_lo, self.soft_hi = self.kin.q_lo + margin, self.kin.q_hi - margin
        # Uniform noise half-widths on the proprioceptive head of the observation.
        self.noise = torch.cat((
            torch.full((J,), R.NOISE_JOINT_POS), torch.full((J,), R.NOISE_JOINT_VEL), torch.full((3,), R.NOISE_ANG_VEL),
            torch.full((3,), R.NOISE_GRAVITY), torch.zeros(A), torch.full((6,), R.NOISE_REF_ORI))).to(device)  # fmt: skip
        self.thresholds = torch.tensor([cfg.term_root_pos, cfg.term_tilt, cfg.term_object_pos, cfg.term_object_tilt], device=device)
        self.temperatures = torch.tensor([0.02, 0.04, 0.02, 0.04], device=device)
        self.max_step_reward = CONTROL_DT * sum(getattr(cfg, name) for name in cfg.BONUSES)

    def fk(self, s):
        return self.kin.forward(*s[:6])

    def add_noise(self, obs):
        if not self.noisy:
            return obs
        n = self.noise.numel()
        return torch.cat((obs[..., :n] + (2 * torch.rand_like(obs[..., :n]) - 1) * self.noise, obs[..., n:]), -1)

    def obs(self, s, f, previous_action, hand_targets, context, fk=None):
        """Noise-free actor observation of state ``s`` at reference frame ``f``."""
        p, Rm, v, w, q, qd, op, oR, ov, ow = s
        m = self.motion
        body_p = (self.fk(s) if fk is None else fk)[0]
        Ht, Rt = yaw_mat(Rm).transpose(-1, -2), Rm.transpose(-1, -2)
        tips = mv3(oR.transpose(-1, -2).unsqueeze(-3), body_p[..., self.tips, :] - op.unsqueeze(-2))
        return torch.cat((
            q - self.layout.offsets(context) - self.controller.default, qd, mv3(Rt, w), -Rm[..., 2, :],
            previous_action, mat_to_6d(mm3(Rt, m.root_R[f])),
            mv3(Ht, op - p), mat_to_6d(mm3(Ht, oR)), mv3(Ht, ov - v), mv3(Ht, ow) * 0.25, tips.flatten(-2),
            hand_targets, m.key(f), m.finger_intent[f],
            # Global tracking error: observable even when robot and object drift together.
            mv3(Ht, m.root_p[f] - p)), -1)  # fmt: skip

    def critic_input(self, s, f, previous_action, hand_targets, context, goal, fk=None):
        # Finger commands beyond [-1, 1] have identical physical effects; do not expose their
        # magnitude to the value function as an actor-to-value shortcut.
        nb = self.controller.num_body
        previous_action = torch.cat((previous_action[..., :nb], previous_action[..., nb:].clamp(-1, 1)), -1)
        fk = self.fk(s) if fk is None else fk
        Ht = yaw_mat(s[1]).transpose(-1, -2)
        body_error = mv3(Ht.unsqueeze(-3), self.motion.body_p[f] - fk[0]).flatten(-2)
        return torch.cat((self.obs(s, f, previous_action, hand_targets, context, fk), mv3(Ht, s[2]),
                          self.motion.root_p[f] - s[0], body_error, context, goal), -1)  # fmt: skip

    def reward(self, s, f, action, previous_action, finger_scores, episode_start, fk=None):
        """Reward of arriving in ``s`` at reference frame ``f`` (the post-transition frame).

        ``finger_scores`` (..., 10) are measured or predicted force scores. ``episode_start`` marks
        the first transition after a reset, whose previous body command is zero.
        """
        m, c, T = self.motion, self.cfg, self.num_tracked
        p, Rm, v, w, q, qd, op, oR, ov, ow = s
        bp, bR, bv, bw = self.fk(s) if fk is None else fk
        e_pos = (bp[..., :T, :] - m.body_p[f][..., :T, :]).square().sum(-1).mean(-1)
        e_ori = angle_squared(bR[..., :T, :, :], m.body_R[f]).mean(-1)
        e_vel = (bv[..., :T, :] - m.body_v[f]).square().sum(-1).mean(-1)
        e_angvel = (bw[..., :T, :] - m.body_w[f]).square().sum(-1).mean(-1)

        finger_intent = m.finger_intent[f]
        hand_intent = finger_intent.unflatten(-1, (2, 5)).amax(-1)
        tips = mv3(oR.transpose(-1, -2).unsqueeze(-3), bp[..., self.tips, :] - op.unsqueeze(-2))
        ref_tips = mv3(m.object_R[f].transpose(-1, -2).unsqueeze(-3), m.body_p[f][..., self.tips, :] - m.object_p[f].unsqueeze(-2))
        tip_error = (tips - ref_tips).square().sum(-1).unflatten(-1, (2, 5)).mean(-1)
        # A Cauchy curve keeps a useful gradient beyond the narrow grasp region. Hands the
        # reference does not have in contact get no positional reward or gradient.
        relative = 1 / (1 + tip_error / c.relative_std**2)
        relative = (relative * hand_intent).sum(-1) / hand_intent.sum(-1).clamp_min(1)

        # Body action rate on the full normalized command (reference skip + residual).
        nb = self.controller.num_body
        previous_frame = torch.maximum(f - 1, m.clip_start[m.clip_id[f]])
        command = self.controller.body_command(action, m.q[f])
        previous_command = self.controller.body_command(previous_action, m.q[previous_frame])
        previous_command = torch.where(episode_start[..., None], 0.0, previous_command)
        hand_rate = (action[..., nb:] - previous_action[..., nb:]).square().mean(-1)
        limits = torch.relu(self.soft_lo - q) + torch.relu(q - self.soft_hi)
        hand_joints = torch.cat((self.controller.driver, self.controller.mimic))
        foot_speed = bv[..., self.feet, :2].square().sum(-1)
        foot_down = torch.sigmoid((R.FOOT_CONTACT_Z - bp[..., self.feet, 2]) / 0.01)
        terms = {
            "body_pos": c.body_pos * torch.exp(-e_pos / c.body_pos_std**2),
            "body_ori": c.body_ori * torch.exp(-e_ori / c.body_ori_std**2),
            "body_velocity": c.body_velocity * torch.exp(-e_vel),
            "body_angular_velocity": c.body_angular_velocity * torch.exp(-e_angvel / c.body_angvel_std**2),
            "root_pos": c.root_pos * torch.exp(-(p - m.root_p[f]).square().sum(-1) / c.body_pos_std**2),
            "root_ori": c.root_ori * torch.exp(-angle_squared(Rm, m.root_R[f]) / c.body_ori_std**2),
            "object_pos": c.object_pos * torch.exp(-(op - m.object_p[f]).square().sum(-1) / c.object_pos_std**2),
            "object_ori": c.object_ori * torch.exp(-angle_squared(oR, m.object_R[f]) / c.object_ori_std**2),
            "object_velocity": c.object_velocity * torch.exp(
                -(ov - m.object_v[f]).square().sum(-1) - (ow - m.object_w[f]).square().sum(-1) / 3.14**2),
            "relative": c.relative * relative,
            "contact": c.contact * contact_score(finger_scores, finger_intent),
            "action_rate": -c.action_rate * ((command - previous_command).square().sum(-1) + hand_rate),
            "joint_limit": -c.joint_limit * (limits[..., self.controller.body].sum(-1) + limits[..., hand_joints].mean(-1)),
            "foot_slip": -c.foot_slip * (foot_speed * foot_down).mean(-1),
            "hand_saturation": -c.hand_saturation * torch.relu(action[..., nb:].abs() - 1).square().mean(-1),
        }  # fmt: skip
        terms = {k: v * CONTROL_DT for k, v in terms.items()}
        return sum(terms.values()), terms

    def errors(self, s, f):
        m = self.motion
        return torch.stack((
            (s[0] - m.root_p[f]).square().sum(-1).add(1e-12).sqrt(), (s[1][..., 2, 2] - m.root_R[f][..., 2, 2]).abs(),
            (s[6] - m.object_p[f]).square().sum(-1).add(1e-12).sqrt(), (s[7][..., 2, 2] - m.object_R[f][..., 2, 2]).abs()), -1)  # fmt: skip

    def hard_termination(self, s, f):
        """(failed, reasons) in simulation; reasons follow FAILURE_REASONS."""
        flags = self.errors(s, f) > self.thresholds
        return flags.any(-1), flags

    def soft_termination(self, s, f):
        """Differentiable surrogate of ``hard_termination`` for imagination."""
        return 1 - chain_product(1 - torch.sigmoid((self.errors(s, f) - self.thresholds) / self.temperatures), -1)
