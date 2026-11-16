"""World model, contact-reward head, actor (motion encoder + decoder) and critic."""

from __future__ import annotations

import torch
import torch.nn as nn

from .rotations import mat_to_6d, mm3, mv3, so3_exp, so3_log, yaw_mat

# bf16 autocast for MLP matmuls only; state integration, FK and rewards stay fp32. Set by the runner.
AMP_DTYPE: torch.dtype | None = None


def mlp(inp: int, hidden, out: int, act=nn.ELU, layer_norm: bool = False) -> nn.Sequential:
    layers, d = [], inp
    for h in hidden:
        layers.append(nn.Linear(d, h))
        if layer_norm:
            layers.append(nn.LayerNorm(h))
        layers.append(act())
        d = h
    layers.append(nn.Linear(d, out))
    return nn.Sequential(*layers)


def run_mlp(net: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if AMP_DTYPE is None:
        return net(x)
    with torch.autocast("cuda", dtype=AMP_DTYPE):
        return net(x).float()


class RunningNorm(nn.Module):
    """Running mean/std normalizer, updated from data without gradients and frozen after ``max_count`` rows.

    Updates are fixed-shape and free of host reads (the freeze is a device-side select), so they
    can run inside CUDA graphs.
    """

    CHUNK_ROWS = 16384  # bounds the scratch memory of an update on rollout-sized batches

    def __init__(self, dim: int, max_count: float = 2e8, eps: float = 1e-2):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.zeros(()))
        self.max_count, self.eps = max_count, eps

    @torch.no_grad()
    def update(self, x: torch.Tensor, valid: torch.Tensor | None = None):
        """Merge the rows of ``x`` (..., dim) where ``valid`` (...) is true (default: all rows)."""
        x = x.reshape(-1, x.shape[-1]).float()
        valid = torch.ones_like(x[:, :1], dtype=torch.bool) if valid is None else valid.reshape(-1, 1)
        n = valid.float().sum()
        # Two passes keep the variance centered; rejected rows may hold NaNs and are zeroed first.
        total = torch.zeros_like(self.mean)
        for part, mask in zip(x.split(self.CHUNK_ROWS), valid.split(self.CHUNK_ROWS)):
            total += torch.where(mask, part, 0.0).sum(0)
        mean = total / n.clamp_min(1)
        square = torch.zeros_like(self.var)
        for part, mask in zip(x.split(self.CHUNK_ROWS), valid.split(self.CHUNK_ROWS)):
            square += torch.where(mask, (torch.where(mask, part, 0.0) - mean).square(), 0.0).sum(0)
        var = square / n.clamp_min(1)
        count = self.count + n
        delta = mean - self.mean
        new_mean = self.mean + delta * n / count.clamp_min(1)
        new_var = (self.var * self.count + var * n + delta.square() * self.count * n / count.clamp_min(1)) / count.clamp_min(1)
        live = (self.count < self.max_count) & (n > 0)
        self.mean.copy_(torch.where(live, new_mean, self.mean))
        self.var.copy_(torch.where(live, new_var, self.var))
        self.count.copy_(torch.where(live, count, self.count))

    def std(self):
        return torch.sqrt(self.var + self.eps**2)

    def forward(self, x):
        return (x - self.mean) / self.std()


def transition_features(s, u, key, context, surface):
    """Heading- and position-invariant inputs shared by the world model and the contact-reward head.
    ``surface``: the MDP's hand-to-surface features of ``s``."""
    p, R, v, w, q, qd, op, oR, ov, ow = s
    Ht, Rt = yaw_mat(R).transpose(-1, -2), R.transpose(-1, -2)
    return torch.cat((
        p[..., 2:3], mat_to_6d(mm3(Ht, R)), mv3(Rt, v), mv3(Rt, w), q, qd,
        mv3(Ht, op - p), mat_to_6d(mm3(Ht, oR)), mv3(Ht, ov), mv3(oR.transpose(-1, -2), ow),
        u, key, context, surface), -1)  # fmt: skip


def feature_dim(J: int, key_dim: int, context_dim: int, surface_dim: int) -> int:
    return (13 + 2 * J) + 15 + J + key_dim + context_dim + surface_dim


class WorldModel(nn.Module):
    """Deterministic coupled robot/object dynamics.

    The network predicts frame-local velocity increments plus small position corrections on top
    of a semi-implicit drift: positions follow the old and the predicted new velocities. Increment
    layout (``delta_dim``): root [dp, dtheta, dv, dw], joints [dq, dqd], object [dp, dtheta, dv, dw].
    Root quantities use the root's yaw/body frames; the object's rotation and angular velocity use
    its own frame, its position and velocity the root's yaw frame.
    """

    DT = 0.02
    # Weight of the new velocity in the position drift: four semi-implicit substeps with a linearly
    # changing velocity give (n + 1) / (2 n) = 0.625.
    DRIFT_W = 0.625

    def __init__(self, J: int, key_dim: int, context_dim: int, surface_dim: int, hidden=(1024, 1024, 1024)):
        super().__init__()
        self.J = J
        self.in_dim, self.delta_dim = feature_dim(J, key_dim, context_dim, surface_dim), 24 + 2 * J
        self.net = mlp(self.in_dim, list(hidden), self.delta_dim, act=nn.SiLU, layer_norm=True)
        self.in_norm = RunningNorm(self.in_dim)
        self.out_norm = RunningNorm(self.delta_dim, eps=1e-4)  # scale of the regression targets
        self.error_norm = RunningNorm(self.delta_dim, eps=1e-4)  # scale of the plain increments, for the loss
        rigid = [0.003] * 3 + [0.005] * 3 + [0.02] * 3 + [0.05] * 3
        self.register_buffer("error_floor", torch.tensor(rigid + [0.003] * J + [0.05] * J + rigid))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def _split(self, d):
        return d.split((3, 3, 3, 3, self.J, self.J, 3, 3, 3, 3), -1)

    def increments(self, s, s1):
        """Frame-local change from ``s`` to ``s1`` in the frames of ``s``."""
        p, R, v, w, q, qd, op, oR, ov, ow = s
        p1, R1, v1, w1, q1, qd1, op1, oR1, ov1, ow1 = s1
        Ht, Rt, oRt = yaw_mat(R).transpose(-1, -2), R.transpose(-1, -2), oR.transpose(-1, -2)
        return torch.cat((
            mv3(Ht, p1 - p), so3_log(mm3(Rt, R1)), mv3(Rt, v1 - v), mv3(Rt, w1 - w), q1 - q, qd1 - qd,
            mv3(Ht, op1 - op), so3_log(mm3(oRt, oR1)), mv3(Ht, ov1 - ov), mv3(oRt, ow1 - ow)), -1)  # fmt: skip

    def _drift(self, s, v1, w1, qd1, ov1, ow1):
        """Position change implied by the old and new velocities, in the increment frames."""
        p, R, v, w, q, qd, op, oR, ov, ow = s
        Ht, Rt, oRt = yaw_mat(R).transpose(-1, -2), R.transpose(-1, -2), oR.transpose(-1, -2)
        a, b = self.DT * (1 - self.DRIFT_W), self.DT * self.DRIFT_W
        return (mv3(Ht, a * v + b * v1), mv3(Rt, a * w + b * w1), a * qd + b * qd1,
                mv3(Ht, a * ov + b * ov1), mv3(oRt, a * ow + b * ow1))  # fmt: skip

    def targets(self, s, s1):
        """Regression targets: the increments with the drift removed from the position parts."""
        dp, dth, dv, dw, dq, dqd, odp, odth, odv, odw = self._split(self.increments(s, s1))
        gp, gth, gq, ogp, ogth = self._drift(s, s1[2], s1[3], s1[5], s1[8], s1[9])
        return torch.cat((dp - gp, dth - gth, dv, dw, dq - gq, dqd, odp - ogp, odth - ogth, odv, odw), -1)

    def integrate(self, s, d):
        """Next state from denormalized network outputs (the layout of ``targets``)."""
        p, R, v, w, q, qd, op, oR, ov, ow = s
        dp, dth, dv, dw, dq, dqd, odp, odth, odv, odw = self._split(d)
        H = yaw_mat(R)
        v1, w1, qd1 = v + mv3(R, dv), w + mv3(R, dw), qd + dqd
        ov1, ow1 = ov + mv3(H, odv), ow + mv3(oR, odw)
        gp, gth, gq, ogp, ogth = self._drift(s, v1, w1, qd1, ov1, ow1)
        return (p + mv3(H, dp + gp), mm3(R, so3_exp(dth + gth)), v1, w1, q + dq + gq, qd1,
                op + mv3(H, odp + ogp), mm3(oR, so3_exp(odth + ogth)), ov1, ow1)  # fmt: skip

    def forward(self, s, u, key, context, surface):
        """Next state for normalized PD targets ``u``."""
        out = run_mlp(self.net, self.in_norm(transition_features(s, u, key, context, surface)))
        return self.integrate(s, out * self.out_norm.std() + self.out_norm.mean)

    def error(self, predicted, target):
        """Frame-local error of a predicted state, in the true state's frames."""
        return self.increments(target, predicted)


class ContactRewardHead(nn.Module):
    """Predicts the ten per-finger force scores of the upcoming transition, separately from the dynamics."""

    def __init__(self, J: int, key_dim: int, context_dim: int, surface_dim: int, hidden=(256, 128)):
        super().__init__()
        dim = feature_dim(J, key_dim, context_dim, surface_dim)
        self.norm = RunningNorm(dim)
        self.net = mlp(dim, list(hidden), 10, act=nn.SiLU, layer_norm=True)
        nn.init.normal_(self.net[-1].weight, std=0.01)
        nn.init.constant_(self.net[-1].bias, -2.0)

    def forward(self, s, u, key, context, surface):
        return run_mlp(self.net, self.norm(transition_features(s, u, key, context, surface))).sigmoid()


class Actor(nn.Module):
    """a = dec(enc(goal), o_{t-H+1:t}); deterministic, latent bounded by tanh."""

    def __init__(self, obs_dim: int, goal_dim: int, act_dim: int, history: int, hidden=(2048, 1024, 512, 256), z_dim: int = 32):
        super().__init__()
        self.enc = mlp(goal_dim, list(hidden), z_dim)
        self.dec = mlp(z_dim + obs_dim * history, list(hidden), act_dim)
        with torch.no_grad():
            self.dec[-1].weight.mul_(0.01)
            self.dec[-1].bias.zero_()
        self.obs_norm = RunningNorm(obs_dim)
        self.goal_norm = RunningNorm(goal_dim)

    def encode(self, goal):
        """Motion latent from reference-only goal features."""
        return torch.tanh(run_mlp(self.enc, self.goal_norm(goal)))

    def forward(self, obs_history, goal, z=None):
        """``obs_history``: (B, history, obs_dim), newest last. ``z``: optional precomputed ``encode(goal)``."""
        z = self.encode(goal) if z is None else z
        return run_mlp(self.dec, torch.cat((z, self.obs_norm(obs_history).flatten(-2)), -1))


class Critic(nn.Module):
    def __init__(self, in_dim: int, hidden=(256, 128)):
        super().__init__()
        self.net = mlp(in_dim, list(hidden), 1)
        self.norm = RunningNorm(in_dim)

    def forward(self, x):
        return run_mlp(self.net, self.norm(x)).squeeze(-1)
