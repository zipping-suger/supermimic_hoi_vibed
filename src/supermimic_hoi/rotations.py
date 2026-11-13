"""Batched rotation utilities. Quaternions are wxyz; matrices act on column vectors.

3x3 products are written elementwise (multiply + sum) so torch.compile fuses them into its
kernels instead of launching one batched BLAS call per rotation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def mm3(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """A @ B for (..., 3, 3) matrices (broadcasting)."""
    return (A.unsqueeze(-1) * B.unsqueeze(-3)).sum(-2)


def mv3(A: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """A @ v for (..., 3, 3) matrices and (..., 3) vectors (broadcasting)."""
    return (A * v.unsqueeze(-2)).sum(-1)


def quat_to_mat(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = F.normalize(q, dim=-1).unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(q.shape[:-1] + (3, 3))  # fmt: skip


def mat_to_quat(m: torch.Tensor) -> torch.Tensor:
    """Robust matrix -> quaternion (wxyz, w >= 0)."""
    m00, m01, m02 = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    m10, m11, m12 = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    m20, m21, m22 = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]
    qw = torch.sqrt(torch.clamp(1 + m00 + m11 + m22, min=1e-10)) / 2
    qx = torch.sqrt(torch.clamp(1 + m00 - m11 - m22, min=1e-10)) / 2
    qy = torch.sqrt(torch.clamp(1 - m00 + m11 - m22, min=1e-10)) / 2
    qz = torch.sqrt(torch.clamp(1 - m00 - m11 + m22, min=1e-10)) / 2
    candidates = torch.stack(
        (
            torch.stack((qw, (m21 - m12) / (4 * qw), (m02 - m20) / (4 * qw), (m10 - m01) / (4 * qw)), -1),
            torch.stack(((m21 - m12) / (4 * qx), qx, (m01 + m10) / (4 * qx), (m02 + m20) / (4 * qx)), -1),
            torch.stack(((m02 - m20) / (4 * qy), (m01 + m10) / (4 * qy), qy, (m12 + m21) / (4 * qy)), -1),
            torch.stack(((m10 - m01) / (4 * qz), (m02 + m20) / (4 * qz), (m12 + m21) / (4 * qz), qz), -1),
        ),
        dim=-2,
    )
    best = torch.stack((qw, qx, qy, qz), -1).argmax(-1)
    q = torch.gather(candidates, -2, best[..., None, None].expand(best.shape + (1, 4))).squeeze(-2)
    q = F.normalize(q, dim=-1)
    return q * torch.where(q[..., :1] < 0, -1.0, 1.0)


def axis_angle_to_mat(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rodrigues for a fixed unit axis (3,) and angles (...)."""
    x, y, z = axis.unbind(-1)
    c, s = torch.cos(angle), torch.sin(angle)
    C = 1 - c
    return torch.stack(
        (
            c + x * x * C, x * y * C - z * s, x * z * C + y * s,
            y * x * C + z * s, c + y * y * C, y * z * C - x * s,
            z * x * C - y * s, z * y * C + x * s, c + z * z * C,
        ),
        dim=-1,
    ).reshape(angle.shape + (3, 3))  # fmt: skip


def skew(v: torch.Tensor) -> torch.Tensor:
    x, y, z = v.unbind(-1)
    o = torch.zeros_like(x)
    return torch.stack((o, -z, y, z, o, -x, -y, x, o), -1).reshape(v.shape[:-1] + (3, 3))


def so3_exp(v: torch.Tensor) -> torch.Tensor:
    """Rotation vector (..., 3) -> matrix, smooth at zero."""
    theta2 = (v * v).sum(-1, keepdim=True)
    theta = torch.sqrt(theta2 + 1e-12)
    small = theta2 < 1e-8
    a = torch.where(small, 1 - theta2 / 6, torch.sin(theta) / theta)
    b = torch.where(small, 0.5 - theta2 / 24, (1 - torch.cos(theta)) / (theta2 + 1e-12))
    K = skew(v)
    eye = torch.eye(3, device=v.device, dtype=v.dtype).expand(K.shape)
    return eye + a[..., None] * K + b[..., None] * mm3(K, K)


def so3_log(R: torch.Tensor) -> torch.Tensor:
    """Matrix -> rotation vector via the quaternion (stable near 0 and pi)."""
    q = mat_to_quat(R)
    v = q[..., 1:]
    sin_half = torch.linalg.norm(v, dim=-1, keepdim=True)
    half = torch.atan2(sin_half, q[..., :1])
    scale = torch.where(sin_half < 1e-6, 2.0 / torch.clamp(q[..., :1], min=1e-6), 2 * half / (sin_half + 1e-12))
    return v * scale


def angle_squared(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Squared geodesic angle between rotations.

    atan2 avoids acos's infinite derivative at zero error; the epsilon only protects the norm at
    the origin. Near pi the metric is necessarily nonsmooth.
    """
    r = mm3(a.transpose(-1, -2), b)
    vee = torch.stack((r[..., 2, 1] - r[..., 1, 2], r[..., 0, 2] - r[..., 2, 0], r[..., 1, 0] - r[..., 0, 1]), -1) * 0.5
    sin = (vee.square().sum(-1) + 1e-12).sqrt()
    cos = (r.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) * 0.5
    return torch.atan2(sin, cos).square()


def yaw_mat(R: torch.Tensor) -> torch.Tensor:
    """Heading (yaw-only) rotation of R, from the projected x axis."""
    fx, fy = R[..., 0, 0], R[..., 1, 0]
    n = torch.sqrt(fx * fx + fy * fy + 1e-9)
    c, s = fx / n, fy / n
    o, one = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack((c, -s, o, s, c, o, o, o, one), -1).reshape(R.shape[:-2] + (3, 3))


def mat_to_6d(R: torch.Tensor) -> torch.Tensor:
    """First two columns, flattened (6D continuous rotation representation)."""
    return R[..., :, :2].reshape(R.shape[:-2] + (6,))


def central_difference(x: torch.Tensor, dt: float) -> torch.Tensor:
    """Time derivative along dim 0: central inside, one-sided at both ends."""
    g = torch.zeros_like(x)
    g[1:-1] = (x[2:] - x[:-2]) / (2 * dt)
    g[0] = (x[1] - x[0]) / dt
    g[-1] = (x[-1] - x[-2]) / dt
    return g


def angular_velocity(R: torch.Tensor, dt: float) -> torch.Tensor:
    """World angular velocity of a rotation trajectory (T, 3, 3) by finite differences."""
    w = torch.zeros(R.shape[0], 3, dtype=R.dtype, device=R.device)
    w[1:-1] = so3_log(mm3(R[2:], R[:-2].transpose(-1, -2))) / (2 * dt)
    w[0] = so3_log(mm3(R[1], R[0].transpose(-1, -2))) / dt
    w[-1] = so3_log(mm3(R[-1], R[-2].transpose(-1, -2))) / dt
    return w
