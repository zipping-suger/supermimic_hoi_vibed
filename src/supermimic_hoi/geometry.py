"""Collision geometry of the objects for analytic queries: hull vertices and a signed distance field.

``prepare`` writes ``geometry.npz`` next to the object registry. Per object it holds the vertices
of the convex collision parts and a regular grid over their bounding box plus a margin with the
signed distance to the union of the parts (negative inside) and its unit gradient, all in the
object's body frame. Lookups are trilinear, fixed-shape and differentiable in the query points.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

GRID_SHAPE = (64, 64, 144)
GRID_MARGIN = 0.15  # the grid extends this far beyond the collision geometry [m]


def segment_distance_squared(p, a, b):
    ab = b - a
    t = (((p[:, None] - a) * ab).sum(-1) / (ab * ab).sum(-1).clamp_min(1e-20)).clamp(0, 1)
    return (p[:, None] - (a + t[..., None] * ab)).square().sum(-1)


def convex_signed_distance(points, triangles):
    """Exact signed distance of ``points`` (P, 3) to a convex polytope given by its outward triangles (T, 3, 3)."""
    a, b, c = triangles.unbind(1)
    n = torch.linalg.cross(b - a, c - a)
    n = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-20)
    height = ((points[:, None] - a) * n).sum(-1)  # (P, T)
    foot = points[:, None] - height[..., None] * n
    within = torch.ones_like(height, dtype=torch.bool)
    for u, v in ((a, b), (b, c), (c, a)):
        within &= (torch.linalg.cross((v - u).expand_as(foot), foot - u) * n).sum(-1) >= 0
    edge = torch.minimum(torch.minimum(segment_distance_squared(points, a, b), segment_distance_squared(points, b, c)),
                         segment_distance_squared(points, c, a))  # fmt: skip
    outside = torch.where(within, height.square(), edge).amin(-1).sqrt()
    inside = height.amax(-1)
    return torch.where(inside <= 0, inside, outside)


def distance_grid(parts: list[np.ndarray], device: str = "cpu"):
    """(grid (*GRID_SHAPE, 4), origin (3,), spacing (3,)) for convex parts given as triangles (T, 3, 3)."""
    oriented = []
    for part in parts:
        a, b, c = part[:, 0], part[:, 1], part[:, 2]
        inward = (np.cross(b - a, c - a) * (a - part.reshape(-1, 3).mean(0))).sum(-1) < 0
        oriented.append(np.where(inward[:, None, None], part[:, [0, 2, 1]], part))
    parts = oriented
    everything = np.concatenate([part.reshape(-1, 3) for part in parts])
    lo, hi = everything.min(0) - GRID_MARGIN, everything.max(0) + GRID_MARGIN
    spacing = (hi - lo) / (np.array(GRID_SHAPE) - 1)
    axes = [torch.linspace(lo[i], hi[i], GRID_SHAPE[i], dtype=torch.float64, device=device) for i in range(3)]
    points = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    distance = torch.full((len(points),), torch.inf, dtype=torch.float64, device=device)
    for part in parts:
        triangles = torch.as_tensor(part, dtype=torch.float64, device=device)
        for rows in torch.arange(len(points), device=device).split(16384):
            distance[rows] = torch.minimum(distance[rows], convex_signed_distance(points[rows], triangles))
    field = distance.reshape(GRID_SHAPE)
    gradient = torch.stack(torch.gradient(field, spacing=[float(x) for x in spacing]), -1)
    gradient = gradient / gradient.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    grid = torch.cat((field[..., None], gradient), -1).float().cpu().numpy()
    return grid, lo.astype(np.float32), spacing.astype(np.float32)


class ObjectGeometry:
    def __init__(self, folder: str | Path, object_names: list[str], device: str):
        with np.load(Path(folder) / "geometry.npz") as data:
            if data["names"].tolist() != list(object_names):
                raise ValueError("geometry.npz does not match the object registry; rerun supermimic_hoi.prepare")
            grid = torch.from_numpy(data["grid"]).to(device)
            self.origin = torch.from_numpy(data["origin"]).to(device)
            self.inverse_spacing = 1 / torch.from_numpy(data["spacing"]).to(device)
            self.vertices = torch.from_numpy(data["vertices"]).to(device)  # (objects, V, 3), padded by repetition
        self.shape = tuple(grid.shape[1:4])
        self.cells = self.shape[0] * self.shape[1] * self.shape[2]
        self.grid = grid.reshape(-1, 4)
        self.limit = torch.tensor(self.shape, device=device, dtype=torch.float32) - 1.001
        corners = [(i, j, k) for i in (0, 1) for j in (0, 1) for k in (0, 1)]
        self.corner_offset = torch.tensor([(i * self.shape[1] + j) * self.shape[2] + k for i, j, k in corners], device=device)
        self.corner_bits = torch.tensor(corners, device=device, dtype=torch.float32)

    def surface(self, obj, points):
        """Signed distance and unit direction away from the surface at ``points`` (..., K, 3) given in the
        frame of the objects ``obj`` (...,). Returns (..., K, 4); beyond the grid the border value holds."""
        u = (points - self.origin[obj].unsqueeze(-2)) * self.inverse_spacing[obj].unsqueeze(-2)
        u = torch.minimum(u.clamp_min(0), self.limit)
        base = u.detach().floor()
        t = u - base
        cell = base.long()
        index = obj[..., None] * self.cells + (cell[..., 0] * self.shape[1] + cell[..., 1]) * self.shape[2] + cell[..., 2]
        values = self.grid[index[..., None] + self.corner_offset]  # (..., K, 8, 4)
        weights = torch.where(self.corner_bits > 0, t.unsqueeze(-2), 1 - t.unsqueeze(-2))  # (..., K, 8, 3)
        weights = weights[..., 0] * weights[..., 1] * weights[..., 2]
        return (values * weights[..., None]).sum(-2)

    def bottom(self, obj, position, rotation):
        """Height of the lowest collision vertex of the objects ``obj`` (...,) at the given poses."""
        return (self.vertices[obj] * rotation[..., None, 2, :]).sum(-1).amin(-1) + position[..., 2]
