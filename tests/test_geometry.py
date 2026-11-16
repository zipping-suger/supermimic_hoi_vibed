import numpy as np
import torch

from supermimic_hoi.geometry import GRID_MARGIN, ObjectGeometry, distance_grid

# A box of half extents (0.1, 0.2, 0.3) as twelve triangles with mixed orientation.
HALF = np.array([0.1, 0.2, 0.3])


def box_triangles():
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * HALF
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    return np.array([corners[[q[i], q[j], q[k]]] for q in quads for i, j, k in ((0, 1, 2), (0, 3, 2))])


def test_distance_field_of_a_box_and_its_lowest_vertex(tmp_path):
    grid, origin, spacing = distance_grid([box_triangles()])
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=np.float32) * HALF.astype(np.float32)
    np.savez(tmp_path / "geometry.npz", names=np.array(["box", "copy"]), grid=np.stack((grid, grid)),
             origin=np.stack((origin, origin)), spacing=np.stack((spacing, spacing)), vertices=np.stack((corners, corners)))  # fmt: skip
    geometry = ObjectGeometry(tmp_path, ["box", "copy"], "cpu")
    assert np.allclose(origin, -HALF - GRID_MARGIN, atol=1e-6)
    points = torch.tensor([[[0.0, 0.0, 0.0], [0.15, 0.0, 0.0], [0.0, 0.0, -0.4], [0.2, 0.3, 0.0], [0.05, 0.0, 0.0]]], requires_grad=True)
    out = geometry.surface(torch.tensor([1]), points)[0]
    # Inside: minus the distance to the nearest face. Outside: to the nearest face or edge.
    expected = torch.tensor([-0.1, 0.05, 0.1, 0.1 * 2**0.5, -0.05])
    assert torch.allclose(out[:, 0], expected, atol=4e-3)
    assert torch.allclose(out[1, 1:], torch.tensor([1.0, 0.0, 0.0]), atol=0.05)
    assert torch.allclose(out[2, 1:], torch.tensor([0.0, 0.0, -1.0]), atol=0.05)
    # The lookup is differentiable in the query: moving away from the face increases the distance.
    out[1, 0].backward()
    assert points.grad[0, 1, 0] > 0.9

    upright = geometry.bottom(torch.tensor([0]), torch.tensor([[0.0, 0.0, 1.0]]), torch.eye(3)[None])
    turned = torch.tensor([[[1.0, 0, 0], [0, 0, -1], [0, 1, 0]]])  # the y extent now points up
    assert torch.allclose(upright, torch.tensor([0.7])) and torch.allclose(geometry.bottom(torch.tensor([0]), torch.zeros(1, 3), turned), torch.tensor([-0.2]))
