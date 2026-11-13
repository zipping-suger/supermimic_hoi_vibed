import torch

from supermimic_hoi import rotations as R


def random_rotations(n):
    return R.so3_exp(torch.randn(n, 3, dtype=torch.float64))


def test_products_match_matmul():
    A, B, v = random_rotations(5), random_rotations(5), torch.randn(5, 3, dtype=torch.float64)
    assert torch.allclose(R.mm3(A, B), A @ B)
    assert torch.allclose(R.mv3(A, v), (A @ v[..., None])[..., 0])


def test_exp_log_roundtrip_including_small_angles():
    v = torch.cat((torch.randn(64, 3, dtype=torch.float64), 1e-7 * torch.randn(8, 3, dtype=torch.float64)))
    v = v / v.norm(dim=-1, keepdim=True).clamp_min(1) * v.norm(dim=-1, keepdim=True).clamp(max=3.0)
    assert torch.allclose(R.so3_log(R.so3_exp(v)), v, atol=1e-9)


def test_quaternion_roundtrip():
    rot = random_rotations(32)
    assert torch.allclose(R.quat_to_mat(R.mat_to_quat(rot)), rot, atol=1e-9)


def test_angle_squared_is_geodesic_and_smooth_at_zero():
    axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    angles = torch.tensor([0.0, 0.3, 1.2, 3.0], dtype=torch.float64)
    rot = R.axis_angle_to_mat(axis, angles)
    eye = torch.eye(3, dtype=torch.float64).expand(4, 3, 3)
    assert torch.allclose(R.angle_squared(eye, rot), angles.square(), atol=1e-6)
    x = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    R.angle_squared(torch.eye(3, dtype=torch.float64), R.so3_exp(x)).backward()
    assert torch.isfinite(x.grad).all()


def test_finite_difference_velocities():
    t = torch.arange(50, dtype=torch.float64)[:, None] * 0.02
    assert torch.allclose(R.central_difference(2.0 * t, 0.02), torch.full((50, 1), 2.0, dtype=torch.float64))
    spin = torch.tensor([0.0, 0.0, 1.5], dtype=torch.float64)
    rot = R.so3_exp(t * spin)
    assert torch.allclose(R.angular_velocity(rot, 0.02), spin.expand(50, 3), atol=1e-9)
