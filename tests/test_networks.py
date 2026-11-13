import torch

from supermimic_hoi.networks import Actor, RunningNorm, WorldModel
from supermimic_hoi.rotations import so3_exp

J = 5


def random_state(n):
    g = lambda *shape: torch.randn(n, *shape, dtype=torch.float64)  # noqa: E731
    return (g(3), so3_exp(g(3)), g(3), g(3), g(J), g(J), g(3), so3_exp(g(3)), g(3), g(3))


def test_running_norm_matches_masked_moments_and_freezes():
    torch.manual_seed(0)
    norm = RunningNorm(3, max_count=150)
    x = torch.randn(200, 3) * torch.tensor([1.0, 5.0, 0.1]) + torch.tensor([0.0, 2.0, -1.0])
    valid = torch.rand(200) > 0.3
    x[~valid] = float("nan")  # rejected rows must not contaminate the statistics
    norm.update(x[:100], valid[:100])
    norm.update(x[100:], valid[100:])
    kept = x[valid]
    assert torch.allclose(norm.mean, kept.mean(0), atol=1e-5) and torch.allclose(norm.var, kept.var(0, unbiased=False), atol=1e-4)
    assert float(norm.count) == int(valid.sum())
    norm.max_count = 1.0
    mean = norm.mean.clone()
    norm.update(torch.randn(50, 3) + 100)
    assert torch.equal(norm.mean, mean)


def test_running_norm_ignores_an_empty_batch():
    norm = RunningNorm(2)
    norm.update(torch.randn(4, 2), torch.zeros(4, dtype=torch.bool))
    assert float(norm.count) == 0 and torch.equal(norm.mean, torch.zeros(2))


def test_world_model_integrate_inverts_targets():
    torch.manual_seed(0)
    wm = WorldModel(J, key_dim=2, context_dim=3, hidden=(8,)).double()
    s, s1 = random_state(16), random_state(16)
    # Keep the step small enough for a unique rotation logarithm.
    s1 = tuple(a + 0.05 * (b - a) if a.shape[-1] != 3 or a.ndim == 2 else a @ so3_exp(0.05 * torch.randn(16, 3, dtype=torch.float64))
               for a, b in zip(s, s1))
    rebuilt = wm.integrate(s, wm.targets(s, s1))
    for a, b in zip(rebuilt, s1):
        assert torch.allclose(a, b, atol=1e-9)
    assert torch.allclose(wm.error(s1, s1), torch.zeros(16, wm.delta_dim, dtype=torch.float64), atol=1e-9)
    assert torch.allclose(wm.error(rebuilt, s1), torch.zeros(16, wm.delta_dim, dtype=torch.float64), atol=1e-8)


def test_world_model_starts_as_drift_and_is_differentiable_in_the_action():
    torch.manual_seed(0)
    wm = WorldModel(J, key_dim=2, context_dim=3, hidden=(8,))
    s = tuple(x.float() for x in random_state(4))
    u = torch.randn(4, J, requires_grad=True)
    s1 = wm(s, u, torch.eye(2)[[0, 1, 0, 1]], torch.zeros(4, 3))
    assert torch.allclose(s1[4], s[4] + 0.02 * s[5], atol=1e-5)
    wm.net[-1].weight.data.normal_(std=0.1)
    wm(s, u, torch.eye(2)[[0, 1, 0, 1]], torch.zeros(4, 3))[0].sum().backward()
    assert u.grad.abs().sum() > 0


def test_actor_uses_precomputed_latent():
    torch.manual_seed(0)
    actor = Actor(obs_dim=6, goal_dim=9, act_dim=4, history=3, hidden=(16, 16))
    history, goal = torch.randn(5, 3, 6), torch.randn(5, 9)
    assert torch.allclose(actor(history, goal), actor(history, goal, actor.encode(goal)))
