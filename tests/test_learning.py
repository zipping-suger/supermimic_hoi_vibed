import math
from types import SimpleNamespace

import torch
from torch import nn

from supermimic_hoi.learning import actor_objective, bounded_bootstrap, discounted_return_bound, gae_targets, guarded_step, lr_at
from supermimic_hoi.optim import MuonAdam, make_adam

CFG = SimpleNamespace(grad_clip_norm=1.0, max_gradient_norm=1e4)


def test_gae_matches_definition():
    torch.manual_seed(0)
    values, rewards = torch.randn(4, 3), torch.randn(3, 3)
    dones = torch.tensor([[0.0, 0, 1], [0, 0, 0], [0, 1, 0]])
    gamma, lam = 0.9, 0.8
    out = gae_targets(values, rewards, dones, gamma, lam)
    target = values[-1]
    for t in (2, 1, 0):
        target = rewards[t] + gamma * (1 - dones[t]) * ((1 - lam) * values[t + 1] + lam * target)
        assert torch.allclose(out[t], target)
    # A terminal transition is worth exactly its reward.
    assert torch.allclose(out[0, 2], rewards[0, 2])


def test_actor_objective_matches_definition():
    torch.manual_seed(0)
    r, d, boot, gamma = torch.rand(5, 4), torch.rand(5, 4) * 0.3, torch.rand(4), 0.97
    J, survival = actor_objective(r, d, boot, gamma)
    alive = torch.cumprod(1 - d, 0)
    c = torch.cat((torch.ones(1, 4), alive[:-1]))
    expected = (gamma ** torch.arange(5.0)[:, None] * c * r).sum(0) + gamma**5 * alive[-1] * boot
    assert torch.allclose(J, expected) and torch.allclose(survival, alive[-1])


def test_return_bound_and_bootstrap():
    bound = discounted_return_bound(torch.tensor([0, 1, 3]), 0.2, 0.9)
    assert torch.allclose(bound, torch.tensor([0.0, 0.2, 0.2 * (1 + 0.9 + 0.81)]))
    values = torch.tensor([5.0, 5.0, -1.0])
    assert torch.equal(bounded_bootstrap(values, torch.tensor([0.0, 0.3, 0.3])), torch.tensor([0.0, 0.3, -1.0]))


def test_lr_schedule_endpoints():
    assert lr_at(0, 1.0, 100) == 1.0 and math.isclose(lr_at(100, 1.0, 100), 0.03) and lr_at(10**6, 1.0, 100) == 0.03


def _model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 16), nn.Tanh(), nn.Linear(16, 16), nn.Tanh(), nn.Linear(16, 2))


def _snapshot(model, optimizer):
    tensors = [p.detach().clone() for p in model.parameters()]
    for opt in (optimizer.muon, optimizer.adam) if isinstance(optimizer, MuonAdam) else (optimizer,):
        for state in opt.state.values():
            tensors += [v.detach().clone() for v in state.values() if torch.is_tensor(v)]
    return tensors


def test_guarded_step_updates_then_skips_invalid_updates_completely():
    for make in (lambda m: make_adam(m.parameters(), 1e-2), lambda m: MuonAdam(m, [m], 1e-2)):
        model = _model()
        optimizer = make(model)
        x = torch.randn(8, 4)
        before = _snapshot(model, optimizer)
        optimizer.zero_grad()
        norm, ok = guarded_step(model(x).square().mean(), model, optimizer, CFG)
        assert bool(ok) and float(norm) > 0
        after = _snapshot(model, optimizer)
        assert any(not torch.equal(a, b) for a, b in zip(before, after))
        # A nonfinite loss leaves parameters, moments and step counts untouched.
        optimizer.zero_grad()
        _, ok = guarded_step(model(x).square().mean() * float("nan"), model, optimizer, CFG)
        assert not bool(ok)
        assert all(torch.equal(a, b) for a, b in zip(after, _snapshot(model, optimizer)))
        # A disabled step is skipped without being reported as invalid.
        optimizer.zero_grad()
        _, ok = guarded_step(model(x).square().mean(), model, optimizer, CFG, enabled=torch.tensor(False))
        assert bool(ok)
        assert all(torch.equal(a, b) for a, b in zip(after, _snapshot(model, optimizer)))


def test_muon_takes_hidden_matrices_and_learning_rate_is_live():
    model = _model()
    optimizer = MuonAdam(model, [model], 1e-2)
    assert [p.shape for g in optimizer.muon.param_groups for p in g["params"]] == [model[2].weight.shape]
    for group in optimizer.param_groups:
        group["lr"].fill_(0.0)
    before = [p.detach().clone() for p in model.parameters()]
    optimizer.zero_grad()
    guarded_step(model(torch.randn(8, 4)).square().mean(), model, optimizer, CFG)
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))
