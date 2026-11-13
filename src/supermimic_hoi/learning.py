"""Learning rules that do not depend on the task: returns, schedules and the gated update."""

from __future__ import annotations

import math

import torch

from .optim import MuonAdam, adam_step


def discounted_return_bound(remaining_steps, max_step_reward: float, gamma: float):
    """Largest discounted return obtainable in the remaining steps of a finite clip."""
    steps = remaining_steps.clamp_min(0).to(torch.float32)
    # expm1 avoids cancellation near the last frame.
    return -torch.expm1(steps * math.log(gamma)) * (max_step_reward / (1 - gamma))


def bounded_bootstrap(values, bounds):
    """Cap optimistic values at the attainable return; a terminal state (bound 0) is worth exactly 0."""
    return torch.where(bounds == 0, 0.0, values.clamp(max=bounds))


def gae_targets(values, rewards, dones, gamma: float, lam: float):
    """Lambda-return targets. values: (H+1, B); rewards, dones: (H, B)."""
    target, targets = values[-1], []
    for t in range(rewards.shape[0] - 1, -1, -1):
        target = rewards[t] + gamma * (1 - dones[t]) * ((1 - lam) * values[t + 1] + lam * target)
        targets.append(target)
    return torch.stack(targets[::-1])


def actor_objective(rewards, dones, bootstrap, gamma: float):
    """J = sum_t gamma^t c_t r_t + gamma^H c_H V_H with survival c_t = prod_{k<t} (1 - d_k).

    Returns (J (B,), final survival (B,)). The survival product is a loop of multiplications:
    cumprod's backward tests for zeros on the host, which a CUDA graph cannot capture.
    """
    H = rewards.shape[0]
    alive, objective = torch.ones_like(dones[0]), torch.zeros_like(rewards[0])
    for t in range(H):
        objective = objective + gamma**t * alive * rewards[t]
        alive = alive * (1 - dones[t])
    return objective + gamma**H * alive * bootstrap, alive


def lr_at(iteration: int, lr: float, decay_iters: int) -> float:
    """Cosine decay to 3 % of the base rate."""
    if iteration >= decay_iters:
        return lr * 0.03
    return lr * (0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * iteration / decay_iters)))


def guarded_step(loss, model, optimizer, cfg, *, value_clip: float = 0.0, enabled=None):
    """Backward, clip and step, skipping the whole update when it is invalid.

    Returns (pre-clipping gradient norm, ok). ``ok`` is false for a nonfinite loss or gradient or
    a norm above ``cfg.max_gradient_norm``; then neither parameters nor optimizer state change,
    and the caller is expected to stop. ``enabled`` (0-dim bool) skips a step without marking it
    invalid.
    """
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
    ok = torch.isfinite(loss.detach()) & torch.isfinite(norm) & (norm <= cfg.max_gradient_norm)
    step = ok if enabled is None else ok & enabled
    if value_clip > 0:
        torch.nn.utils.clip_grad_value_(model.parameters(), value_clip)
    for p in model.parameters():
        if p.grad is not None:
            p.grad.copy_(torch.where(step, p.grad, 0.0))
    if isinstance(optimizer, MuonAdam):
        optimizer.step(step)
    else:
        adam_step(optimizer, step)
    return norm.detach(), ok
