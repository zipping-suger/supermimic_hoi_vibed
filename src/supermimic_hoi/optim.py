"""Optimizers whose whole step can be captured in a CUDA graph and gated on the device.

Learning rates are 0-dim device tensors (set them with ``fill_``), so a captured step follows the
schedule. A step takes ``ok``, a 0-dim bool tensor: where it is false nothing changes, neither
parameters nor optimizer state, without a host read.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def make_adam(params, lr: float, betas=(0.9, 0.999)) -> torch.optim.Adam:
    params = list(params)
    return torch.optim.Adam(params, lr=torch.tensor(float(lr), device=params[0].device), betas=betas,
                            fused=True, capturable=True)  # fmt: skip


def adam_step(adam: torch.optim.Adam, ok: torch.Tensor):
    adam.found_inf = (~ok).float()  # the fused kernel skips the step and undoes the step count
    try:
        adam.step()
    finally:
        adam.found_inf = None


class MuonAdam:
    """Muon on the hidden weight matrices of ``mlps`` (every Linear except each MLP's first and
    last), Adam on everything else.

    Muon's update is rescaled to Adam's update RMS (``match_rms_adamw``), so one learning rate
    serves both. The Muon step repeats torch.optim.Muon's arithmetic with the tensor learning
    rate, which torch.optim.Muon itself would read on the host.
    """

    def __init__(self, model: nn.Module, mlps: list[nn.Module], lr: float, betas=(0.9, 0.95)):
        linears = lambda m: [x for x in m.modules() if isinstance(x, nn.Linear)]  # noqa: E731
        hidden = {id(layer.weight) for m in mlps for layer in linears(m)[1:-1]}
        muon = [p for p in model.parameters() if id(p) in hidden]
        self.muon = torch.optim.Muon(muon, lr=torch.tensor(float(lr), device=muon[0].device), weight_decay=0.0,
                                     adjust_lr_fn="match_rms_adamw")  # fmt: skip
        self.adam = make_adam([p for p in model.parameters() if id(p) not in hidden], lr, betas)

    @property
    def param_groups(self):
        # A property: load_state_dict replaces the group dicts.
        return self.muon.param_groups + self.adam.param_groups

    def zero_grad(self, set_to_none: bool = True):
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adam.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, ok: torch.Tensor):
        from torch.optim._muon import _adjust_lr, _zeropower_via_newtonschulz

        for group in self.muon.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.muon.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p.grad)
                buffer = state["momentum_buffer"]
                buffer.copy_(torch.where(ok, buffer.lerp(p.grad, 1 - group["momentum"]), buffer))
                update = p.grad.lerp(buffer, group["momentum"]) if group["nesterov"] else buffer
                update = _zeropower_via_newtonschulz(update, group["ns_coefficients"], group["ns_steps"], group["eps"])
                ratio = _adjust_lr(1.0, group["adjust_lr_fn"], p.shape)
                p.add_(torch.where(ok, update.to(p.dtype) * (group["lr"] * -ratio), 0.0))
        adam_step(self.adam, ok)

    def state_dict(self):
        return {"muon": self.muon.state_dict(), "adam": self.adam.state_dict()}

    def load_state_dict(self, state: dict):
        self.muon.load_state_dict(state["muon"])
        self.adam.load_state_dict(state["adam"])


def load_optimizer(optimizer, state: dict):
    """Restore optimizer state while keeping the live learning-rate tensors the training loop fills."""
    rates = [group["lr"] for group in optimizer.param_groups]
    optimizer.load_state_dict(state)
    for group, rate in zip(optimizer.param_groups, rates):
        rate.fill_(float(group["lr"]))
        group["lr"] = rate
