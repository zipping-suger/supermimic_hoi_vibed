"""Time-major replay on the training device. Terminal transitions are kept; windows never cross resets."""

from __future__ import annotations

import torch

FIELDS = ("s", "s1", "f", "age", "prev_action", "hand", "u", "context", "finger_score", "o", "cut", "valid")


class Replay:
    """Ring buffer of (steps, num_envs) transitions.

    Every clip owns a contiguous, equally sized block of worlds. Sampling first draws a clip
    uniformly and then a transition of that clip, so a hard clip whose episodes end early keeps
    the same weight as an easy one.
    """

    def __init__(self, steps: int, num_envs: int, num_clips: int, device: str):
        if num_envs % num_clips:
            raise ValueError("Replay needs the same number of worlds for every clip")
        self.T, self.N, self.clips, self.device = steps, num_envs, num_clips, device
        self.data, self.size, self.ptr = {}, 0, 0

    def add(self, row: dict):
        if not self.data:
            self.data = {k: torch.empty((self.T, *row[k].shape), dtype=row[k].dtype, device=self.device) for k in FIELDS}
        for k in FIELDS:
            self.data[k][self.ptr].copy_(row[k])
        self.ptr = (self.ptr + 1) % self.T
        self.size = min(self.T, self.size + 1)

    def row(self, age):
        """Storage row of the transition ``age`` steps after the oldest one."""
        return (self.ptr - self.size + age) % self.T

    def sample(self, batch: int, length: int):
        """(age, env) of ``batch`` valid transitions among the oldest ``length`` steps, clips drawn uniformly."""
        if length < 1:
            raise ValueError("Cannot sample an empty replay")
        width = self.N // self.clips
        clips = torch.randint(self.clips, (batch,), device=self.device)
        age = torch.zeros(batch, dtype=torch.long, device=self.device)
        env = clips * width
        accepted = torch.zeros(batch, dtype=torch.bool, device=self.device)
        for _ in range(8):
            proposed_age = torch.randint(length, (batch,), device=self.device)
            proposed_env = clips * width + torch.randint(width, (batch,), device=self.device)
            take = ~accepted & self.data["valid"][self.row(proposed_age), proposed_env]
            age, env = torch.where(take, proposed_age, age), torch.where(take, proposed_env, env)
            accepted |= take
            if bool(accepted.all()):
                return age, env
        # Rare: enumerate the valid transitions of each remaining clip instead of switching clips.
        ages = torch.arange(length, device=self.device)
        for clip in clips[~accepted].unique().tolist():
            envs = torch.arange(clip * width, (clip + 1) * width, device=self.device)
            available = self.data["valid"][self.row(ages)[:, None], envs[None, :]].nonzero()
            if not available.numel():
                raise ValueError(f"Replay has no valid transition for clip {clip}")
            ids = ((clips == clip) & ~accepted).nonzero().flatten()
            chosen = available[torch.randint(len(available), (len(ids),), device=self.device)]
            age[ids], env[ids] = chosen[:, 0], envs[chosen[:, 1]]
        return age, env

    def windows(self, batch: int, k: int) -> dict:
        """``k`` consecutive transitions per sample; ``mask`` is 0 after an episode boundary or a rejected step."""
        if self.size < k:
            raise ValueError("Replay is shorter than the requested window")
        age, env = self.sample(batch, self.size - k + 1)
        rows = self.row(age[:, None] + torch.arange(k, device=self.device))
        out = {key: self.data[key][rows, env[:, None]] for key in ("s", "s1", "u", "context", "finger_score", "f", "cut", "valid")}
        proceed = (out["valid"] & ~out["cut"]).float()
        out["mask"] = out["valid"].float() * torch.cumprod(torch.cat((torch.ones(batch, 1, device=self.device), proceed[:, :-1]), 1), 1)
        return out

    def starts(self, batch: int, history: int) -> dict:
        """Imagination starts: a stored state with its controller memory and observation history."""
        age, env = self.sample(batch, self.size)
        row = self.row(age)
        out = {key: self.data[key][row, env] for key in ("s", "f", "age", "prev_action", "hand", "context")}
        # Older observations exist only as far back as both the buffer and the episode reach.
        back = torch.arange(history - 1, -1, -1, device=self.device)
        back = torch.minimum(back[None], torch.minimum(age, out["age"])[:, None])
        out["history"] = self.data["o"][self.row(age[:, None] - back), env[:, None]]
        return out
