"""Serializable configuration shared by training, evaluation and checkpoints."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, fields


@dataclass
class RewardConfig:
    """Reward weights/scales and termination thresholds. Every reward is scaled by the 0.02 s control step."""

    # Tracking bonuses: weight * exp(-error^2 / std^2).
    body_pos: float = 1.0
    body_pos_std: float = 0.3
    body_ori: float = 1.0
    body_ori_std: float = 0.4
    body_velocity: float = 0.5
    body_angular_velocity: float = 1.0
    body_angvel_std: float = 3.14
    root_pos: float = 1.0
    root_ori: float = 1.0
    object_pos: float = 2.0
    object_pos_std: float = 0.2
    object_ori: float = 2.0
    object_ori_std: float = 0.3
    object_velocity: float = 0.25
    # Fingertip positions in the object frame, for hands the reference has in contact.
    relative: float = 1.0
    relative_std: float = 0.1
    # Measured (simulation) or predicted (imagination) finger force on intended fingers.
    contact: float = 2.0
    contact_force_threshold: float = 1.0
    contact_force_saturation: float = 5.0
    # Costs.
    action_rate: float = 0.1
    joint_limit: float = 2.0
    foot_slip: float = 0.1
    hand_saturation: float = 1.0
    # Termination.
    term_root_pos: float = 0.25
    term_tilt: float = 0.8
    term_object_pos: float = 0.3
    term_object_tilt: float = 0.3  # difference of the vertical component of the object's z axis

    BONUSES = ("body_pos", "body_ori", "body_velocity", "body_angular_velocity", "root_pos", "root_ori",
               "object_pos", "object_ori", "object_velocity", "relative", "contact")  # fmt: skip
    COSTS = ("action_rate", "joint_limit", "foot_slip", "hand_saturation")

    def validate(self):
        # The critic's finite return bound assumes nonnegative bounded bonuses and costs.
        for name in (*self.BONUSES, *self.COSTS):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"reward.{name} must be nonnegative and finite")
        for name in ("body_pos_std", "body_ori_std", "body_angvel_std", "object_pos_std", "object_ori_std", "relative_std"):
            if not getattr(self, name) > 0:
                raise ValueError(f"reward.{name} must be positive")
        if not 0 <= self.contact_force_threshold < self.contact_force_saturation:
            raise ValueError("Contact force saturation must exceed the nonnegative threshold")


@dataclass
class TrainConfig:
    # Data produced by `python -m supermimic_hoi.prepare`.
    dataset: str = "data/weave600"
    robot_assets: str = "data/robot"
    object_assets: str = "data/objects"
    clips: list[str] = field(default_factory=list)  # empty: every clip of the dataset

    # Simulation. Every clip owns num_envs / num_clips worlds.
    device: str = "cuda:0"
    backend: str = "warp"
    num_envs: int = 600
    max_episode_steps: int = 1000
    randomize: bool = True  # robot friction, torso CoM, encoder offsets, observation noise
    nconmax: int = 128
    njmax: int = 1000
    nccdmax: int | None = None
    # Numerical admission limits, not MDP termination: a world beyond them is rejected and reset.
    max_linear_speed: float = 20.0
    max_angular_speed: float = 200.0
    max_joint_speed: float = 500.0

    # Episode starts: frame zero, shortly before a contact phase, otherwise uniform over the clip.
    reset_frame_zero_probability: float = 0.25
    reset_pregrasp_probability: float = 0.25
    # Deepest reference overlap [m] admitted in a reset state, per kind of collision pair.
    reset_overlap_tolerance: dict[str, float] = field(default_factory=lambda: {"hand_object": 0.04, "robot_ground": 0.1, "other": 0.02})

    # Collection and replay.
    iterations: int = 30000
    collect_steps: int = 5
    buffer_steps: int = 1000
    warmup_steps: int = 64
    exploration: float = 0.1

    # World model and contact-reward head.
    hidden: list[int] = field(default_factory=lambda: [1024, 1024, 1024])
    reward_head_hidden: list[int] = field(default_factory=lambda: [256, 128])
    wm_window: int = 8
    wm_batch: int = 2048
    wm_updates: int = 5

    # Actor and critic.
    policy_hidden: list[int] = field(default_factory=lambda: [2048, 1024, 512, 256])
    critic_hidden: list[int] = field(default_factory=lambda: [256, 128])
    history: int = 10
    future: int = 10
    pi_batch: int = 2048
    horizon: int = 32
    critic_epochs: int = 5
    critic_minibatches: int = 4
    target_critic_ema: float = 0.4  # fraction of the bootstrap critic retained per policy update
    gamma: float = 0.99
    lam: float = 0.95
    ema: float = 0.995

    # Optimization.
    lr: float = 4e-4
    lr_decay_iters: int = 250000
    pi_lr_mult: float = 4.0  # actor multiplier until the return trigger, then ramped to 1
    lr_decay_ret: float = 0.75
    lr_decay_by: int = 1400
    lr_decay_len: int = 300
    pi_beta1: float = 0.9
    pi_clip_value: float = 0.1
    grad_clip_norm: float = 1.0
    max_gradient_norm: float = 1e4  # a larger pre-clipping norm stops training instead of updating
    norm_max_count: float = 2e8  # running normalizers freeze after this many rows
    body_action_clip: float = 20.0

    # Execution.
    amp: str = "bf16"
    compile: bool = True
    cuda_graph: bool = True

    # Bookkeeping.
    seed: int = 0
    log_dir: str = "logs/hoi"
    log_every: int = 10
    save_every: int = 1000
    resume: str = ""
    swanlab_mode: str = "disabled"
    swanlab_project: str = "supermimic_hoi"
    swanlab_name: str | None = None

    reward: RewardConfig = field(default_factory=RewardConfig)

    def __post_init__(self):
        if isinstance(self.reward, dict):
            self.reward = RewardConfig(**self.reward)

    @classmethod
    def from_json(cls, path, **overrides):
        values = json.loads(open(path).read()) if path else {}
        values.update({k: v for k, v in overrides.items() if v is not None})
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        return cls(**values)

    def validate(self):
        self.reward.validate()
        for name in ("num_envs", "max_episode_steps", "nconmax", "njmax", "iterations", "collect_steps", "buffer_steps",
                     "warmup_steps", "wm_window", "wm_batch", "wm_updates", "history", "future", "pi_batch", "horizon",
                     "critic_epochs", "critic_minibatches", "lr_decay_iters", "lr_decay_len", "log_every", "save_every"):  # fmt: skip
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        for name in ("max_linear_speed", "max_angular_speed", "max_joint_speed", "lr", "pi_lr_mult", "grad_clip_norm",
                     "max_gradient_norm", "norm_max_count", "body_action_clip"):  # fmt: skip
            if not (math.isfinite(getattr(self, name)) and getattr(self, name) > 0):
                raise ValueError(f"{name} must be positive and finite")
        if self.backend not in ("cpu", "warp") or self.amp not in ("none", "bf16"):
            raise ValueError("Unsupported backend or AMP mode")
        if (self.backend == "cpu") != (self.device == "cpu"):
            raise ValueError("backend='cpu' requires device='cpu'; backend='warp' requires a CUDA device")
        if self.backend == "cpu" and (self.randomize or self.cuda_graph):
            raise ValueError("The CPU backend supports neither randomization nor CUDA graphs")
        if self.nccdmax is not None and not 0 < self.nccdmax <= self.nconmax:
            raise ValueError("nccdmax must be positive and at most nconmax")
        probabilities = (self.reset_frame_zero_probability, self.reset_pregrasp_probability)
        if any(not 0 <= p <= 1 for p in probabilities) or sum(probabilities) > 1:
            raise ValueError("Invalid reset mixture probabilities")
        if set(self.reset_overlap_tolerance) != {"hand_object", "robot_ground", "other"} or any(
                not (math.isfinite(v) and v >= 0) for v in self.reset_overlap_tolerance.values()):  # fmt: skip
            raise ValueError("reset_overlap_tolerance needs finite nonnegative hand_object, robot_ground and other depths")
        if self.buffer_steps < self.wm_window or self.warmup_steps < self.wm_window:
            raise ValueError("Replay and warmup must cover wm_window")
        if (self.pi_batch * self.horizon) % self.critic_minibatches:
            raise ValueError("pi_batch * horizon must be divisible by critic_minibatches")
        if not (0 < self.gamma < 1 and 0 <= self.lam <= 1 and 0 <= self.ema < 1 and 0 <= self.pi_beta1 < 1):
            raise ValueError("Invalid discount, lambda, EMA or beta1")
        if not 0 <= self.target_critic_ema < 1 or not 0 <= self.exploration:
            raise ValueError("Invalid target critic retention or exploration scale")
        if self.swanlab_mode not in ("disabled", "offline", "online"):
            raise ValueError("Unsupported SwanLab mode")
