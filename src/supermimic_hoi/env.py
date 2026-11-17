"""Reference-synchronized robot/object episodes on top of the physics backend."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch

from .control import Controller
from .geometry import ObjectGeometry
from .mdp import HOIMDP, force_scores
from .motion import Motion
from .randomization import ContextLayout, Randomizer
from .reset_guard import build_reset_guard
from .robot import Robot
from .robots import load_robot
from .sim import CPUPhysics, WarpPhysics, load_registry, robot_signature
from .state import pack


class HOIEnv:
    """Fixed-shape vector environment.

    A world is not tied to a clip: every reset draws one and the world takes that clip's object, so
    the number of worlds is independent of the number of clips. The clip that has had the least
    simulated time is the most likely to be drawn, which gives every clip the same share of time
    however long its episodes last. Robot and object are always reset together to the same
    reference frame, with reference velocities; there is no independent pose jitter, which would
    break the contact geometry. All per-world tensors are updated in place, so a control step and
    its resets can be captured in a CUDA graph.
    """

    def __init__(self, motion: Motion, controller: Controller, mdp: HOIMDP, robot: Robot, registry: dict, cfg):
        self.motion, self.controller, self.mdp, self.cfg = motion, controller, mdp, cfg
        self.device, self.N = cfg.device, cfg.num_envs
        backend = CPUPhysics if cfg.backend == "cpu" else WarpPhysics
        self.physics = backend(robot, registry, cfg.object_assets, sorted(set(motion.clip_object.tolist())), self.N, cfg)
        # Simulated steps every clip has received, counted from the least served clip.
        self.clip_steps = torch.zeros(len(motion.names), device=self.device)
        self.randomizer = Randomizer(robot, self.physics, mdp.layout, cfg.randomize)
        self.reset_map, self.reset_depths, self.reset_summary = build_reset_guard(
            motion, robot, registry, cfg.object_assets, cfg.reset_overlap_tolerance)
        self.pregrasp, self.pregrasp_count = motion.pregrasp_frames()

        N, dev = self.N, self.device
        self.f = torch.zeros(N, dtype=torch.long, device=dev)
        self.age = torch.zeros_like(self.f)
        self.episode_return = torch.zeros(N, device=dev)
        self.previous_action = torch.zeros(N, controller.num_body + controller.num_hand, device=dev)
        self.hand_targets = torch.zeros(N, controller.num_hand, device=dev)
        self.context = self.randomizer.context().clone()
        # Tensor-only hooks; the runner may replace them with compiled versions.
        self.observe_fn, self.transition_fn = self.observe_state, self.transition_outputs
        self.history = None
        everyone = torch.ones(N, dtype=torch.bool, device=dev)
        self.reset(everyone)
        self.history = self.observe()[:, None].repeat(1, cfg.history, 1)

    def sample_frames(self):
        """Episode starts. The clip: drawn in proportion to how far each clip is behind the most served
        one. The frame: frame zero, shortly before a contact phase, or uniform over the clip; then
        moved to the nearest frame that passes the penetration screen."""
        c, m = self.cfg, self.motion
        uniform = lambda: torch.rand(self.N, device=self.device)  # noqa: E731
        edges = (self.clip_steps.max() - self.clip_steps + 1).cumsum(0)
        clips = torch.searchsorted(edges, uniform() * edges[-1], right=True).clamp(max=len(edges) - 1)
        start, end = m.clip_start[clips], m.clip_end[clips]
        frames = start + (uniform() * (end - start)).long()
        draw = uniform()
        choice = (uniform() * self.pregrasp_count[clips]).long()
        frames = torch.where(draw < c.reset_frame_zero_probability + c.reset_pregrasp_probability, self.pregrasp[clips, choice], frames)
        frames = torch.where(draw < c.reset_frame_zero_probability, start, frames)
        return self.reset_map[frames]

    def reset(self, mask, frames=None):
        """Start new episodes where ``mask`` is true. ``frames`` (N,) overrides the sampled starts as given."""
        frames = self.sample_frames() if frames is None else frames
        state = self.motion.ref_state(frames)
        self.randomizer.resample(mask)
        self.physics.reset(mask, state, self.motion.clip_object[self.motion.clip_id[frames]])
        self.context.copy_(torch.where(mask[:, None], self.randomizer.context(), self.context))
        self.f.copy_(torch.where(mask, frames, self.f))
        for value in (self.age, self.episode_return, self.previous_action):
            value.masked_fill_(mask.reshape(-1, *([1] * (value.ndim - 1))), 0)
        self.hand_targets.copy_(torch.where(mask[:, None], self.controller.seed(state[4]), self.hand_targets))
        if self.history is not None:
            self.history.copy_(torch.where(mask[:, None, None], self.observe()[:, None], self.history))

    def observe(self):
        return self.observe_fn(self.physics.state(), self.f, self.previous_action, self.hand_targets, self.context)

    def observe_state(self, state, frame, previous_action, hand_targets, context):
        return self.mdp.add_noise(self.mdp.obs(state, frame, previous_action, hand_targets, context))

    def transition_outputs(self, state, frame, action, previous_action, hand_targets, context, hand_force, finger_force, episode_start):
        """Everything derived from the post-physics state in one fused tensor computation."""
        c = self.cfg.reward
        contact = (hand_force > c.contact_force_threshold).float()
        scores = force_scores(finger_force, c.contact_force_threshold, c.contact_force_saturation)
        fk = self.mdp.fk(state)
        reward, terms = self.mdp.reward(state, frame, action, previous_action, scores, episode_start, fk)
        failure, reasons = self.mdp.hard_termination(state, frame)
        obs = self.mdp.add_noise(self.mdp.obs(state, frame, action, hand_targets, context, fk))
        return contact, scores, reward, terms, failure, reasons, obs

    @torch.no_grad()
    def step(self, action, noise=None):
        """Apply one action everywhere and return the transition. Does not reset finished episodes.

        ``noise`` perturbs only what the drives execute. The action that the policy observes next
        and that the action costs score is the one it chose, as in imagination and evaluation.
        """
        m, c = self.motion, self.controller
        state = self.physics.state()
        f, age = self.f.clone(), self.age.clone()
        previous, hand, context = self.previous_action.clone(), self.hand_targets.clone(), self.context.clone()
        f1 = m.next_frame(f)
        action = c.bound_action(action, m.q[f1])
        executed = action if noise is None else c.bound_action(action + noise, m.q[f1])
        target, hand1 = c(executed, m.q[f1], hand, self.mdp.layout.offsets(context))
        state1, hand_force, finger_force, valid = self.physics.step(target)
        contact, scores, reward, terms, failure, reasons, obs1 = self.transition_fn(
            state1, f1, action, previous, hand1, context, hand_force, finger_force, age == 0)  # fmt: skip
        # A rejected world was restored to keep the kernels safe; that is not a usable transition.
        reward = torch.where(valid, reward, 0.0)
        terms = {k: torch.where(valid, v, 0.0) for k, v in terms.items()}
        failure, reasons = failure | ~valid, reasons & valid[:, None]
        clip_end = f1 >= m.clip_end[m.clip_id[f]]
        done = failure | clip_end | (age + 1 >= self.cfg.max_episode_steps)
        self.episode_return.add_(reward)
        row = dict(s=pack(state), s1=pack(state1), f=f, age=age, action=action, prev_action=previous, hand=hand,
                   u=c.effective(target), context=context, contact=contact, finger_score=scores,
                   o=self.history[:, -1].clone(), cut=done, term=failure, clip_end=clip_end, valid=valid,
                   reward=reward, terms=terms, reasons=reasons, episode_return=self.episode_return.clone())  # fmt: skip
        self.clip_steps.index_add_(0, m.clip_id[f], torch.ones_like(reward))
        self.clip_steps.sub_(self.clip_steps.min())
        self.f.copy_(f1)
        self.age.add_(1)
        self.previous_action.copy_(action)
        self.hand_targets.copy_(hand1)
        self.history.copy_(torch.cat((self.history[:, 1:], obs1[:, None]), 1))
        return row


@dataclass
class Task:
    robot: Robot
    registry: dict
    motion: Motion
    controller: Controller
    mdp: HOIMDP
    env: HOIEnv
    signature: str  # dataset, reference processing and physical assets


def make_task(cfg) -> Task:
    robot = load_robot(cfg.robot_assets)
    registry, object_signature = load_registry(cfg.object_assets)
    motion = Motion(cfg.dataset, robot, [o["name"] for o in registry["objects"]], cfg.device, cfg.clips)
    if cfg.device.startswith("cuda"):
        torch.cuda.empty_cache()  # release reference-FK scratch before Warp allocates its workspace
    controller = Controller(robot, cfg.device, body_action_clip=cfg.body_action_clip)
    geometry = ObjectGeometry(cfg.object_assets, motion.object_names, cfg.device)
    mdp = HOIMDP(motion, controller, ContextLayout(robot, cfg.device), cfg.reward, cfg.randomize, geometry)
    env = HOIEnv(motion, controller, mdp, robot, registry, cfg)
    signature = hashlib.sha256((motion.signature + object_signature + robot_signature(robot)).encode()).hexdigest()
    return Task(robot, registry, motion, controller, mdp, env, signature)
