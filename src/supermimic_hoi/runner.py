"""Training: real collection, multistep world-model fitting, differentiable actor rollouts, lambda-return critic."""

from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from . import networks
from .buffer import Replay
from .cuda_graph import CUDACall, LearningGraphPool
from .env import make_task
from .learning import actor_objective, bounded_bootstrap, discounted_return_bound, gae_targets, guarded_step, lr_at
from .mdp import FAILURE_REASONS, SURFACE_DIM
from .networks import Actor, ContactRewardHead, Critic, RunningNorm, WorldModel
from .optim import MuonAdam, load_optimizer, make_adam
from .state import unpack
from .tracking import Tracker

MODELS = ("actor", "ema", "wm", "reward_head", "critic", "target_critic")
OPTIMIZERS = ("wm_opt", "reward_opt", "pi_opt", "v_opt")


def select(mask, current, fallback):
    """Per-sample choice between two tensors of the same, arbitrary trailing shape."""
    return torch.where(mask.reshape(-1, *([1] * (current.ndim - 1))), current, fallback)


class Runner:
    def __init__(self, cfg):
        cfg.validate()
        self.cfg = cfg
        cuda = cfg.device.startswith("cuda")
        if cuda:
            torch.cuda.set_device(cfg.device)
        torch.manual_seed(cfg.seed)
        torch.set_float32_matmul_precision("high")
        networks.AMP_DTYPE = torch.bfloat16 if cfg.amp == "bf16" and cuda else None

        self.task = make_task(cfg)
        self.motion, self.control, self.mdp, self.env = self.task.motion, self.task.controller, self.task.mdp, self.task.env
        self.replay = Replay(cfg.buffer_steps, cfg.num_envs, len(self.motion.names), cfg.device)

        env, J = self.env, self.task.robot.num_joints
        key_dim, context_dim = len(self.motion.object_names), env.context.shape[-1]
        self.wm = WorldModel(J, key_dim, context_dim, SURFACE_DIM, cfg.hidden).to(cfg.device)
        self.reward_head = ContactRewardHead(J, key_dim, context_dim, SURFACE_DIM, cfg.reward_head_hidden).to(cfg.device)
        self.actor = Actor(env.history.shape[-1], self.motion.goal_dim(cfg.future), self.task.robot.num_actions,
                           cfg.history, cfg.policy_hidden).to(cfg.device)  # fmt: skip
        self.ema = copy.deepcopy(self.actor).requires_grad_(False)
        with torch.no_grad():
            critic_dim = self.mdp.critic_input(env.physics.state(), env.f, env.previous_action, env.hand_targets, env.context,
                                               self.motion.goal(env.f, cfg.future)).shape[-1]  # fmt: skip
        self.critic = Critic(critic_dim, cfg.critic_hidden).to(cfg.device)
        # Actor rollouts and lambda-returns bootstrap from a slowly updated copy; freezing its
        # weights does not block derivatives with respect to its input.
        self.target_critic = copy.deepcopy(self.critic).eval().requires_grad_(False)
        for name in MODELS:
            for module in getattr(self, name).modules():
                if isinstance(module, RunningNorm):
                    module.max_count = cfg.norm_max_count

        self.wm_opt = MuonAdam(self.wm, [self.wm.net], cfg.lr, betas=(0.9, 0.999))
        self.pi_opt = MuonAdam(self.actor, [self.actor.enc, self.actor.dec], cfg.lr, betas=(cfg.pi_beta1, 0.95))
        self.v_opt = make_adam(self.critic.parameters(), cfg.lr)
        self.reward_opt = make_adam(self.reward_head.parameters(), cfg.lr)
        self.iteration = 0
        self.ema_updates = 0
        self.decay_at = None  # iteration at which the actor learning-rate multiplier starts to ramp down
        self.rejected_transitions = 0
        self.skipped = {}  # per model: [updates skipped as invalid, of which consecutive and most recent]

        # Compile one reusable imagination step rather than the whole unrolled horizon.
        compile_fn = (lambda fn, **options: torch.compile(fn, fullgraph=True, dynamic=False, **options)) if cfg.compile else (lambda fn, **options: fn)
        small_kernels = dict(options={"max_fusion_size": 16}) if cfg.compile else {}
        self._wm_loss = compile_fn(self.wm_loss)
        self._contact_reward_loss = compile_fn(self.contact_reward_loss)
        self._imagine_step = compile_fn(self.imagine_step, **small_kernels)
        self._act = compile_fn(self.act, **small_kernels)
        self._observe_transition = compile_fn(self.observe_transition, **small_kernels)
        env.observe_fn = compile_fn(env.observe_state, **small_kernels)
        env.transition_fn = compile_fn(env.transition_outputs, **small_kernels)

        if cfg.resume:
            self.load(cfg.resume)
        self._collect = self.collect_step
        self._wm_update, self._reward_update, self._actor_update = self.wm_update, self.reward_update, self.actor_update
        self._critic_batch, self._critic_update = self.critic_batch, self.critic_update
        if cfg.cuda_graph:
            pool = LearningGraphPool()
            self._collect = CUDACall(self.collect_step, simulation=env.physics.sim)
            self._wm_update = CUDACall(self.wm_update, pool=pool)
            self._reward_update = CUDACall(self.reward_update, pool=pool, capture_after=self._wm_update)
            self._actor_update = CUDACall(self.actor_update, pool=pool, capture_after=self._reward_update)
            # The critic batch masks the actor's exported feature buffer in place and returns a view of it.
            self._critic_batch = CUDACall(self.critic_batch, pool=pool, capture_after=self._actor_update, borrowed_inputs=(0,))
            self._critic_update = CUDACall(self.critic_update, pool=pool, capture_after=self._critic_batch)

    # ------------------------------------------------------------------ collection
    @torch.no_grad()
    def act(self, history, frames):
        """The actor's action and the exploration noise to execute it with."""
        goal = self.motion.goal(frames, self.cfg.future)
        self.actor.obs_norm.update(history[:, -1])
        self.actor.goal_norm.update(goal)
        action = self.actor(history, goal)
        return action, self.cfg.exploration * torch.randn_like(action)

    @torch.no_grad()
    def observe_transition(self, row):
        """Normalizer statistics count every fresh simulator transition exactly once."""
        s, s1 = unpack(row["s"]), unpack(row["s1"])
        features = networks.transition_features(s, row["u"], self.motion.key(row["f"]), row["context"],
                                                self.mdp.surface_features(s, row["f"]))  # fmt: skip
        self.wm.in_norm.update(features, row["valid"])
        self.wm.out_norm.update(self.wm.targets(s, s1), row["valid"])
        self.wm.error_norm.update(self.wm.increments(s, s1), row["valid"])
        self.reward_head.norm.update(features, row["valid"])

    @torch.no_grad()
    def collect_step(self):
        """One fixed-shape control step of every world, including resets and randomization."""
        env = self.env
        row = env.step(*self._act(env.history, env.f))
        ok = env.physics.ok
        # After a fatal fault keep the evidence in place: no resets, and nothing is admitted.
        env.reset(row["cut"] & ok)
        row["valid"] = row["valid"] & env.physics.ok
        self._observe_transition(row)
        return row

    def collect(self, steps: int):
        rows = []
        for _ in range(steps):
            row = self._collect()
            self.rejected_transitions += self.env.physics.check()
            self.replay.add(row)
            rows.append(row)
        return rows

    # ------------------------------------------------------------------ world model
    def wm_loss(self, w):
        """Multistep prediction loss over replay windows, in units of each coordinate's increment scale."""
        initial = unpack(w["s"][:, 0])
        s = initial
        scale = torch.maximum(self.wm.error_norm.std(), self.wm.error_floor)
        J, body = self.wm.J, self.control.body
        hand = torch.cat((self.control.driver, self.control.mimic))
        errors, groups = [], []
        for i in range(w["mask"].shape[1]):
            valid = w["mask"][:, i].bool()
            # A terminal transition is still supervised. Later rows belong to another episode:
            # evaluate a valid stand-in there instead of extrapolating through foreign commands.
            s = tuple(select(valid, a, b) for a, b in zip(s, initial))
            u, context = select(valid, w["u"][:, i], w["u"][:, 0]), select(valid, w["context"][:, i], w["context"][:, 0])
            f = torch.where(valid, w["f"][:, i], w["f"][:, 0])
            s = self.wm(s, u, self.motion.key(f), context, self.mdp.surface_features(s, f))
            e = (self.wm.error(s, unpack(select(valid, w["s1"][:, i], w["s1"][:, 0]))) / scale).square()
            errors.append(e.mean(-1))
            groups.append(torch.stack((e[..., :12].mean(-1), torch.cat((e[..., 12 + body], e[..., 12 + J + body]), -1).mean(-1),
                                       torch.cat((e[..., 12 + hand], e[..., 12 + J + hand]), -1).mean(-1),
                                       e[..., 12 + 2 * J :].mean(-1)), -1).detach())  # fmt: skip
        mask = w["mask"]
        # Summed over time, averaged over windows; all coordinates weigh equally.
        loss = (torch.stack(errors, 1) * mask).sum() / mask.shape[0]
        group_losses = (torch.stack(groups, 1) * mask[..., None]).sum((0, 1)) / mask.shape[0]
        return loss, group_losses

    def wm_update(self, w):
        self.wm_opt.zero_grad(set_to_none=True)
        loss, groups = self._wm_loss(w)
        norm, ok = guarded_step(loss, self.wm, self.wm_opt, self.cfg)
        return loss.detach(), groups.detach(), norm, ok

    def contact_reward_loss(self, w):
        """MSE on measured per-finger force scores, including terminal transitions.

        Inputs follow detached world-model rollouts so they match what the head sees in
        imagination. Reference intent gates the reward in the MDP, never these physical labels.
        """
        initial = tuple(x.detach() for x in unpack(w["s"][:, 0]))
        s = initial
        steps = w["mask"].shape[1]
        losses, absolute, targets, predictions = [], [], [], []
        for i in range(steps):
            valid = w["mask"][:, i].bool()
            s = tuple(select(valid, a, b) for a, b in zip(s, initial))
            u, context = select(valid, w["u"][:, i], w["u"][:, 0]), select(valid, w["context"][:, i], w["context"][:, 0])
            f = torch.where(valid, w["f"][:, i], w["f"][:, 0])
            target = select(valid, w["finger_score"][:, i], w["finger_score"][:, 0])
            key, surface = self.motion.key(f), self.mdp.surface_features(s, f).detach()
            prediction = self.reward_head(s, u, key, context, surface)
            losses.append((prediction - target).square().mean(-1))
            absolute.append((prediction - target).abs().mean(-1).detach())
            targets.append(target.mean(-1))
            predictions.append(prediction.detach().mean(-1))
            if i + 1 < steps:
                with torch.no_grad():
                    s = self.wm(s, u, key, context, surface)
        mask = w["mask"]
        mean = lambda xs: (torch.stack(xs, 1) * mask).sum() / mask.sum().clamp_min(1)  # noqa: E731
        return mean(losses), mean(absolute), mean(targets), mean(predictions)

    def reward_update(self, w):
        self.reward_opt.zero_grad(set_to_none=True)
        loss, mae, target, prediction = self._contact_reward_loss(w)
        norm, ok = guarded_step(loss, self.reward_head, self.reward_opt, self.cfg)
        return loss.detach(), mae, target, prediction, norm, ok

    def fit_wm(self) -> dict:
        w = self.replay.windows(self.cfg.wm_batch, self.cfg.wm_window)
        loss, groups, norm, ok = self._wm_update(w)
        self.admit(ok, "wm", loss, norm)
        k = self.cfg.wm_window
        metrics = {"wm/loss": loss / k, "grad/wm": norm.clone(),
                   **{f"wm/{name}": value / k for name, value in zip(("root", "body", "hand", "object"), groups)}}  # fmt: skip
        loss, mae, target, prediction, norm, ok = self._reward_update(w)
        self.admit(ok, "contact_reward", loss, norm)
        metrics.update({"contact_reward/loss": loss, "contact_reward/mae": mae, "contact_reward/target_mean": target,
                        "contact_reward/prediction_mean": prediction, "grad/contact_reward": norm.clone()})  # fmt: skip
        return {k: v.detach().clone() for k, v in metrics.items()}

    # ------------------------------------------------------------------ actor
    def imagine_step(self, s, f, f1, previous, hand, history, context, key, goal, z, fk, episode_start, active, fallback):
        """One imagined control step from frame ``f`` to ``f1``.

        Lanes whose clip has ended (``active`` false) are padding: the models are evaluated on the
        real start state instead of an extrapolated one, and the lane keeps its terminal state,
        with zero reward and certain termination.
        """
        features = self.mdp.critic_input(s, f, previous, hand, context, goal, fk)
        surface = self.mdp.surface_features(s, f, fk)
        before = (s, hand, previous, history, fk)
        s0, hand0, previous0, history0, f0, goal0, z0, surface0 = fallback
        s = tuple(select(active, a, b) for a, b in zip(s, s0))
        hand, previous = select(active, hand, hand0), select(active, previous, previous0)
        history, goal, z = select(active, history, history0), select(active, goal, goal0), select(active, z, z0)
        surface, f1 = select(active, surface, surface0), torch.where(active, f1, f0 + 1)

        qref = self.motion.q[f1]
        action = self.control.bound_action(self.actor(history, goal, z), qref)
        target, hand1 = self.control(action, qref, hand, self.mdp.layout.offsets(context))
        u = self.control.effective(target)
        finger_scores = self.reward_head(s, u, key, context, surface)
        s = self.wm(s, u, key, context, surface)
        fk = self.mdp.fk(s)
        reward, _ = self.mdp.reward(s, f1, action, previous, finger_scores, episode_start, fk)
        done = self.mdp.soft_termination(s, f1)
        obs = self.mdp.add_noise(self.mdp.obs(s, f1, action, hand1, context, fk))
        history = torch.cat((history[:, 1:], obs[:, None]), 1)

        s_b, hand_b, previous_b, history_b, fk_b = before
        s = tuple(select(active, a, b) for a, b in zip(s, s_b))
        fk = tuple(select(active, a, b) for a, b in zip(fk, fk_b))
        hand1, action = select(active, hand1, hand_b), select(active, action, previous_b)
        history = select(active, history, history_b)
        return s, hand1, action, history, fk, features, torch.where(active, reward, 0.0), torch.where(active, done, 1.0)

    def imagine(self, start):
        """Differentiable rollout of the actor through the frozen world model from replayed states.

        Returns (loss, critic features (H+1, B, D), rewards (H, B), dones (H, B), bootstrap, survival).
        """
        c, m = self.cfg, self.motion
        s, f = unpack(start["s"]), start["f"]
        hand, previous, history, context = start["hand"], start["prev_action"], start["history"], start["context"]
        end = m.clip_end[m.clip_id[f]]
        frames = torch.minimum(f[None] + torch.arange(c.horizon + 1, device=f.device)[:, None], end)
        goals = m.goal(frames.flatten(), c.future).unflatten(0, frames.shape)
        latents = self.actor.encode(goals[:-1].flatten(0, 1)).unflatten(0, (c.horizon, f.shape[0]))
        key, fk = m.key(f), self.mdp.fk(s)
        fallback = (s, hand, previous, history, f, goals[0], latents[0], self.mdp.surface_features(s, f, fk))
        rewards, dones, features = [], [], []
        for t in range(c.horizon):
            s, hand, previous, history, fk, x, r, d = self._imagine_step(
                s, frames[t], frames[t + 1], previous, hand, history, context, key, goals[t], latents[t], fk,
                start["age"] + t == 0, frames[t] < end, fallback)  # fmt: skip
            features.append(x)
            rewards.append(r)
            dones.append(torch.where(frames[t + 1] >= end, 1.0, d))  # the clip's last transition bootstraps zero
        features.append(self.mdp.critic_input(s, frames[-1], previous, hand, context, goals[-1], fk))
        rewards, dones = torch.stack(rewards), torch.stack(dones)
        bound = discounted_return_bound(end - frames[-1], self.mdp.max_step_reward, c.gamma)
        bootstrap = bounded_bootstrap(self.target_critic(features[-1]), bound)
        objective, survival = actor_objective(rewards, dones, bootstrap, c.gamma)
        return -objective.mean(), torch.stack(features), rewards, dones, bootstrap, survival

    def actor_update(self, start):
        self.pi_opt.zero_grad(set_to_none=True)
        loss, features, rewards, dones, bootstrap, survival = self.imagine(start)
        norm, ok = guarded_step(loss, self.actor, self.pi_opt, self.cfg, value_clip=self.cfg.pi_clip_value)
        return loss.detach(), features.detach(), rewards.detach(), dones.detach(), bootstrap.detach(), survival.detach(), norm, ok

    @torch.no_grad()
    def update_ema(self):
        """Copy on the first update, then average parameters and buffers."""
        target = list(self.ema.parameters()) + list(self.ema.buffers())
        source = [x.detach() for x in list(self.actor.parameters()) + list(self.actor.buffers())]
        if self.ema_updates == 0:
            for a, b in zip(target, source):
                a.copy_(b)
        else:
            torch._foreach_lerp_(target, source, 1 - self.cfg.ema)
        self.ema_updates += 1

    # ------------------------------------------------------------------ critic
    @torch.no_grad()
    def critic_batch(self, features, rewards, dones, valid, bounds):
        """Lambda-return targets for an imagined rollout. Masks ``features`` in place.

        valid (H, B): transition t is real (its clip has not ended). bounds (H+1, B): attainable
        return of every state. Returns (critic inputs (H*B, D), targets (H*B,)).
        """
        state_valid = torch.cat((valid[:1], valid), 0)
        self.critic.norm.update(features, state_valid)
        flat = features.flatten(0, 1)
        values = []
        for rows, mask in zip(flat.split(RunningNorm.CHUNK_ROWS), state_valid.flatten().split(RunningNorm.CHUNK_ROWS)):
            rows.copy_(torch.where(mask[:, None], rows, 0.0))
            values.append(self.target_critic(rows))
        values = bounded_bootstrap(torch.cat(values).reshape(state_valid.shape), bounds)
        targets = gae_targets(values, torch.where(valid, rewards, 0.0), torch.where(valid, dones, 1.0), self.cfg.gamma, self.cfg.lam)
        # Exact in real arithmetic; enforced against a one-ULP overshoot of the discounted sum.
        targets = torch.where(valid, targets.clamp(max=bounds[:-1]), 0.0)
        return features[:-1].flatten(0, 1), targets.flatten()

    def critic_update(self, x, y, valid):
        self.v_opt.zero_grad(set_to_none=True)
        error = (self.critic(torch.where(valid[:, None], x, 0.0)) - y).square()
        loss = torch.where(valid, error, 0.0).sum() / valid.sum().clamp_min(1)
        norm, ok = guarded_step(loss, self.critic, self.v_opt, self.cfg, enabled=valid.any())
        return loss.detach(), norm, ok

    def fit_critic(self, X, Y, valid):
        """Fixed targets, shuffled epochs of equally sized minibatches."""
        losses, norms = [], []
        for _ in range(self.cfg.critic_epochs):
            for idx in torch.randperm(X.shape[0], device=X.device).tensor_split(self.cfg.critic_minibatches):
                loss, norm, ok = self._critic_update(X[idx], Y[idx], valid[idx])
                self.admit(ok, "critic", loss, norm)
                losses.append(loss.clone())
                norms.append(norm.clone())
        return torch.stack(losses).mean(), torch.stack(norms).max()

    @torch.no_grad()
    def update_target_critic(self):
        """In-place EMA of weights and normalizer statistics; storage stays fixed for captured graphs."""
        target = list(self.target_critic.parameters()) + list(self.target_critic.buffers())
        source = [x.detach() for x in list(self.critic.parameters()) + list(self.critic.buffers())]
        torch._foreach_lerp_(target, source, 1 - self.cfg.target_critic_ema)

    def fit_policy(self) -> dict:
        c, m = self.cfg, self.motion
        start = self.replay.starts(c.pi_batch, c.history)
        frozen = (self.wm, self.reward_head)
        for model in frozen:
            model.zero_grad(set_to_none=True)
            model.requires_grad_(False)
        try:
            loss, X, rewards, dones, bootstrap, survival, norm, ok = self._actor_update(start)
            self.admit(ok, "actor", loss, norm)
        finally:
            for model in frozen:
                model.requires_grad_(True)
        self.update_ema()
        f, end = start["f"], m.clip_end[m.clip_id[start["f"]]]
        steps = torch.arange(c.horizon + 1, device=f.device)[:, None]
        valid = (f[None] + steps[:-1]) < end
        bounds = discounted_return_bound(end[None] - f[None] - steps, self.mdp.max_step_reward, c.gamma)
        metrics = {"actor/loss": loss.clone(), "actor/imagined_reward": rewards.sum() / valid.sum(),
                   "actor/bootstrap_mean": bootstrap.mean(), "actor/imagined_survival": survival.mean(),
                   "actor/valid_fraction": valid.float().mean(), "grad/actor": norm.clone()}  # fmt: skip
        X, Y = self._critic_batch(X, rewards, dones, valid, bounds)
        valid = valid.flatten()
        value_loss, value_norm = self.fit_critic(X, Y, valid)
        # One target update after all minibatches: rollouts and targets above saw one fixed target.
        self.update_target_critic()
        metrics.update({"critic/loss": value_loss, "critic/target_mean": (Y * valid).sum() / valid.sum(),
                        "critic/target_max": Y.masked_fill(~valid, -torch.inf).max(), "grad/critic": value_norm})  # fmt: skip
        return {k: v.detach().clone() for k, v in metrics.items()}

    # ------------------------------------------------------------------ bookkeeping
    def admit(self, ok, model: str, loss, norm):
        """Count an update that was skipped as invalid. A single one is an outlier batch and training
        continues; ``max_skipped_updates`` in a row stop it and save the state they left untouched."""
        count = self.skipped.setdefault(model, [0, 0])
        if bool(ok):
            count[1] = 0
            return
        count[0] += 1
        count[1] += 1
        print(f"Skipped an invalid {model} update at iteration {self.iteration + 1}: loss={float(loss):.6g}, "
              f"gradient norm={float(norm):.6g}", flush=True)  # fmt: skip
        if count[1] >= self.cfg.max_skipped_updates:
            path = Path(self.cfg.log_dir) / "failure.pt"
            self.save(path)
            raise RuntimeError(f"{count[1]} consecutive invalid {model} updates; state saved to {path}")

    def set_learning_rate(self):
        c = self.cfg
        lr = lr_at(self.iteration, c.lr, c.lr_decay_iters)
        mult = c.pi_lr_mult
        if self.decay_at is not None:
            mult += (1 - mult) * min(1.0, (self.iteration - self.decay_at) / c.lr_decay_len)
        for name in OPTIMIZERS:
            for group in getattr(self, name).param_groups:
                group["lr"].fill_(lr * (mult if name == "pi_opt" else 1.0))
        return lr, mult

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = dict(
            format_version=1, config=asdict(self.cfg), iteration=self.iteration, ema_updates=self.ema_updates,
            decay_at=self.decay_at, signature=self.task.signature, object_names=self.motion.object_names,
            torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if self.cfg.device.startswith("cuda") else None,
            models={k: getattr(self, k).state_dict() for k in MODELS},
            optimizers={k: getattr(self, k).state_dict() for k in OPTIMIZERS})  # fmt: skip
        temporary = path.with_suffix(".tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(path)

    def load(self, path, weights_only: bool = False):
        """Restore a checkpoint. Episodes and replay start fresh, so a resume is not bit-exact."""
        checkpoint = torch.load(path, map_location=self.cfg.device, weights_only=False)
        if checkpoint.get("format_version") != 1 or checkpoint["object_names"] != self.motion.object_names:
            raise ValueError("Incompatible checkpoint format or object keys")
        if not weights_only and checkpoint["signature"] != self.task.signature:
            raise ValueError("Dataset, reference processing or physical assets differ from the checkpoint")
        for name, state in checkpoint["models"].items():
            getattr(self, name).load_state_dict(state)
        self.iteration, self.ema_updates, self.decay_at = checkpoint["iteration"], checkpoint["ema_updates"], checkpoint["decay_at"]
        if weights_only:
            return
        for name, state in checkpoint["optimizers"].items():
            load_optimizer(getattr(self, name), state)
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        if checkpoint["cuda_rng"]:
            torch.cuda.set_rng_state_all([s.cpu() for s in checkpoint["cuda_rng"]])

    # ------------------------------------------------------------------ training loop
    def train(self):
        c = self.cfg
        out = Path(c.log_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text(json.dumps(asdict(c), indent=2) + "\n")
        (out / "reset_guard.json").write_text(json.dumps(self.env.reset_summary, indent=2) + "\n")
        cuda = c.device.startswith("cuda")
        with Tracker(c) as tracker, (out / "metrics.jsonl").open("a") as log:
            begin = time.monotonic()
            self.collect(c.warmup_steps)
            print(f"startup: replay warm-up took {time.monotonic() - begin:.1f} s", flush=True)
            sums, count = {}, 0
            episodes = torch.zeros(4, device=c.device)  # return sum, length sum, count, normalized-return sum
            window_start, window_iteration = time.monotonic(), self.iteration
            if cuda:
                torch.cuda.reset_peak_memory_stats()
            for it in range(self.iteration, c.iterations):
                lr, mult = self.set_learning_rate()
                rows = self.collect(c.collect_steps)
                wm_metrics = [self.fit_wm() for _ in range(c.wm_updates)]
                metrics = {k: torch.stack([m[k] for m in wm_metrics]).mean() for k in wm_metrics[0]}
                metrics.update(self.fit_policy())
                for name, key in (("reward", "reward"), ("failure_fraction", "term"), ("hand_contact_fraction", "contact"),
                                  ("finger_score_mean", "finger_score"), ("valid_fraction", "valid")):  # fmt: skip
                    metrics["sim/" + name] = torch.stack([r[key].float() for r in rows]).mean()
                for key in rows[0]["terms"]:
                    metrics["reward/" + key] = torch.stack([r["terms"][key] for r in rows]).mean()
                for i, key in enumerate(FAILURE_REASONS):
                    metrics["failure/" + key] = torch.stack([r["reasons"][:, i].float() for r in rows]).mean()
                for row in rows:
                    cut, length = row["cut"], row["age"] + 1
                    normalized = row["episode_return"] / length / self.mdp.max_step_reward
                    episodes += torch.stack((torch.where(cut, row["episode_return"], 0.0).sum(), torch.where(cut, length, 0).sum().float(),
                                             cut.sum().float(), torch.where(cut, normalized, 0.0).sum()))  # fmt: skip
                for key, value in metrics.items():
                    sums[key] = sums.get(key, 0.0) + value
                count += 1
                self.iteration = it + 1

                if self.iteration % c.log_every == 0 or self.iteration == 1:
                    keys = list(sums)
                    record = dict(zip(keys, (torch.stack([sums[k] for k in keys]) / count).cpu().tolist()))
                    sums, count = {}, 0
                    done = float(episodes[2])
                    normalized_return = float(episodes[3]) / max(done, 1.0)
                    elapsed = time.monotonic() - window_start
                    iterations = self.iteration - window_iteration
                    record.update({
                        "iteration": self.iteration, "lr": lr, "actor/lr": lr * mult,
                        "episode/completed": done, "episode/return_mean": float(episodes[0]) / max(done, 1.0),
                        "episode/length_mean": float(episodes[1]) / max(done, 1.0), "episode/return_normalized": normalized_return,
                        "sim/rejected_transitions": self.rejected_transitions,
                        **{f"skipped/{model}": count[0] for model, count in self.skipped.items()},
                        "time/iteration_seconds": elapsed / iterations,
                        "time/real_transitions_per_second": iterations * c.collect_steps * c.num_envs / elapsed})  # fmt: skip
                    # The actor's learning-rate boost ends once tracking works or after a fixed budget.
                    if self.decay_at is None and ((done and normalized_return >= c.lr_decay_ret) or self.iteration >= c.lr_decay_by):
                        self.decay_at = self.iteration
                    episodes.zero_()
                    if cuda:
                        free, total = torch.cuda.mem_get_info()
                        record.update({"gpu/peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                                       "gpu/device_used_gib": (total - free) / 2**30})  # fmt: skip
                        for key, value in zip(("collision_candidates", "contacts", "constraints_per_world"), self.env.physics.capacity_peak.tolist()):
                            record["physics/peak_" + key] = value
                        torch.cuda.reset_peak_memory_stats()
                    log.write(json.dumps(record) + "\n")
                    log.flush()
                    tracker.log(record, self.iteration)
                    print(json.dumps(record), flush=True)
                    window_start, window_iteration = time.monotonic(), self.iteration
                if self.iteration % c.save_every == 0:
                    self.save(out / f"checkpoint_{self.iteration:06}.pt")
            self.save(out / "last.pt")
