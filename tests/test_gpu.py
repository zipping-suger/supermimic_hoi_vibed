"""MuJoCo-Warp backend and CUDA-graph capture. Skipped without a CUDA device."""

from dataclasses import replace

import pytest
import torch

from supermimic_hoi.env import make_task
from supermimic_hoi.runner import Runner
from supermimic_hoi.state import pack

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def gpu(config, **changes):
    return replace(config, device="cuda:0", backend="warp", **changes)


def test_warp_matches_cpu_physics_and_resets_only_masked_worlds(tiny_config):
    cpu, warp = make_task(tiny_config), make_task(gpu(tiny_config))
    frames = cpu.motion.clip_start[torch.tensor([0, 0, 1, 1])] + torch.tensor([0, 30, 60, 90])
    everyone = torch.ones(4, dtype=torch.bool)
    cpu.env.reset(everyone, frames)
    warp.env.reset(everyone.cuda(), frames.cuda())
    assert (pack(warp.env.physics.state()).cpu() - pack(cpu.env.physics.state())).abs().max() < 1e-4
    for _ in range(5):
        action = cpu.controller.reference_action(cpu.motion.q[cpu.motion.next_frame(cpu.env.f)])
        row_cpu, row_warp = cpu.env.step(action), warp.env.step(action.cuda())
        assert warp.env.physics.check() == 0 and bool(row_warp["valid"].all())
    s_cpu, s_warp = cpu.env.physics.state(), tuple(x.cpu() for x in warp.env.physics.state())
    # Different contact solvers; joint and object positions still agree closely after 0.1 s.
    assert (s_cpu[0] - s_warp[0]).abs().max() < 5e-3 and (s_cpu[4] - s_warp[4]).abs().max() < 2e-2
    assert (s_cpu[6] - s_warp[6]).abs().max() < 5e-3
    assert torch.allclose(row_cpu["reward"], row_warp["reward"].cpu(), atol=5e-3)

    before = pack(warp.env.physics.state())
    mask = torch.tensor([True, False, False, True], device="cuda:0")
    warp.env.reset(mask)
    after = pack(warp.env.physics.state())
    assert torch.equal(after[~mask], before[~mask]) and not torch.equal(after[mask], before[mask])
    assert (warp.env.age == torch.tensor([0, 5, 5, 0], device="cuda:0")).all()


def test_an_excursion_is_rejected_contained_and_reset(tiny_config):
    task = make_task(gpu(tiny_config, max_joint_speed=1e-3))  # every moving world violates this limit
    env = task.env
    row = env.step(torch.zeros(4, task.robot.num_actions, device="cuda:0"))
    assert not bool(row["valid"].any()) and bool(row["term"].all()) and float(row["reward"].abs().sum()) == 0
    assert env.physics.check() == 4 and bool(env.physics.ok)
    assert torch.isfinite(pack(env.physics.state())).all()


def test_a_nonfinite_target_is_fatal(tiny_config):
    task = make_task(gpu(tiny_config))
    env = task.env
    action = torch.zeros(4, task.robot.num_actions, device="cuda:0")
    action[2, 0] = float("nan")
    row = env.step(action)
    assert row["valid"].tolist() == [True, True, False, True] and not bool(env.physics.ok)
    with pytest.raises(RuntimeError, match="nonfinite controller target"):
        env.physics.check()


def test_training_with_randomization_and_cuda_graphs(tiny_config):
    config = gpu(tiny_config, randomize=True, cuda_graph=True, amp="bf16", iterations=6, warmup_steps=8)
    runner = Runner(config)
    runner.train()
    assert runner.iteration == 6 and runner.rejected_transitions == 0
    calls = {name: getattr(runner, name) for name in ("_collect", "_wm_update", "_reward_update", "_actor_update",
                                                      "_critic_batch", "_critic_update")}  # fmt: skip
    for name, call in calls.items():
        assert call.graph is not None and call.replays > 0, name
    # One learning pool, one separate collection pool, and exactly one update per call.
    assert len({call.pool for name, call in calls.items() if name != "_collect"}) == 1 and calls["_collect"].pool is None
    assert calls["_collect"].calls == config.warmup_steps + 6 * config.collect_steps
    assert calls["_wm_update"].calls == 6 * config.wm_updates and calls["_actor_update"].calls == 6
    assert calls["_critic_update"].calls == 6 * config.critic_epochs * config.critic_minibatches
    for model in (runner.wm, runner.actor, runner.critic, runner.reward_head, runner.ema, runner.target_critic):
        assert all(torch.isfinite(p).all() for p in model.parameters())
    assert float(runner.wm.in_norm.count) == calls["_collect"].calls * config.num_envs


def test_a_world_that_takes_another_object_matches_the_worlds_built_with_it(tiny_config):
    task = make_task(gpu(tiny_config))
    env, m = task.env, task.motion
    everyone = torch.ones(4, dtype=torch.bool, device="cuda:0")
    body = env.physics.sim.mj_model.body("object/object").id
    # Two worlds were built with each object, so in every round two of them hold an object they took at a reset.
    for clip in (0, 1, 0):
        touching = (m.clip_id == clip) & (m.hand_intent(torch.arange(m.num_frames, device="cuda:0")).sum(-1) > 0)
        frames = env.reset_map[touching.nonzero()[0]].expand(4)
        env.reset(everyone, frames)
        entry = task.registry["objects"][int(m.clip_object[clip])]
        assert torch.allclose(env.physics.sim.model.body_mass[:, body], torch.full((4,), entry["mass"], device="cuda:0"))
        assert torch.allclose(env.context[:, :13], torch.tensor(entry["context"], device="cuda:0").expand(4, 13))
        for _ in range(3):
            row = env.step(task.controller.reference_action(m.q[m.next_frame(env.f)]))
            assert env.physics.check() == 0 and bool(row["valid"].all())
        # Same object, state and targets: the worlds agree up to the order in which contacts are summed,
        # which moves contact velocities by a few 1e-3. A world left with the other object differs by more than 1.
        s = pack(env.physics.state())
        assert (s - s[:1]).abs().max() < 5e-2
