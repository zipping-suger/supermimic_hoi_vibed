import torch

from supermimic_hoi.env import make_task
from supermimic_hoi.state import pack, unpack


def test_reference_library(data, robot, two_clips):
    from supermimic_hoi.motion import Motion

    m = Motion(data / "weave600", robot, ["smallbox", "woodchair"], clips=two_clips)
    assert m.num_frames == int(m.clip_len.sum()) and m.clip_object.tolist() == [0, 1]
    frames = torch.arange(m.num_frames)
    assert (m.q >= m.kin.q_lo - 1e-6).all() and (m.q <= m.kin.q_hi + 1e-6).all()
    assert m.goal(frames[:7], 10).shape == (7, m.goal_dim(10))
    # The goal of a clip's last frame repeats that frame instead of reading the next clip.
    last = m.clip_end[:1]
    assert torch.equal(m.next_frame(last), last)
    assert torch.allclose(m.goal(last, 3).unflatten(-1, (3, -1))[0, 0], m.goal(last, 3).unflatten(-1, (3, -1))[0, 2])
    # Weave's clips begin with an approach: no finger is intended to touch at frame zero.
    assert m.finger_intent[m.clip_start].sum() == 0 and m.finger_intent.sum() > 0
    table, count = m.pregrasp_frames()
    for clip in range(2):
        frame = int(table[clip, 0])
        assert m.clip_start[clip] <= frame < m.clip_end[clip] and m.hand_intent(torch.tensor(frame)).sum() == 0
        assert m.hand_intent(torch.tensor(frame + 10)).sum() > 0


def test_reset_reproduces_the_reference_and_steps_are_consistent(tiny_config):
    task = make_task(tiny_config)
    env, m, mdp = task.env, task.motion, task.mdp
    everyone = torch.ones(env.N, dtype=torch.bool)
    frames = m.clip_start[env.env_clip] + torch.tensor([0, 40, 0, 40])
    env.reset(everyone, frames)
    assert (pack(env.physics.state()) - pack(m.ref_state(frames))).abs().max() < 1e-5
    assert env.age.sum() == 0 and env.previous_action.abs().sum() == 0
    # At the reference state every tracking bonus is at its weight.
    reward, terms = mdp.reward(m.ref_state(frames), frames, torch.zeros(4, 41), torch.zeros(4, 41), torch.zeros(4, 10),
                               torch.ones(4, dtype=torch.bool))  # fmt: skip
    for name in ("body_pos", "body_ori", "body_velocity", "body_angular_velocity", "root_pos", "root_ori", "object_pos", "object_ori"):
        assert torch.allclose(terms[name], torch.full((4,), 0.02 * getattr(tiny_config.reward, name)), atol=1e-5), name
    assert (reward <= mdp.max_step_reward + 1e-6).all()

    before = env.history.clone()
    row = env.step(task.controller.reference_action(m.q[m.next_frame(env.f)]))
    assert torch.equal(row["f"], frames) and torch.equal(env.f, frames + 1) and (env.age == 1).all()
    assert torch.equal(row["o"], before[:, -1]) and torch.equal(env.history[:, :-1], before[:, 1:])
    assert all(torch.isfinite(v).all() for v in row.values() if torch.is_tensor(v) and v.is_floating_point())
    assert torch.allclose(pack(unpack(row["s1"])), row["s1"])
    # Only the masked worlds restart.
    mask = torch.tensor([True, False, False, True])
    env.reset(mask)
    assert (env.age == torch.tensor([0, 1, 1, 0])).all()
    assert (env.reset_map[env.f[mask]] == env.f[mask]).all()
    assert torch.equal(m.clip_id[env.f], env.env_clip)


def test_exploration_noise_is_executed_but_not_recorded(tiny_config):
    clean, noisy = make_task(tiny_config), make_task(tiny_config)
    m = clean.motion
    everyone = torch.ones(4, dtype=torch.bool)
    frames = m.clip_start[clean.env.env_clip] + 20
    clean.env.reset(everyone, frames)
    noisy.env.reset(everyone, frames)
    action = 0.1 * torch.randn(4, 41)
    row, noisy_row = clean.env.step(action), noisy.env.step(action, 0.5 * torch.randn(4, 41))
    # The drives received different targets, yet the recorded action and its costs are the policy's own.
    assert not torch.allclose(row["u"], noisy_row["u"])
    assert torch.equal(row["action"], noisy_row["action"]) and torch.equal(clean.env.previous_action, noisy.env.previous_action)
    for name in ("action_rate", "hand_saturation"):
        assert torch.equal(row["terms"][name], noisy_row["terms"][name])
    assert torch.equal(clean.env.history[:, -1, 112:153], noisy.env.history[:, -1, 112:153])
    assert not torch.equal(clean.env.hand_targets, noisy.env.hand_targets)  # the finger filter follows the drives


def test_termination_follows_the_pose_errors_in_simulation_and_imagination(tiny_config):
    task = make_task(tiny_config)
    mdp, m = task.mdp, task.motion
    frame = (m.hand_intent(torch.arange(m.num_frames)).sum(-1) == 2).nonzero()[0]
    state = m.ref_state(frame)
    # On the reference nothing fails; contact is not a termination condition.
    failed, reasons = mdp.hard_termination(state, frame)
    assert not bool(failed) and not reasons.any() and float(mdp.soft_termination(state, frame)) < 0.01
    dropped = (*state[:6], state[6] + torch.tensor([0.0, 0.0, -0.4]), *state[7:])
    failed, reasons = mdp.hard_termination(dropped, frame)
    assert bool(failed) and reasons[0].tolist() == [False, False, True, False]
    assert float(mdp.soft_termination(dropped, frame)) > 0.99
