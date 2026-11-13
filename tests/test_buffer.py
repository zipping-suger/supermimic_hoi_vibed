import torch

from supermimic_hoi.buffer import FIELDS, Replay


def make_row(step, num_envs, cut=None, valid=None):
    row = {k: torch.full((num_envs, 2), float(step)) for k in FIELDS}
    row["f"] = torch.full((num_envs,), step)
    row["age"] = torch.full((num_envs,), step)
    row["o"] = torch.full((num_envs, 3), float(step))
    row["cut"] = torch.zeros(num_envs, dtype=torch.bool) if cut is None else cut
    row["valid"] = torch.ones(num_envs, dtype=torch.bool) if valid is None else valid
    return row


def test_windows_stop_at_episode_boundaries_and_keep_the_terminal_transition():
    torch.manual_seed(0)
    replay = Replay(8, 2, 2, "cpu")
    for step in range(6):
        replay.add(make_row(step, 2, cut=torch.tensor([step == 2, False])))
    w = replay.windows(64, 4)
    for start, mask in zip(w["f"][:, 0].tolist(), w["mask"].tolist()):
        world0 = any(m == 0 for m in mask)  # only world 0 has a boundary
        if world0:
            assert start <= 2 and mask == [1.0] * (3 - start) + [0.0] * (1 + start)
    assert (w["mask"][:, 0] == 1).all()


def test_starts_clip_history_at_the_episode_and_buffer_start():
    torch.manual_seed(0)
    replay = Replay(4, 2, 2, "cpu")
    for step in range(6):  # wraps: steps 2..5 remain
        row = make_row(step, 2)
        row["age"] = torch.full((2,), step - 4 if step >= 4 else step)  # a new episode began at step 4
        replay.add(row)
    out = replay.starts(64, 3)
    for f, history in zip(out["f"].tolist(), out["history"][:, :, 0].tolist()):
        expected = {2: [2, 2, 2], 3: [2, 2, 3], 4: [4, 4, 4], 5: [4, 4, 5]}[f]
        assert history == expected


def test_sampling_draws_clips_uniformly_and_skips_rejected_transitions():
    torch.manual_seed(0)
    replay = Replay(16, 4, 2, "cpu")
    for step in range(16):
        # Clip 1 (worlds 2, 3) is mostly rejected physics; it must keep half of the samples.
        replay.add(make_row(step, 4, valid=torch.tensor([True, True, step == 3, False])))
    age, env = replay.sample(4000, 16)
    assert replay.data["valid"][replay.row(age), env].all()
    assert abs(float((env >= 2).float().mean()) - 0.5) < 0.05
    assert set(env[env >= 2].tolist()) == {2} and set(age[env >= 2].tolist()) == {3}
