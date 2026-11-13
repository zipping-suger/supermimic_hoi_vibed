import json
from dataclasses import replace
from pathlib import Path

import torch

from supermimic_hoi.evaluate import evaluate, load_runner
from supermimic_hoi.runner import MODELS, Runner
from supermimic_hoi.state import unpack


def test_actor_gradient_flows_through_dynamics_reward_head_and_finger_filter(tiny_config):
    runner = Runner(tiny_config)
    runner.collect(tiny_config.warmup_steps)
    # Give the zero-initialized output layers a response so the gradient paths are visible.
    torch.nn.init.normal_(runner.wm.net[-1].weight, std=0.05)
    torch.nn.init.normal_(runner.actor.dec[-1].weight, std=0.05)
    start = runner.replay.starts(tiny_config.pi_batch, tiny_config.history)
    loss, features, rewards, dones, bootstrap, survival = runner.imagine(start)
    assert features.shape[:2] == (tiny_config.horizon + 1, tiny_config.pi_batch)
    assert rewards.shape == dones.shape == (tiny_config.horizon, tiny_config.pi_batch)
    assert (dones >= 0).all() and (dones <= 1).all() and torch.isfinite(loss)
    loss.backward()
    body = runner.actor.dec[-1].weight.grad[:29].abs().sum()
    hand = runner.actor.dec[-1].weight.grad[29:].abs().sum()
    assert body > 0 and hand > 0 and runner.actor.enc[0].weight.grad.abs().sum() > 0


def test_padding_after_the_clip_end_is_inert(tiny_config):
    runner = Runner(tiny_config)
    runner.collect(tiny_config.warmup_steps)
    start = runner.replay.starts(tiny_config.pi_batch, tiny_config.history)
    m = runner.motion
    # Move every start to two frames before its clip's end: only two real transitions remain.
    end = m.clip_end[m.clip_id[start["f"]]]
    start["f"] = end - 2
    start["s"] = torch.cat([x.flatten(1) for x in m.ref_state(start["f"])], -1)
    assert torch.equal(start["s"], torch.cat([x.flatten(1) for x in unpack(start["s"])], -1))
    with torch.no_grad():
        _, features, rewards, dones, _, survival = runner.imagine(start)
    assert (rewards[2:] == 0).all() and (dones[1:] == 1).all() and (survival == 0).all()
    assert torch.equal(features[3], features[-1])  # the terminal state is frozen, not extrapolated


def test_train_checkpoint_resume_and_evaluate(tiny_config):
    runner = Runner(tiny_config)
    runner.train()
    out = Path(tiny_config.log_dir)
    records = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    assert [r["iteration"] for r in records] == [1, 2]
    assert all(torch.isfinite(torch.tensor(v)) for r in records for v in r.values())
    assert runner.ema_updates == 2 and runner.replay.size == tiny_config.warmup_steps + 2 * tiny_config.collect_steps

    resumed = Runner(replace(tiny_config, resume=str(out / "last.pt"), iterations=3))
    assert resumed.iteration == 2
    for name in MODELS:
        for a, b in zip(getattr(runner, name).state_dict().values(), getattr(resumed, name).state_dict().values()):
            assert torch.equal(a, b), name
    resumed.train()
    assert resumed.iteration == 3

    evaluator = load_runner(str(out / "last.pt"), device="cpu")
    assert evaluator.env.N == 2
    report, trajectories = evaluate(evaluator, record=[tiny_config.clips[0]])
    assert report["clips"] == 2 and 0 <= report["success_rate"] <= 1
    for result in report["results"]:
        assert 1 <= result["steps"] <= result["reference_steps"]
        assert result["completed"] == (result["steps"] == result["reference_steps"] and not result["failure_reasons"])
    states = trajectories[tiny_config.clips[0]]
    assert states.shape == (report["results"][0]["steps"] + 1, 142)


def test_checkpoint_rejects_different_physical_assets(tiny_config, tmp_path):
    runner = Runner(tiny_config)
    path = tmp_path / "checkpoint.pt"
    runner.save(path)
    checkpoint = torch.load(path, weights_only=False)
    checkpoint["signature"] = "0" * 64
    torch.save(checkpoint, path)
    try:
        Runner(replace(tiny_config, resume=str(path)))
    except ValueError as error:
        assert "differ from the checkpoint" in str(error)
    else:
        raise AssertionError("A checkpoint for different assets was accepted")
