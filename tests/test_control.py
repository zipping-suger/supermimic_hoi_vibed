import torch

from supermimic_hoi.control import Controller


def test_body_follows_the_reference_and_fingers_are_absolute(robot):
    torch.manual_seed(0)
    c = Controller(robot)
    lo, hi = c.limits.unbind(-1)
    qref = lo + torch.rand(3, 53) * (hi - lo)
    hand = c.seed(qref)
    zero_offset = torch.zeros(3, 53)

    action = torch.zeros(3, 41)
    target, _ = c(action, qref, hand, zero_offset)
    assert torch.allclose(target[:, c.body], qref[:, c.body])

    action[:, :29] = 1.0
    target, _ = c(c.bound_action(action, qref), qref, hand, zero_offset)
    assert torch.allclose(target[:, c.body], qref[:, c.body] + c.scale[c.body], atol=1e-6)

    # A constant finger command converges to the mapped joint target; mimics follow their driver.
    action = torch.cat((torch.zeros(3, 29), torch.full((3, 12), 0.5)), -1)
    for _ in range(200):
        target, hand = c(action, qref, hand, zero_offset)
    d_lo, d_hi = lo[c.driver], hi[c.driver]
    assert torch.allclose(target[:, c.driver], (d_lo + 0.75 * (d_hi - d_lo)).expand(3, 12), atol=1e-4)
    expected = (target[:, c.driver][:, c.mimic_driver] * c.mimic_ratio).clamp(lo[c.mimic], hi[c.mimic])
    assert torch.allclose(target[:, c.mimic], expected, atol=1e-6)
    assert (target >= lo - 1e-6).all() and (target <= hi + 1e-6).all()


def test_encoder_offsets_shift_every_target(robot):
    c = Controller(robot)
    qref = c.default.expand(2, 53).clone()
    offset = torch.full((2, 53), 0.01)
    with_offset, _ = c(torch.zeros(2, 41), qref, c.seed(qref), offset)
    without, _ = c(torch.zeros(2, 41), qref, c.seed(qref), torch.zeros(2, 53))
    assert torch.allclose(with_offset - without, offset)


def test_full_body_command_is_clipped_with_its_reference_skip(robot):
    c = Controller(robot, body_action_clip=2.0)
    qref = c.default.expand(1, 53).clone()
    qref[:, c.body[0]] += 1.5 * c.scale[c.body[0]]
    action = torch.zeros(1, 41)
    action[:, 0] = 1.0  # skip 1.5 + residual 1.0 exceeds the clip of 2.0
    bounded = c.bound_action(action, qref)
    assert torch.allclose(c.body_command(bounded, qref)[:, 0], torch.tensor(2.0))
    assert torch.equal(bounded[:, 1:], action[:, 1:])


def test_reference_action_reproduces_reference_finger_targets(robot):
    c = Controller(robot)
    lo, hi = c.limits.unbind(-1)
    qref = lo + torch.rand(4, 53) * (hi - lo)
    hand = c.seed(qref)
    for _ in range(300):
        target, hand = c(c.reference_action(qref), qref, hand, torch.zeros(4, 53))
    assert torch.allclose(target[:, c.driver], qref[:, c.driver], atol=1e-4)
    assert torch.allclose(target[:, c.body], qref[:, c.body])
