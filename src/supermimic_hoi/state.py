"""Flat storage of a state tuple for replay. Rotations stay matrices: no quaternion round trip."""

import torch


def pack(state):
    p, R, v, w, q, qd, op, oR, ov, ow = state
    return torch.cat((p, R.flatten(-2), v, w, q, qd, op, oR.flatten(-2), ov, ow), -1)


def unpack(x):
    J = (x.shape[-1] - 36) // 2
    p, R, v, w, q, qd, op, oR, ov, ow = x.split((3, 9, 3, 3, J, J, 3, 9, 3, 3), -1)
    return p, R.unflatten(-1, (3, 3)), v, w, q, qd, op, oR.unflatten(-1, (3, 3)), ov, ow
