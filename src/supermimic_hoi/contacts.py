"""Measured hand-object contact forces. Simulator only: imagination never queries contact geometry.

Twelve force totals per world: [left hand, right hand, ten fingers]. A hand total covers its
palm and all finger links; a finger total covers every link of that finger. Only contacts between
a hand geom and an object geom count, and only their normal force.
"""

from __future__ import annotations

import mujoco
import numpy as np
import warp as wp
from mujoco_warp._src.support import contact_force_fn
from mujoco_warp._src.types import vec5

NUM_GROUPS = 12


class ContactGroups:
    """Per-geom lookup tables for a compiled scene with ``robot/`` and ``object/`` entities."""

    def __init__(self, model: mujoco.MjModel, robot):
        self.object_geom = np.zeros(model.ngeom, dtype=np.int32)
        self.hand = np.full(model.ngeom, -1, dtype=np.int32)
        self.finger = np.full(model.ngeom, -1, dtype=np.int32)
        for geom in range(model.ngeom):
            body = model.body(model.geom_bodyid[geom]).name
            self.object_geom[geom] = body.startswith("object/")
            if body.startswith("robot/"):
                self.hand[geom], self.finger[geom] = robot.hand_link(body.removeprefix("robot/"))

    def cpu_forces(self, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
        forces, wrench = np.zeros(NUM_GROUPS), np.zeros(6)
        for i in range(data.ncon):
            g0, g1 = data.contact[i].geom
            if min(g0, g1) < 0 or self.object_geom[g0] == self.object_geom[g1]:
                continue
            other = g0 if self.object_geom[g1] else g1
            if self.hand[other] < 0:
                continue
            mujoco.mj_contactForce(model, data, i, wrench)
            forces[self.hand[other]] += max(0.0, wrench[0])
            if self.finger[other] >= 0:
                forces[2 + self.finger[other]] += max(0.0, wrench[0])
        return forces


@wp.kernel(enable_backward=False)
def reduce_contacts(cone: int, njmax: int, nacon: wp.array(dtype=wp.int32),
                    geom: wp.array(dtype=wp.vec2i), worldid: wp.array(dtype=wp.int32),
                    frame: wp.array(dtype=wp.mat33), friction: wp.array(dtype=vec5),
                    dim: wp.array(dtype=wp.int32), efc_address: wp.array2d(dtype=wp.int32),
                    efc_force: wp.array2d(dtype=wp.float32), object_geom: wp.array(dtype=wp.int32),
                    hand: wp.array(dtype=wp.int32), finger: wp.array(dtype=wp.int32),
                    result: wp.array2d(dtype=wp.float32)):  # fmt: skip
    i = wp.tid()
    if i >= nacon[0]:
        return
    g = geom[i]
    if g[0] < 0 or g[1] < 0 or object_geom[g[0]] == object_geom[g[1]]:
        return
    other = g[0]
    if object_geom[g[0]] != 0:
        other = g[1]
    side = hand[other]
    if side < 0:
        return
    world = worldid[i]
    force = contact_force_fn(cone, frame, friction, dim, efc_address, efc_force, njmax, nacon, world, i, False)
    normal = wp.max(force[0], 0.0)
    wp.atomic_add(result, world, side, normal)
    if finger[other] >= 0:
        wp.atomic_add(result, world, 2 + finger[other], normal)


def kernel_inputs(model, data):
    """The MuJoCo-Warp arrays ``reduce_contacts`` reads (its dataclasses are host objects)."""
    c = data.contact
    return [int(model.opt.cone), data.njmax, data.nacon, c.geom, c.worldid, c.frame, c.friction, c.dim,
            c.efc_address, data.efc.force]  # fmt: skip
