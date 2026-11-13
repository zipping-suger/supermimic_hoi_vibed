"""Exclude empty object-variant mesh slots before MuJoCo-Warp's narrowphase.

Objects have different numbers of convex parts. mjlab pads the shorter ones and marks the
missing meshes with ``geom_dataid = -1``. MuJoCo-Warp 3.11's broadphase still admits the padded
geoms' template bounds, and its convex support function then treats a missing mesh as a point.
This filter redirects such candidate pairs to a plane-plane pair, which no narrowphase routine
handles, so they produce no contact. Real geometry is untouched.
"""

import mujoco
import numpy as np
import warp as wp

MESH = wp.constant(int(mujoco.mjtGeom.mjGEOM_MESH))


@wp.kernel(enable_backward=False)
def mask_empty_mesh_pairs(geom_type: wp.array(dtype=wp.int32), geom_dataid: wp.array2d(dtype=wp.int32),
                          count: wp.array(dtype=wp.int32), world: wp.array(dtype=wp.int32),
                          pairs: wp.array(dtype=wp.vec2i), plane: int):  # fmt: skip
    i = wp.tid()
    if i >= count[0]:
        return
    pair = pairs[i]
    row = world[i] % geom_dataid.shape[0]
    absent = geom_type[pair[0]] == MESH and geom_dataid[row, pair[0]] < 0
    absent = absent or (geom_type[pair[1]] == MESH and geom_dataid[row, pair[1]] < 0)
    if absent:
        pairs[i] = wp.vec2i(plane, plane)


def filter_empty_variant_meshes(model, host_model) -> int:
    """Install the filter for ``model`` and return the number of empty slots; recapture graphs afterwards."""
    from mujoco_warp._src import collision_driver

    missing = (model.geom_dataid.numpy() < 0) & (np.asarray(host_model.geom_type)[None, :] == int(MESH))
    if not missing.any():
        return 0
    planes = np.flatnonzero(host_model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)
    if not len(planes) or (0, 0) in collision_driver.MJ_COLLISION_TABLE:
        raise ValueError("The empty-slot filter needs a plane geom and a runtime without plane-plane collision")
    model._empty_mesh_plane = int(planes[0])
    if not getattr(collision_driver._narrowphase, "_filters_empty_meshes", False):
        original = collision_driver._narrowphase

        def narrowphase(m, d, ctx):
            plane = getattr(m, "_empty_mesh_plane", None)
            if plane is not None:
                wp.launch(mask_empty_mesh_pairs, dim=d.naconmax, device=d.ncollision.device,
                          inputs=[m.geom_type, m.geom_dataid, d.ncollision, ctx.collision_worldid,
                                  ctx.collision_pair, plane])  # fmt: skip
            original(m, d, ctx)

        narrowphase._filters_empty_meshes = True
        collision_driver._narrowphase = narrowphase
    return int(missing.sum())
