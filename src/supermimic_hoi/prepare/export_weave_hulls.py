"""Export the convex hulls PhysX cooks for Weave's mesh-collider objects into ``weave_hulls.npz``.

Weave's object USDs request a convex decomposition that PhysX computes when the stage loads, so
the shapes Weave simulates exist only inside Isaac Sim. This script asks PhysX for them. It needs
the Isaac Sim that Weave pins (5.1) and is run with that installation's Python, once, whenever
Weave's object assets change; ``prepare`` reads the committed result.

    python src/supermimic_hoi/prepare/export_weave_hulls.py /path/to/Weave
"""

import sys
from pathlib import Path

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from omni.physx import get_physx_cooking_interface  # noqa: E402
from omni.physx.bindings._physx import PhysxCollisionRepresentationResult  # noqa: E402
from pxr import PhysicsSchemaTools, Usd, UsdGeom, UsdPhysics, UsdUtils  # noqa: E402

OBJECT_DIR = "source/g1_hoi_learning/g1_hoi_learning/objects"


def cooked_hulls(usd: Path) -> list[np.ndarray]:
    """Vertices of every convex hull of the stage's mesh colliders, in the object frame."""
    context = omni.usd.get_context()
    context.open_stage(str(usd))
    stage = context.get_stage()
    if UsdGeom.GetStageMetersPerUnit(stage) != 1.0:
        raise ValueError(f"{usd} is not in metres")
    stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
    hulls = []
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        if not (prim.HasAPI(UsdPhysics.CollisionAPI) and prim.IsA(UsdGeom.Mesh)):
            continue
        to_object = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))  # row vectors
        received = []
        get_physx_cooking_interface().request_convex_collision_representation(
            stage_id=stage_id, collision_prim_id=PhysicsSchemaTools.sdfPathToInt(str(prim.GetPath())),
            run_asynchronously=False, on_result=lambda result, convexes: received.append((result, convexes)))
        result, convexes = received[0]
        if result != PhysxCollisionRepresentationResult.RESULT_VALID:
            raise ValueError(f"PhysX returned {result} for {prim.GetPath()}")
        for convex in convexes:
            local = np.array([[v.x, v.y, v.z, 1.0] for v in convex.vertices])
            hulls.append((local @ to_object)[:, :3].astype(np.float32))
    # PhysX reports the hulls in no fixed order.
    return sorted(hulls, key=lambda h: tuple(h.mean(0)))


def main():
    weave = Path(sys.argv[1])
    archive = {}
    for usd in sorted((weave / OBJECT_DIR).glob("*/*.usd")):
        if usd.with_suffix(".usda").is_file():
            continue  # primitive colliders, read from the .usda by prepare
        hulls = cooked_hulls(usd)
        if not hulls:
            raise ValueError(f"{usd} has no mesh collider")
        archive.update({f"{usd.stem}/{i:02}": hull for i, hull in enumerate(hulls)})
        print(f"{usd.stem}: {len(hulls)} hulls, at most {max(map(len, hulls))} vertices", flush=True)
    np.savez_compressed(Path(__file__).with_name("weave_hulls.npz"), **archive)


if __name__ == "__main__":
    main()
    app.close()
