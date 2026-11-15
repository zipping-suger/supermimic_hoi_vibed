"""Convex collision assets and explicit inertials for Weave's objects.

Collision shapes are the ones Weave simulates. Where Weave ships hand-fitted primitive colliders
(``<name>.usda``, used for the clothes stand) those are converted to convex meshes. Every other
object is a convex decomposition that PhysX computes when it loads Weave's USD; those hulls are
exported once by ``export_weave_hulls.py`` into ``weave_hulls.npz``. Mass is Weave's density on
the collision volume.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

OBJECT_DIR = "source/g1_hoi_learning/g1_hoi_learning/objects"
# Weave spawns every object with density 200 kg/m^3 on its collision volume; the large box uses 25.
DENSITY = {"largebox": 25.0}
DEFAULT_DENSITY = 200.0
FRICTION = (1.0, 0.005, 0.0001)
MAX_HULL_VERTICES = 32  # Weave's hull vertex limit; MuJoCo-Warp needs small hulls too
WEAVE_HULLS = Path(__file__).with_name("weave_hulls.npz")
CONTEXT_FIELDS = ("mass", "com_x", "com_y", "com_z", "Ixx", "Iyy", "Izz", "Ixy", "Ixz", "Iyz",
                  "sliding_friction", "torsional_friction", "rolling_friction")  # fmt: skip


def capped_hull(vertices: np.ndarray):
    """Convex hull with at most MAX_HULL_VERTICES vertices, as MuJoCo's ``maxhullvert`` builds it."""
    import mujoco
    import trimesh

    spec = mujoco.MjSpec()
    mesh = spec.add_mesh(name="part", uservert=np.asarray(vertices, dtype=np.float64).flatten())
    mesh.maxhullvert = MAX_HULL_VERTICES
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname="part")
    model = spec.compile()
    start, count = int(model.mesh_graphadr[0]), int(model.mesh_graph[model.mesh_graphadr[0]])
    ids = model.mesh_graph[start + 2 + count : start + 2 + 2 * count]
    # MuJoCo stores vertices in the mesh's inertial frame; map the hull back to the source frame.
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, model.mesh_quat[0])
    points = model.mesh_vert[ids] @ rotation.reshape(3, 3).T + model.mesh_pos[0]
    return trimesh.Trimesh(points).convex_hull


# Vertices around a primitive's axis. A capsule uses four rings and two poles (30 vertices), a
# cylinder two rings (32), so both stay within MAX_HULL_VERTICES.
CAPSULE_SIDES, CYLINDER_SIDES = 7, 16


def read_primitive_colliders(usda: Path) -> list[dict]:
    """Capsule and cylinder colliders of a Weave ``.usda``: name, kind, radius, height and the
    4x4 transform (USD row-vector convention) into the object frame."""
    import re

    text = usda.read_text()
    number = r"[-+0-9.eE]+"
    colliders = []
    for kind, name, body in re.findall(r'def (\w+) "(\w+)" \(\s*prepend apiSchemas = \["PhysicsCollisionAPI"\]\s*\)\s*\{(.*?)\n    \}', text, re.S):
        if kind not in ("Capsule", "Cylinder") or 'token axis = "Z"' not in body or 'xformOpOrder = ["xformOp:transform"]' not in body:
            raise ValueError(f"Unsupported collider {name} ({kind}) in {usda}")
        radius, height = (float(re.search(rf"double {key} = ({number})", body).group(1)) for key in ("radius", "height"))
        matrix = re.search(r"matrix4d xformOp:transform = \((.*)\)\s*\n", body).group(1)
        transform = np.array([float(x) for x in re.findall(number, matrix)]).reshape(4, 4)
        colliders.append(dict(name=name, kind=kind.lower(), radius=radius, height=height, transform=transform))
    if not colliders or text.count("PhysicsCollisionAPI") != len(colliders):
        raise ValueError(f"Could not read every collider of {usda}")
    return colliders


def primitive_vertices(kind: str, radius: float, height: float) -> np.ndarray:
    """Vertices of a convex polyhedron standing in for a z-axis capsule or cylinder.

    The polygonal cross-section has the area of the circle, so the cylindrical part keeps its
    volume (and with it Weave's mass) while its faces stay within 4 % of the true radius.
    """
    sides = CAPSULE_SIDES if kind == "capsule" else CYLINDER_SIDES
    angle = 2 * np.pi * np.arange(sides) / sides
    ring = np.sqrt(2 * np.pi / (sides * np.sin(2 * np.pi / sides))) * radius * np.stack((np.cos(angle), np.sin(angle)), -1)
    level = lambda scale, z: np.concatenate((scale * ring, np.full((sides, 1), z)), -1)  # noqa: E731
    half = height / 2
    if kind == "cylinder":
        return np.concatenate((level(1.0, -half), level(1.0, half)))
    cap, lift = np.cos(np.pi / 4), radius * np.sin(np.pi / 4)
    return np.concatenate((level(1.0, -half), level(1.0, half), level(cap, -half - lift), level(cap, half + lift),
                           [[0.0, 0.0, -half - radius], [0.0, 0.0, half + radius]]))  # fmt: skip


def primitive_parts(usda: Path, part_dir: Path) -> list[Path]:
    """Convex parts built from Weave's primitive colliders, cached on the file checksum."""
    settings = dict(usda_sha256=hashlib.sha256(usda.read_bytes()).hexdigest(), capsule_sides=CAPSULE_SIDES,
                    cylinder_sides=CYLINDER_SIDES, max_hull_vertices=MAX_HULL_VERTICES)  # fmt: skip
    marker = part_dir / "complete.json"
    if marker.is_file() and json.loads(marker.read_text())["settings"] == settings:
        return sorted(part_dir.glob("part_*.obj"))
    shutil.rmtree(part_dir, ignore_errors=True)
    part_dir.mkdir(parents=True)
    for i, collider in enumerate(read_primitive_colliders(usda)):
        local = primitive_vertices(collider["kind"], collider["radius"], collider["height"])
        transform = collider["transform"]
        capped_hull(local @ transform[:3, :3] + transform[3, :3]).export(part_dir / f"part_{i:03}.obj")
    marker.write_text(json.dumps(dict(settings=settings), indent=2) + "\n")
    return sorted(part_dir.glob("part_*.obj"))


def cooked_parts(name: str, part_dir: Path) -> list[Path]:
    """The convex hulls PhysX cooks for one of Weave's mesh colliders, cached on the archive checksum."""
    import trimesh

    settings = dict(weave_hulls_sha256=hashlib.sha256(WEAVE_HULLS.read_bytes()).hexdigest())
    marker = part_dir / "complete.json"
    if marker.is_file() and json.loads(marker.read_text())["settings"] == settings:
        return sorted(part_dir.glob("part_*.obj"))
    with np.load(WEAVE_HULLS) as archive:
        hulls = [archive[key].astype(np.float64) for key in sorted(archive.files) if key.split("/")[0] == name]
    if not hulls:
        raise ValueError(f"No exported collision hulls for {name}; run prepare/export_weave_hulls.py")
    if max(map(len, hulls)) > MAX_HULL_VERTICES:
        raise ValueError(f"A hull of {name} exceeds {MAX_HULL_VERTICES} vertices")
    shutil.rmtree(part_dir, ignore_errors=True)
    part_dir.mkdir(parents=True)
    for i, vertices in enumerate(hulls):
        trimesh.Trimesh(vertices).convex_hull.export(part_dir / f"part_{i:03}.obj")
    marker.write_text(json.dumps(dict(settings=settings), indent=2) + "\n")
    return sorted(part_dir.glob("part_*.obj"))


def prepare_objects(weave: Path, output: Path, names: list[str]) -> dict:
    """Write one MJCF per object and ``registry.json``; sorted names define the object keys."""
    import mujoco
    import trimesh

    output.mkdir(parents=True, exist_ok=True)
    entries, files = [], {}
    values = lambda x: " ".join(f"{v:.12g}" for v in np.asarray(x).flatten())  # noqa: E731
    for name in sorted(set(names)):
        source = Path(weave) / OBJECT_DIR / name / f"{name}.obj"
        primitives = source.with_suffix(".usda")
        part_dir = output / "parts" / name
        parts = primitive_parts(primitives, part_dir) if primitives.is_file() else cooked_parts(name, part_dir)
        visual = output / "visual" / f"{name}.obj"
        visual.parent.mkdir(exist_ok=True)
        shutil.copyfile(source, visual)
        meshes = [trimesh.load(p, force="mesh") for p in parts]
        volumes = np.array([abs(m.volume) for m in meshes])
        mass = float(DENSITY.get(name, DEFAULT_DENSITY) * volumes.sum())
        # Uniform density over the convex parts (overlaps are counted twice, as PhysX does).
        weights = volumes / volumes.sum() * mass
        com = sum(w * m.center_mass for w, m in zip(weights, meshes)) / mass
        inertia = np.zeros((3, 3))
        for weight, mesh, volume in zip(weights, meshes, volumes):
            d = mesh.center_mass - com
            inertia += mesh.moment_inertia * weight / volume + weight * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
        full = inertia[[0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]]

        xml = ET.Element("mujoco", model=name)
        ET.SubElement(xml, "compiler", angle="radian", inertiafromgeom="false")
        asset = ET.SubElement(xml, "asset")
        body = ET.SubElement(ET.SubElement(xml, "worldbody"), "body", name="object")
        ET.SubElement(body, "freejoint", name="root")
        ET.SubElement(body, "inertial", pos=values(com), mass=f"{mass:.12g}", fullinertia=values(full))
        ET.SubElement(asset, "mesh", name="visual", file=str(visual.relative_to(output)))
        ET.SubElement(body, "geom", name="visual", type="mesh", mesh="visual", contype="0", conaffinity="0",
                      group="1", density="0", rgba="0.85 0.45 0.2 1")  # fmt: skip
        for i, part in enumerate(parts):
            ET.SubElement(asset, "mesh", name=f"part_{i}", file=str(part.relative_to(output)))
            ET.SubElement(body, "geom", name=f"collision_{i}", type="mesh", mesh=f"part_{i}", contype="1",
                          conaffinity="1", group="3", friction=values(FRICTION), density="0")  # fmt: skip
        path = output / f"{name}.xml"
        ET.indent(xml)
        ET.ElementTree(xml).write(path, encoding="unicode")
        compiled = mujoco.MjModel.from_xml_path(str(path))
        if abs(float(compiled.body_mass.sum()) - mass) > 1e-6 * mass:
            raise ValueError(f"Compiled mass differs for {name}")
        for file in (path, visual, *parts):
            files[str(file.relative_to(output))] = hashlib.sha256(file.read_bytes()).hexdigest()
        entries.append(dict(name=name, xml=path.name, parts=len(parts), mass=mass,
                            context=[mass, *com.tolist(), *full.tolist(), *FRICTION]))  # fmt: skip
        print(f"Prepared {name}: {len(parts)} convex parts, {mass:.3f} kg", flush=True)
    registry = dict(format_version=1, objects=entries, context_fields=list(CONTEXT_FIELDS), files=files)
    (output / "registry.json").write_text(json.dumps(registry, indent=2) + "\n")
    return registry
