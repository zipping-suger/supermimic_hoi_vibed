"""Convex collision assets and explicit inertials for Weave's objects."""

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
FRICTION = (0.8, 0.005, 0.0001)
MAX_PARTS = 32
MAX_HULL_VERTICES = 32
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


def decompose(mesh_path: Path, part_dir: Path) -> list[Path]:
    """CoACD convex parts of one object mesh, cached on the mesh checksum."""
    import coacd
    import trimesh

    digest = hashlib.sha256(mesh_path.read_bytes()).hexdigest()
    settings = dict(mesh_sha256=digest, coacd="1.0.14", threshold=0.05, max_convex_hull=MAX_PARTS,
                    max_hull_vertices=MAX_HULL_VERTICES, seed=0)  # fmt: skip
    marker = part_dir / "complete.json"
    if marker.is_file() and json.loads(marker.read_text()) == settings:
        return sorted(part_dir.glob("part_*.obj"))
    mesh = trimesh.load(mesh_path, force="mesh", process=True)
    coacd.set_log_level("warn")
    parts = coacd.run_coacd(coacd.Mesh(np.asarray(mesh.vertices), np.asarray(mesh.faces)), threshold=0.05,
                            max_convex_hull=MAX_PARTS, preprocess_mode="auto", resolution=2000,
                            mcts_nodes=10, mcts_iterations=50, mcts_max_depth=3, seed=0)  # fmt: skip
    if not parts:
        raise ValueError(f"Empty collision decomposition: {mesh_path}")
    shutil.rmtree(part_dir, ignore_errors=True)
    part_dir.mkdir(parents=True)
    for i, (vertices, _) in enumerate(parts):
        capped_hull(vertices).export(part_dir / f"part_{i:03}.obj")
    marker.write_text(json.dumps(settings, indent=2) + "\n")
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
        parts = decompose(source, output / "parts" / name)
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
