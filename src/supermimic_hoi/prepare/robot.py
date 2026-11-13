"""Compile Weave's G1 + Inspire URDF into the MJCF used for simulation and torch FK."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import mujoco

URDF = "source/g1_hoi_learning/g1_hoi_learning/assets/unitree_g1/g1_29dof_rev_1_0_with_inspire_hand_DFQ.urdf"
MAX_HULL_VERTICES = 32  # collision uses each link mesh's convex hull, capped at this many vertices
VISUAL_GROUP, COLLISION_GROUP = 1, 3


def prepare_robot(weave: Path, output: Path) -> Path:
    """Write ``output/g1_inspire.xml`` and its meshes. Kinematics and inertias are the URDF's.

    Every URDF link stays its own body (tracked link and fingertip frames exist) and the pelvis
    gets a free joint. Collision geoms touch the ground and the object but not each other, as in
    Weave (self-collisions disabled). The URDF mimic tags are dropped: mimic joints keep their own
    joint and PD drive, and the controller writes their targets.
    """
    source = Path(weave) / URDF
    text = re.sub(r"<!--.*?-->", "", source.read_text(), flags=re.S)
    text = re.sub(r"<mujoco>.*?</mujoco>", "", text, flags=re.S)
    compiler = (f'<mujoco><compiler meshdir="{source.parent}" balanceinertia="true" discardvisual="false" '
                'fusestatic="false" strippath="false"/></mujoco>')  # fmt: skip
    text = re.sub(r"(<robot[^>]*>)", lambda m: m.group(1) + compiler, text, count=1)

    # MuJoCo's URDF importer resolves everything; re-open the result as an editable MJCF spec.
    output.mkdir(parents=True, exist_ok=True)
    raw = output / "_urdf_import.xml"
    mujoco.mj_saveLastXML(str(raw), mujoco.MjModel.from_xml_string(text))
    spec = mujoco.MjSpec.from_file(str(raw))
    raw.unlink()

    spec.modelname = "g1_inspire"
    spec.body("pelvis").add_freejoint(name="floating_base_joint")
    collision_meshes = set()
    for body in spec.bodies:
        counts = {"visual": 0, "collision": 0}
        for geom in body.geoms:
            role = "visual" if geom.contype == 0 and geom.conaffinity == 0 else "collision"
            geom.name = f"{body.name}_{role}_{counts[role]}"
            counts[role] += 1
            if role == "visual":
                geom.group = VISUAL_GROUP
                continue
            geom.group, geom.contype, geom.conaffinity = COLLISION_GROUP, 0, 1
            if geom.type == mujoco.mjtGeom.mjGEOM_CYLINDER:
                # Weave imports the URDF with replace_cylinders_with_capsules; keep the overall length.
                radius, half = geom.size[0], geom.size[1]
                geom.type = mujoco.mjtGeom.mjGEOM_CAPSULE
                geom.size = [radius, max(half - radius, 1e-3), 0.0]
            elif geom.type == mujoco.mjtGeom.mjGEOM_MESH:
                collision_meshes.add(geom.meshname)

    mesh_dir = output / "meshes"
    mesh_dir.mkdir(exist_ok=True)
    for mesh in spec.meshes:
        file = source.parent / mesh.file
        shutil.copyfile(file, mesh_dir / file.name)
        mesh.file = file.name
    spec.meshdir = "meshes"

    spec.compile()
    xml = spec.to_xml()
    # MuJoCo 3.11 does not serialize maxhullvert, so set it in the text.
    for name in collision_meshes:
        xml, count = re.subn(f'<mesh name="{name}" ', f'<mesh name="{name}" maxhullvert="{MAX_HULL_VERTICES}" ', xml)
        if count != 1:
            raise ValueError(f"Collision mesh not found in MJCF: {name}")
    path = output / "g1_inspire.xml"
    path.write_text(xml)
    return path
