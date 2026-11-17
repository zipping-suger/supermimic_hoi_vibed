import re

import numpy as np
import pytest

from supermimic_hoi.prepare.dataset import ROLLOUT_SUFFIX, prepare_dataset, select_clips
from supermimic_hoi.prepare.objects import (MAX_HULL_VERTICES, WEAVE_HULLS, cooked_parts, primitive_parts, primitive_vertices,
                                            read_primitive_colliders)  # fmt: skip


def test_selection_covers_every_base_interaction_before_repeating_one():
    names = np.array([f"sub1_box_{base:03}_v{v:02}" for base in range(6) for v in range(5)])
    valid = np.ones(len(names), dtype=bool)
    valid[[0, 1, 2, 3, 4]] = False  # base 000 is corrupted in every variant
    chosen = select_clips(names, valid, 8, np.random.default_rng(0))
    bases = [re.sub(r"_v\d+$", "", names[i]) for i in chosen]
    assert len(set(chosen)) == 8 and "sub1_box_000" not in bases
    assert len(set(bases)) == 5 and max(bases.count(b) for b in set(bases)) == 2
    assert chosen == select_clips(names, valid, 8, np.random.default_rng(0))


USDA = """#usda 1.0
def Xform "stand"
{
    def Cylinder "base" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        uniform token axis = "Z"
        double height = 0.04
        double radius = 0.2
        matrix4d xformOp:transform = ( (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0.1, 0, -0.8, 1) )
        uniform token[] xformOpOrder = ["xformOp:transform"]
    }

    def Capsule "peg" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        uniform token axis = "Z"
        double height = 0.1
        double radius = 0.01
        matrix4d xformOp:transform = ( (1, 0, 0, 0), (0, 0, 1, 0), (0, -1, 0, 0), (0, 0.3, 0.5, 1) )
        uniform token[] xformOpOrder = ["xformOp:transform"]
    }
}
"""


def test_primitive_colliders_become_convex_parts_in_the_object_frame(tmp_path):
    import trimesh

    usda = tmp_path / "stand.usda"
    usda.write_text(USDA)
    base, peg = read_primitive_colliders(usda)
    assert (base["kind"], base["radius"], base["height"]) == ("cylinder", 0.2, 0.04) and peg["kind"] == "capsule"
    for kind in ("cylinder", "capsule"):
        assert len(primitive_vertices(kind, 0.02, 0.3)) <= MAX_HULL_VERTICES
    parts = [trimesh.load(path, force="mesh") for path in primitive_parts(usda, tmp_path / "parts")]
    # The cylinder keeps its volume and its place; the capsule's axis is rotated onto -y (USD row vectors).
    assert abs(parts[0].volume - np.pi * 0.2**2 * 0.04) < 1e-3 * parts[0].volume
    assert np.allclose(parts[0].centroid, (0.1, 0, -0.8), atol=1e-6) and np.allclose(parts[0].extents[2], 0.04)
    assert np.allclose(parts[1].centroid, (0, 0.3, 0.5), atol=1e-6) and np.allclose(parts[1].extents[1], 0.12, atol=1e-6)
    capsule = np.pi * 0.01**2 * 0.1 + 4 / 3 * np.pi * 0.01**3
    assert abs(parts[1].volume - capsule) < 0.05 * capsule
    assert primitive_parts(usda, tmp_path / "parts") == sorted((tmp_path / "parts").glob("part_*.obj"))  # cached


def test_cooked_parts_are_the_exported_weave_hulls(tmp_path):
    import pytest
    import trimesh

    with np.load(WEAVE_HULLS) as archive:
        exported = [archive[key] for key in sorted(archive.files) if key.startswith("tripod/")]
    parts = cooked_parts("tripod", tmp_path / "tripod")
    assert len(parts) == len(exported) > 1
    for path, vertices in zip(parts, exported):
        part = trimesh.load(path, force="mesh")
        assert len(part.vertices) <= MAX_HULL_VERTICES and part.is_convex
        assert np.allclose(part.bounds, (vertices.min(0), vertices.max(0)), atol=1e-6)
    assert cooked_parts("tripod", tmp_path / "tripod") == parts  # cached
    with pytest.raises(ValueError, match="No exported collision hulls"):
        cooked_parts("clothesstand", tmp_path / "clothesstand")  # primitive colliders, not in the archive


@pytest.mark.parametrize("assets", ["g1_inspire"], indirect=True)
def test_rollout_archives_are_sampled_under_their_reference_clip_names(robot, tmp_path):
    """A Weave rollout archive is a reference archive whose clip names carry ROLLOUT_SUFFIX."""
    from supermimic_hoi.kinematics import Kinematics
    from supermimic_hoi.prepare.dataset import ISAAC_BODIES
    import torch

    model = robot.model
    isaac_joints = [model.joint(model.body_jntadr[model.body(b).id]).name for b in ISAAC_BODIES[1:]]
    frames, lengths = 6, [2, 4]
    q = torch.from_numpy(robot.default_joint_pos).repeat(frames, 1)
    root = torch.tensor([0.0, 0.0, 0.8]).repeat(frames, 1)
    zero = torch.zeros(frames, 3)
    body_pos = Kinematics(robot, bodies=ISAAC_BODIES).forward(root, torch.eye(3).repeat(frames, 1, 1), zero, zero, q, torch.zeros_like(q))[0]
    identity = np.tile(np.array([1.0, 0, 0, 0], dtype=np.float32), (frames, 1))
    folder = tmp_path / "data" / "rollout" / "test"
    folder.mkdir(parents=True)
    np.savez(folder / "box_rollout.npz",
             joint_pos=q.numpy()[:, [robot.joint_names.index(n) for n in isaac_joints]], body_pos_w=body_pos.numpy(),
             body_quat_w=np.tile(identity[:, None], (1, len(ISAAC_BODIES), 1)), object_pos_w=np.zeros((frames, 3), np.float32),
             object_quat_w=identity, contact_label=np.zeros((frames, len(ISAAC_BODIES)), np.float32),
             motion_lengths=np.array(lengths), fps=np.array([50]), object_names=np.array(["box", "box"]),
             motion_names=np.array([f"sub1_box_000_v00{ROLLOUT_SUFFIX}", f"sub1_box_001_v03{ROLLOUT_SUFFIX}"]))  # fmt: skip
    manifest = prepare_dataset(tmp_path, tmp_path / "out", robot, "rollout/test", 2, 0)
    assert manifest["clip_names"] == ["sub1_box_000_v00", "sub1_box_001_v03"] and manifest["split"] == "rollout/test"
    assert manifest["frames"] == frames and manifest["fk_max_error_m"] < 1e-5
