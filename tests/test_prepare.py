import re

import numpy as np

from supermimic_hoi.prepare.dataset import select_clips
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
