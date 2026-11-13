"""Render a reference clip, or an evaluation rollout beside its reference, to an MP4.

    python -m supermimic_hoi.render --clip sub17_smallbox_034_v00 --output results/reference.mp4
    python -m supermimic_hoi.render --rollout results/evaluation_<clip>.npz --output results/rollout.mp4

Rollout files come from ``python -m supermimic_hoi.evaluate ... --record <clip>``. Needs the
``render`` extra (imageio-ffmpeg) and an offscreen OpenGL backend (EGL by default).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from .motion import CONTROL_DT, Motion  # noqa: E402
from .robot import Robot  # noqa: E402
from .sim import Indexing, encode, load_registry, make_scene  # noqa: E402
from .state import unpack  # noqa: E402

WIDTH, HEIGHT = 640, 480


def to_qpos(states, model, idx) -> np.ndarray:
    """State tuples (T, ...) -> MuJoCo qpos rows."""
    qpos = torch.from_numpy(np.tile(model.qpos0, (len(states[0]), 1)))
    encode(tuple(x.double() for x in states), qpos, torch.zeros(len(qpos), model.nv, dtype=torch.float64), idx)
    return qpos.numpy()


def render(clip: str, output: Path, rollout: Path | None, dataset: str, robot_assets: str, object_assets: str, fps: int = 25):
    import imageio_ffmpeg

    robot = Robot(Path(robot_assets))
    registry, _ = load_registry(object_assets)
    motion = Motion(dataset, robot, [o["name"] for o in registry["objects"]], clips=[clip])
    model = make_scene(robot, registry, object_assets, [int(motion.clip_object[0])], 1, "cpu").compile()
    idx = Indexing(model, robot)
    first = 0
    panels = []
    if rollout is not None:
        with np.load(rollout) as data:
            first = int(data["start_frame"])
            panels.append(to_qpos(unpack(torch.from_numpy(data["state"])), model, idx))
    frames = torch.arange(first, first + len(panels[0]) if panels else motion.num_frames)
    panels.append(to_qpos(motion.ref_state(frames), model, idx))

    model.vis.global_.offwidth, model.vis.global_.offheight = WIDTH, HEIGHT
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.azimuth, camera.elevation, camera.distance = 135, -15, 3.2
    pelvis = model.body("robot/pelvis").id
    step = max(1, round(1 / (fps * CONTROL_DT)))
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(str(output), (WIDTH * len(panels), HEIGHT), fps=fps, macro_block_size=1)
    writer.send(None)
    with mujoco.Renderer(model, height=HEIGHT, width=WIDTH) as renderer:
        for t in range(0, len(panels[-1]), step):
            images = []
            for qpos in reversed(panels):  # the reference comes first and sets the shared camera target
                data.qpos[:] = qpos[t]
                mujoco.mj_forward(model, data)
                if not images:
                    camera.lookat[:] = (data.xpos[pelvis][0], data.xpos[pelvis][1], 0.7)
                renderer.update_scene(data, camera=camera)
                images.append(renderer.render().copy())
            writer.send(np.ascontiguousarray(np.concatenate(images[::-1], axis=1)))
    writer.close()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clip", help="Clip name (taken from the rollout file when one is given)")
    p.add_argument("--rollout", type=Path, help="Recorded evaluation trajectory; shown left of the reference")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dataset", default="data/weave600")
    p.add_argument("--robot-assets", default="data/robot")
    p.add_argument("--object-assets", default="data/objects")
    a = p.parse_args()
    clip = a.clip
    if a.rollout is not None:
        with np.load(a.rollout) as data:
            clip = str(data["clip"])
    if not clip:
        p.error("--clip or --rollout is required")
    render(clip, a.output, a.rollout, a.dataset, a.robot_assets, a.object_assets)


if __name__ == "__main__":
    main()
