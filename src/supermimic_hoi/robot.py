"""Unitree G1 (29 DoF) with two Inspire hands: joint layout, actuation and tracked bodies.

The simulated state has 53 joints in MJCF order. The policy commands 41 of them: 29 body joints
as residuals on the reference and 12 finger drivers as absolute targets. The other 12 finger
joints are URDF mimic joints; each follows its driver's target through its own PD drive, as in
Weave's ``MimicJointPositionAction``.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import mujoco
import numpy as np

SIDES = ("L", "R")
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
FINGER_NAMES = tuple(f"{side}_{finger}" for side in SIDES for finger in FINGERS)

# Policy order of the finger drivers (per hand).
HAND_DRIVERS = ("thumb_proximal_yaw", "thumb_proximal_pitch", "index_proximal", "middle_proximal",
                "ring_proximal", "pinky_proximal")  # fmt: skip
# Mimic joint -> (driver, multiplier), from the URDF <mimic> tags.
HAND_MIMICS = {
    "thumb_intermediate": ("thumb_proximal_pitch", 1.6),
    "thumb_distal": ("thumb_proximal_pitch", 2.4),
    "index_intermediate": ("index_proximal", 1.0),
    "middle_intermediate": ("middle_proximal", 1.0),
    "ring_intermediate": ("ring_proximal", 1.0),
    "pinky_intermediate": ("pinky_proximal", 1.0),
}

# Weave's tracked bodies; the first one is the root (anchor).
TRACKED_BODIES = (
    "pelvis",
    "left_hip_pitch_link", "left_knee_link", "left_ankle_roll_link",
    "right_hip_pitch_link", "right_knee_link", "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
    "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
)  # fmt: skip
FINGERTIPS = tuple(f"{name}_tip" for name in FINGER_NAMES)
FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")
FOOT_CONTACT_Z = 0.045  # ankle-roll origin height below which a foot counts as on the ground

DEFAULT_JOINT_POS = {
    ".*_hip_pitch_joint": -0.312,
    ".*_knee_joint": 0.669,
    ".*_ankle_pitch_joint": -0.363,
    ".*_elbow_joint": 0.6,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_pitch_joint": 0.2,
}

# Body drives: implicit PD at 10 Hz natural frequency, damping ratio 2, on the reflected rotor
# inertia (BeyondMimic / Weave). Values are (effort limit [Nm], armature [kg m^2]).
NATURAL_FREQ = 10 * 2 * math.pi
DAMPING_RATIO = 2.0
ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425
BODY_MOTORS = {
    "hip_pitch": (88.0, ARMATURE_7520_14),
    "hip_roll": (139.0, ARMATURE_7520_22),
    "hip_yaw": (88.0, ARMATURE_7520_14),
    "knee": (139.0, ARMATURE_7520_22),
    "ankle_pitch": (50.0, 2 * ARMATURE_5020),
    "ankle_roll": (50.0, 2 * ARMATURE_5020),
    "waist_yaw": (88.0, ARMATURE_7520_14),
    "waist_roll": (50.0, 2 * ARMATURE_5020),
    "waist_pitch": (50.0, 2 * ARMATURE_5020),
    "shoulder_pitch": (25.0, ARMATURE_5020),
    "shoulder_roll": (25.0, ARMATURE_5020),
    "shoulder_yaw": (25.0, ARMATURE_5020),
    "elbow": (25.0, ARMATURE_5020),
    "wrist_roll": (25.0, ARMATURE_5020),
    "wrist_pitch": (5.0, ARMATURE_4010),
    "wrist_yaw": (5.0, ARMATURE_4010),
}
# Inspire finger drives (Weave): the same PD on drivers and mimic joints.
FINGER_STIFFNESS, FINGER_DAMPING, FINGER_EFFORT, FINGER_ARMATURE = 5.0, 0.5, 1.0, 0.01

# Domain randomization (the paper's G1 setup) and observation noise half-widths (Table A.1).
FRICTION_RANGE = (0.3, 1.2)
COM_BODY = "torso_link"
COM_RANGE = ((-0.025, 0.025), (-0.05, 0.05), (-0.05, 0.05))
JOINT_OFFSET = 0.01
NOISE_JOINT_POS, NOISE_JOINT_VEL, NOISE_ANG_VEL, NOISE_GRAVITY, NOISE_REF_ORI = 0.01, 0.5, 0.2, 0.05, 0.05


def is_finger_joint(name: str) -> bool:
    return name[:2] in ("L_", "R_")


def finger_channel(body: str) -> int:
    """Finger index 0..9 (left then right; thumb..pinky) of a hand link, or -1 (palm, tips, body)."""
    if body.endswith("_tip"):
        return -1
    return next((i for i, name in enumerate(FINGER_NAMES) if body.startswith(name + "_")), -1)


def hand_side(body: str) -> int:
    """0 (left) / 1 (right) for palm and finger links, -1 for everything else."""
    return SIDES.index(body[0]) if body[:2] in ("L_", "R_") and not body.endswith("_tip") else -1


def body_motor(joint: str) -> tuple[float, float]:
    return BODY_MOTORS[re.sub(r"^(left|right)_", "", joint).removesuffix("_joint")]


@dataclass(frozen=True)
class Robot:
    """The compiled robot description shared by kinematics, control, physics and the MDP."""

    asset_dir: Path

    @property
    def mjcf_path(self) -> Path:
        return Path(self.asset_dir) / "g1_inspire.xml"

    @cached_property
    def model(self) -> mujoco.MjModel:
        if not self.mjcf_path.is_file():
            raise FileNotFoundError(f"Missing {self.mjcf_path}; run `python -m supermimic_hoi.prepare`")
        return mujoco.MjModel.from_xml_path(str(self.mjcf_path))

    @cached_property
    def joint_names(self) -> tuple[str, ...]:
        m = self.model
        return tuple(m.joint(j).name for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE)

    @property
    def num_joints(self) -> int:
        return len(self.joint_names)

    @cached_property
    def body_joint_names(self) -> tuple[str, ...]:
        return tuple(n for n in self.joint_names if not is_finger_joint(n))

    @cached_property
    def driver_joint_names(self) -> tuple[str, ...]:
        return tuple(f"{side}_{name}_joint" for side in SIDES for name in HAND_DRIVERS)

    @cached_property
    def mimic_joint_names(self) -> tuple[str, ...]:
        return tuple(f"{side}_{name}_joint" for side in SIDES for name in HAND_MIMICS)

    def indices(self, names) -> list[int]:
        return [self.joint_names.index(n) for n in names]

    @cached_property
    def body_idx(self) -> list[int]:
        return self.indices(self.body_joint_names)

    @cached_property
    def driver_idx(self) -> list[int]:
        return self.indices(self.driver_joint_names)

    @cached_property
    def mimic_idx(self) -> list[int]:
        return self.indices(self.mimic_joint_names)

    @cached_property
    def mimic_driver(self) -> list[int]:
        """For every mimic joint, the position of its driver in the 12 driver commands."""
        return [self.driver_joint_names.index(f"{side}_{HAND_MIMICS[name][0]}_joint")
                for side in SIDES for name in HAND_MIMICS]  # fmt: skip

    @cached_property
    def mimic_ratio(self) -> list[float]:
        return [HAND_MIMICS[name][1] for _ in SIDES for name in HAND_MIMICS]

    @property
    def num_actions(self) -> int:
        return len(self.body_joint_names) + len(self.driver_joint_names)

    @cached_property
    def joint_limits(self) -> np.ndarray:
        """(J, 2) position limits in joint order."""
        m = self.model
        return np.array([m.jnt_range[m.joint(n).id] for n in self.joint_names], dtype=np.float32)

    @cached_property
    def default_joint_pos(self) -> np.ndarray:
        q = np.zeros(self.num_joints, dtype=np.float32)
        for i, name in enumerate(self.joint_names):
            for pattern, value in DEFAULT_JOINT_POS.items():
                if re.fullmatch(pattern, name):
                    q[i] = value
        return q

    @cached_property
    def control_center(self) -> np.ndarray:
        """Per-joint zero of the normalized PD target: default pose (body), mid-range (fingers)."""
        mid = self.joint_limits.mean(-1)
        return np.where([is_finger_joint(n) for n in self.joint_names], mid, self.default_joint_pos).astype(np.float32)

    @cached_property
    def control_scale(self) -> np.ndarray:
        """Radians per unit of normalized PD target: 0.25 effort / kp (body), half-range (fingers)."""
        half = (self.joint_limits[:, 1] - self.joint_limits[:, 0]) / 2
        scale = [half[i] if is_finger_joint(n) else 0.25 * body_motor(n)[0] / (body_motor(n)[1] * NATURAL_FREQ**2)
                 for i, n in enumerate(self.joint_names)]  # fmt: skip
        return np.asarray(scale, dtype=np.float32)

    @property
    def tracked_bodies(self) -> tuple[str, ...]:
        """Bodies whose FK the MDP needs: the tracking set followed by the ten fingertips."""
        return TRACKED_BODIES + FINGERTIPS

    def entity_cfg(self):
        from mjlab.actuator import BuiltinPositionActuatorCfg
        from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

        actuators = []
        for name in self.joint_names:
            if is_finger_joint(name):
                kp, kd, effort, armature = FINGER_STIFFNESS, FINGER_DAMPING, FINGER_EFFORT, FINGER_ARMATURE
            else:
                effort, armature = body_motor(name)
                kp, kd = armature * NATURAL_FREQ**2, 2 * DAMPING_RATIO * armature * NATURAL_FREQ
            actuators.append(BuiltinPositionActuatorCfg(
                target_names_expr=(name,), stiffness=kp, damping=kd, effort_limit=effort, armature=armature))  # fmt: skip
        return EntityCfg(
            init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, 0.76), joint_pos=DEFAULT_JOINT_POS, joint_vel={".*": 0.0}),
            spec_fn=lambda: mujoco.MjSpec.from_file(str(self.mjcf_path)),
            articulation=EntityArticulationInfoCfg(actuators=tuple(actuators), soft_joint_pos_limit_factor=0.9),
        )
