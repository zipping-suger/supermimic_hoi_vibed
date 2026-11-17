"""Unitree G1 (29 DoF) with two Inspire hands.

The simulated state has 53 joints in MJCF order. The policy commands 41 of them: 29 body joints
and 12 finger drivers. The other 12 finger joints are URDF mimic joints; each follows its driver's
target through its own PD drive, as in Weave's ``MimicJointPositionAction``.
"""

from __future__ import annotations

from functools import cached_property

from ..robot import (ARMATURE_4010, ARMATURE_5020, ARMATURE_7520_14, ARMATURE_7520_22, FINGERS, Drive, Robot,
                     motor_drive, side_joint)  # fmt: skip

SIDES = ("L", "R")
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

# Body motors: (effort limit [Nm], reflected inertia [kg m^2]).
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
FINGER_DRIVE = Drive(stiffness=5.0, damping=0.5, effort=1.0, armature=0.01)


class G1Inspire(Robot):
    name = "g1_inspire"
    mjcf = "g1_inspire.xml"
    # Weave's tracked bodies.
    tracked_bodies = (
        "pelvis",
        "left_hip_pitch_link", "left_knee_link", "left_ankle_roll_link",
        "right_hip_pitch_link", "right_knee_link", "right_ankle_roll_link",
        "torso_link",
        "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
        "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
    )  # fmt: skip
    fingertips = tuple(f"{name}_tip" for name in FINGER_NAMES)
    surface_bodies = ("left_wrist_yaw_link", "right_wrist_yaw_link")
    foot_bodies = ("left_ankle_roll_link", "right_ankle_roll_link")
    foot_contact_z = 0.045
    com_body = "torso_link"
    root_height = 0.76
    default_pose = {
        ".*_hip_pitch_joint": -0.312,
        ".*_knee_joint": 0.669,
        ".*_ankle_pitch_joint": -0.363,
        ".*_elbow_joint": 0.6,
        "left_shoulder_roll_joint": 0.2,
        "left_shoulder_pitch_joint": 0.2,
        "right_shoulder_roll_joint": -0.2,
        "right_shoulder_pitch_joint": 0.2,
    }

    @cached_property
    def driver_joint_names(self) -> tuple[str, ...]:
        return tuple(f"{side}_{name}_joint" for side in SIDES for name in HAND_DRIVERS)

    @cached_property
    def mimics(self) -> dict[str, tuple[str, float]]:
        return {f"{side}_{name}_joint": (f"{side}_{driver}_joint", ratio)
                for side in SIDES for name, (driver, ratio) in HAND_MIMICS.items()}  # fmt: skip

    def drive(self, joint: str) -> Drive:
        return FINGER_DRIVE if joint in self.hand_joint_names else motor_drive(*BODY_MOTORS[side_joint(joint)])

    def hand_link(self, body: str) -> tuple[int, int]:
        if body[:2] not in ("L_", "R_") or body.endswith("_tip"):
            return -1, -1
        return SIDES.index(body[0]), next((i for i, name in enumerate(FINGER_NAMES) if body.startswith(name + "_")), -1)
