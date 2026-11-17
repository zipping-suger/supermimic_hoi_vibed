"""A floating-base humanoid with two five-fingered hands: joint layout, drives and task bodies.

Each robot in ``robots/`` is a subclass that names its bodies and supplies three rules: which
finger joints the policy drives (and which follow a driver), the PD drive of every joint, and the
hand and finger a link belongs to. Everything else here is derived from the compiled MJCF, so
kinematics, control, physics and the MDP never mention a particular robot.

The policy commands the body joints as residuals on the reference and the finger drivers as
absolute targets. Fingers are counted left hand first, thumb to pinky.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import mujoco
import numpy as np

FINGERS = ("thumb", "index", "middle", "ring", "pinky")

# Body drives: implicit PD at 10 Hz natural frequency, damping ratio 2, on the reflected rotor
# inertia (BeyondMimic / Weave). Reflected inertias [kg m^2] of the Unitree actuators.
NATURAL_FREQ = 10 * 2 * math.pi
DAMPING_RATIO = 2.0
ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425

# Domain randomization and observation noise half-widths (BlinkMimic, Table A.1).
FRICTION_RANGE = (0.3, 1.2)
COM_RANGE = ((-0.025, 0.025), (-0.05, 0.05), (-0.05, 0.05))
JOINT_OFFSET = 0.01
NOISE_JOINT_POS, NOISE_JOINT_VEL, NOISE_ANG_VEL, NOISE_GRAVITY, NOISE_REF_ORI = 0.01, 0.5, 0.2, 0.05, 0.05


@dataclass(frozen=True)
class Drive:
    """PD position drive of one joint."""

    stiffness: float
    damping: float
    effort: float
    armature: float
    frictionloss: float | None = None
    viscous_damping: float | None = None


def motor_drive(effort: float, armature: float, **passive) -> Drive:
    """Body-joint drive from the motor's effort limit [Nm] and reflected inertia."""
    return Drive(armature * NATURAL_FREQ**2, 2 * DAMPING_RATIO * armature * NATURAL_FREQ, effort, armature, **passive)


def side_joint(name: str) -> str:
    """``left_knee_joint`` -> ``knee``: the key of a per-motor table."""
    return re.sub(r"^(left|right)_", "", name).removesuffix("_joint")


class Robot:
    name: str
    mjcf: str  # file name inside the asset directory
    extra_assets: tuple[str, ...] = ()  # further files in the asset directory that define the robot
    tracked_bodies: tuple[str, ...]  # bodies whose poses are tracked; the first one is the root
    fingertips: tuple[str, ...]  # ten fingertip frames
    surface_bodies: tuple[str, str]  # tracked bodies that stand for the left and right palm
    foot_bodies: tuple[str, str]
    foot_contact_z: float  # height of a foot body's origin below which the foot counts as on the ground
    com_body: str  # body whose centre of mass is randomized
    root_height: float
    default_pose: dict[str, float]  # joint-name pattern -> default position; unmatched joints are 0

    def __init__(self, asset_dir: str | Path):
        self.asset_dir = Path(asset_dir)

    # Rules a robot supplies.

    @property
    def driver_joint_names(self) -> tuple[str, ...]:
        """Finger joints the policy commands, in action order."""
        raise NotImplementedError

    @property
    def mimics(self) -> dict[str, tuple[str, float]]:
        """Finger joint -> (driver joint, ratio): its target is the ratio times the driver's target."""
        raise NotImplementedError

    def drive(self, joint: str) -> Drive:
        raise NotImplementedError

    def hand_link(self, body: str) -> tuple[int, int]:
        """(hand, finger) of a link: hand 0 (left) / 1 (right) for palm and finger links, finger 0..9
        for finger links; -1 where it does not apply (the body, fingertip frames, a palm's finger)."""
        raise NotImplementedError

    # Derived layout.

    @property
    def mjcf_path(self) -> Path:
        return self.asset_dir / self.mjcf

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
    def mimic_joint_names(self) -> tuple[str, ...]:
        return tuple(self.mimics)

    @cached_property
    def hand_joint_names(self) -> frozenset[str]:
        return frozenset(self.driver_joint_names + self.mimic_joint_names)

    @cached_property
    def body_joint_names(self) -> tuple[str, ...]:
        return tuple(n for n in self.joint_names if n not in self.hand_joint_names)

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
        """For every mimic joint, the position of its driver among the driver commands."""
        return [self.driver_joint_names.index(driver) for driver, _ in self.mimics.values()]

    @cached_property
    def mimic_ratio(self) -> list[float]:
        return [ratio for _, ratio in self.mimics.values()]

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
            for pattern, value in self.default_pose.items():
                if re.fullmatch(pattern, name):
                    q[i] = value
        return q

    @cached_property
    def control_center(self) -> np.ndarray:
        """Per-joint zero of the normalized PD target: default pose (body), mid-range (fingers)."""
        mid = self.joint_limits.mean(-1)
        return np.where([n in self.hand_joint_names for n in self.joint_names], mid, self.default_joint_pos).astype(np.float32)

    @cached_property
    def control_scale(self) -> np.ndarray:
        """Radians per unit of normalized PD target: 0.25 effort / kp (body), half-range (fingers)."""
        half = (self.joint_limits[:, 1] - self.joint_limits[:, 0]) / 2
        scale = [half[i] if n in self.hand_joint_names else 0.25 * self.drive(n).effort / self.drive(n).stiffness
                 for i, n in enumerate(self.joint_names)]  # fmt: skip
        return np.asarray(scale, dtype=np.float32)

    @property
    def fk_bodies(self) -> tuple[str, ...]:
        """Bodies whose FK the MDP needs: the tracked bodies followed by the ten fingertips."""
        return self.tracked_bodies + self.fingertips

    def entity_cfg(self):
        from mjlab.actuator import BuiltinPositionActuatorCfg
        from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

        actuators = []
        for name in self.joint_names:
            d = self.drive(name)
            actuators.append(BuiltinPositionActuatorCfg(
                target_names_expr=(name,), stiffness=d.stiffness, damping=d.damping, effort_limit=d.effort,
                armature=d.armature, frictionloss=d.frictionloss, viscous_damping=d.viscous_damping))  # fmt: skip
        return EntityCfg(
            init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, self.root_height), joint_pos=self.default_pose, joint_vel={".*": 0.0}),
            spec_fn=lambda: mujoco.MjSpec.from_file(str(self.mjcf_path)),
            articulation=EntityArticulationInfoCfg(actuators=tuple(actuators), soft_joint_pos_limit_factor=0.9),
        )
