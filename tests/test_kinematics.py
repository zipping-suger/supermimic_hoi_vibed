import mujoco
import numpy as np
import torch

from supermimic_hoi.kinematics import Kinematics
from supermimic_hoi.rotations import mat_to_quat, so3_exp


# (joints, actions, body joints, finger drivers, mimic joints)
LAYOUT = {"g1_inspire": (53, 41, 29, 12, 12)}


def test_robot_layout(robot):
    layout = (robot.num_joints, robot.num_actions, len(robot.body_idx), len(robot.driver_idx), len(robot.mimic_idx))
    assert layout == LAYOUT[robot.name]
    assert sorted(robot.body_idx + robot.driver_idx + robot.mimic_idx) == list(range(robot.num_joints))
    # Every mimic joint names a driver on its own hand.
    for mimic, driver in zip(robot.mimic_joint_names, robot.mimic_driver):
        assert mimic[0] == robot.driver_joint_names[driver][0]
    # Both hands have a palm and five fingers of collision links; fingertip frames belong to neither.
    links = [robot.hand_link(robot.model.body(b).name) for b in range(robot.model.nbody)]
    assert {finger for _, finger in links} == set(range(-1, 10)) and {hand for hand, _ in links} == {-1, 0, 1}
    assert all(finger // 5 == hand for hand, finger in links if finger >= 0)
    assert all(robot.hand_link(tip) == (-1, -1) for tip in robot.fingertips)
    assert len(robot.fingertips) == 10 and set(robot.surface_bodies + robot.foot_bodies) <= set(robot.tracked_bodies)
    # No hinge joint has a reference offset, which the torch FK relies on.
    hinge = robot.model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE
    assert not robot.model.qpos0[robot.model.jnt_qposadr[hinge]].any()


def test_fk_matches_mujoco_positions_rotations_and_velocities(robot):
    torch.manual_seed(0)
    model, data = robot.model, mujoco.MjData(robot.model)
    kin = Kinematics(robot)
    lo, hi = kin.q_lo.double(), kin.q_hi.double()
    for _ in range(5):
        p, rot = torch.randn(3, dtype=torch.float64), so3_exp(torch.randn(3, dtype=torch.float64))
        v, w = torch.randn(3, dtype=torch.float64), torch.randn(3, dtype=torch.float64)
        q, qd = lo + torch.rand(len(lo), dtype=torch.float64) * (hi - lo), torch.randn(len(lo), dtype=torch.float64)
        data.qpos[:] = torch.cat((p, mat_to_quat(rot), q)).numpy()
        data.qvel[:] = torch.cat((v, rot.T @ w, qd)).numpy()
        mujoco.mj_forward(model, data)
        kin64 = Kinematics(robot)
        for name in ("axes", "anchors", "off_p", "off_R"):
            setattr(kin64, name, getattr(kin64, name).double())
        bp, bR, bv, bw = kin64.forward(p, rot, v, w, q, qd)
        for i, name in enumerate(robot.fk_bodies):
            body = model.body(name).id
            assert np.allclose(bp[i].numpy(), data.xpos[body], atol=1e-6), name
            assert np.allclose(bR[i].numpy(), data.xmat[body].reshape(3, 3), atol=1e-6), name
            velocity = np.zeros(6)
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, body, velocity, 0)
            assert np.allclose(bw[i].numpy(), velocity[:3], atol=1e-5), name
            assert np.allclose(bv[i].numpy(), velocity[3:], atol=1e-5), name
