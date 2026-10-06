from __future__ import annotations

import numpy as np


CONTROLLED_JOINT_NAMES = [
    "b2_description_FL_hip_joint",
    "b2_description_FL_thigh_joint",
    "b2_description_FL_calf_joint",
    "b2_description_FR_hip_joint",
    "b2_description_FR_thigh_joint",
    "b2_description_FR_calf_joint",
    "b2_description_RL_hip_joint",
    "b2_description_RL_thigh_joint",
    "b2_description_RL_calf_joint",
    "b2_description_RR_hip_joint",
    "b2_description_RR_thigh_joint",
    "b2_description_RR_calf_joint",
    "R5a_joint1",
    "R5a_joint2",
    "R5a_joint3",
    "R5a_joint4",
    "R5a_joint5",
    "R5a_joint6",
]

ARM_JOINT_NAMES = [
    "R5a_joint1",
    "R5a_joint2",
    "R5a_joint3",
    "R5a_joint4",
    "R5a_joint5",
    "R5a_joint6",
]

GRIPPER_JOINT_NAMES = [
    "R5a_joint7",
    "R5a_joint8",
]

Q_DOWN = np.array(
    [
        -0.5,
        1.36,
        -2.65,
        0.5,
        1.36,
        -2.65,
        -0.5,
        1.36,
        -2.65,
        0.5,
        1.36,
        -2.65,
    ],
    dtype=float,
)

DEFAULT_ARM_POS = np.array([0.0, 1.0, 0.8, 0.0, 0.0, 0.0], dtype=float)
GRIPPER_CLOSED_POS = np.array([0.0, 0.0], dtype=float)


def arm_joint_name(active_joint: int) -> str:
    if not 0 <= active_joint < len(ARM_JOINT_NAMES):
        raise ValueError("active_joint must be in [0, 5]")
    return ARM_JOINT_NAMES[active_joint]


def controlled_index_for_arm(active_joint: int) -> int:
    return 12 + active_joint


def controlled_defaults_with_down_base(arm_pos: np.ndarray | None = None) -> np.ndarray:
    arm = DEFAULT_ARM_POS if arm_pos is None else np.asarray(arm_pos, dtype=float)
    if arm.shape != (6,):
        raise ValueError("arm_pos must have shape (6,)")
    return np.concatenate([Q_DOWN, arm])
