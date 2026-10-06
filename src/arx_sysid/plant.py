from __future__ import annotations

import copy
from pathlib import Path
import xml.etree.ElementTree as ET

import newton
import numpy as np
from scipy.spatial.transform import Rotation
import warp as wp


def arm_xml(source: str | Path, free_joints: dict[int, dict[str, float]], pose: list[float],
            physics_dt: float, name: str = "R5a_identification") -> str:
    """Keep source inertias; fix the base, the other joints and the closed gripper.

    `free_joints` maps each one-based joint that stays articulated to MJCF joint
    attribute overrides such as damping, armature or frictionloss. Fixed joints
    are baked into their body orientation at `pose`. No mesh, floor/contact,
    policy or live robot is needed.
    """
    source_root = ET.parse(source).getroot()
    arm = copy.deepcopy(source_root.find(".//body[@name='R5a']"))
    if arm is None:
        raise ValueError("Source MJCF has no R5a body")
    for body in arm.iter("body"):
        for child in list(body):
            if child.tag not in {"body", "inertial", "joint"}:
                body.remove(child)
        for element in list(body.findall("joint")):
            number = int(element.attrib["name"].removeprefix("R5a_joint"))
            if number in free_joints:
                for key, value in free_joints[number].items():
                    element.set(key, str(value))
                continue
            angle = pose[number - 1] if number <= 6 else 0.0
            if element.get("type", "hinge") == "hinge":
                axis = np.fromstring(element.get("axis", "0 0 1"), sep=" ")
                axis /= np.linalg.norm(axis)
                base = Rotation.from_quat(
                    np.fromstring(body.get("quat", "1 0 0 0"), sep=" "), scalar_first=True
                )
                rotation = base * Rotation.from_rotvec(axis * angle)
                body.set("quat", " ".join(map(str, rotation.as_quat(scalar_first=True))))
            elif angle != 0.0:
                raise ValueError("Only the closed gripper pose is supported")
            body.remove(element)
    root = ET.Element("mujoco", model=name)
    ET.SubElement(root, "compiler", angle="radian", autolimits="true", fusestatic="true")
    ET.SubElement(root, "option", gravity="0 0 0", timestep=str(physics_dt), integrator="implicitfast")
    ET.SubElement(root, "worldbody").append(arm)
    return ET.tostring(root, encoding="unicode")


def single_joint_xml(source: str | Path, joint: int, cfg: dict) -> str:
    """A single-joint, fixed-pose plant that keeps the source passive damping."""
    return arm_xml(source, {joint: {"armature": cfg["joint_armature"]}}, cfg["fixed_pose"],
                   cfg["physics_dt"], name=f"R5a_joint{joint}_identification")


def make_builder(xml: str, dofs: int = 1) -> newton.ModelBuilder:
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.add_mjcf(xml, floating=False, parse_meshes=False, parse_visuals=False,
                     parse_sites=False, collapse_fixed_joints=True)
    # Force-only actuation; passive joint damping in the XML remains in the plant.
    builder.joint_target_ke[:] = [0.0] * len(builder.joint_target_ke)
    builder.joint_target_kd[:] = [0.0] * len(builder.joint_target_kd)
    if builder.joint_dof_count != dofs:
        raise ValueError(f"Expected {dofs} active DOF, found {builder.joint_dof_count}")
    return builder


def mass_matrix(model: newton.Model, q: np.ndarray) -> np.ndarray:
    """Joint-space inertia of a single articulation at q, excluding joint armature."""
    state = model.state()
    state.joint_q.assign(np.asarray(q, dtype=np.float32))
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    size = model.max_dofs_per_articulation
    matrix = wp.zeros((model.articulation_count, size, size), device=model.device)
    newton.eval_mass_matrix(model, state, matrix)
    return matrix.numpy()[0].astype(np.float64)


class InverseDynamics:
    """Batch the official Newton 1.6 inverse-dynamics API on the local GPU."""

    def __init__(self, xml: str, batch_size: int, device: str):
        single = make_builder(xml)
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        for i in range(batch_size):
            builder.add_builder(single, label_prefix=f"sample{i}/")
        self.model = builder.finalize(device=device)
        self.state = self.model.state()
        self.batch_size = batch_size
        self.mass = wp.zeros((self.model.articulation_count,
                              self.model.max_dofs_per_articulation,
                              self.model.max_dofs_per_articulation), device=device)
        self.coriolis = wp.zeros(self.model.joint_dof_count, device=device)
        self.gravity = wp.zeros_like(self.coriolis)
        self.qdd = wp.zeros_like(self.coriolis)
        self.torque = wp.zeros_like(self.coriolis)
        self.damping = self.model.joint_damping.numpy()

    def labels(self, q: np.ndarray, velocity: np.ndarray, acceleration: np.ndarray) -> np.ndarray:
        result = np.empty_like(q)
        for start in range(0, len(q), self.batch_size):
            stop = min(start + self.batch_size, len(q))
            count = stop - start
            pad = lambda x: np.pad(x[start:stop], (0, self.batch_size - count), mode="edge").astype(np.float32)
            padded_velocity = pad(velocity)
            self.state.joint_q.assign(pad(q))
            self.state.joint_qd.assign(padded_velocity)
            self.qdd.assign(pad(acceleration))
            newton.eval_fk(self.model, self.state.joint_q, self.state.joint_qd, self.state)
            newton.eval_inverse_dynamics_passive(
                self.model, self.state, mass_matrix=self.mass,
                coriolis_force=self.coriolis, gravity_force=self.gravity,
            )
            newton.eval_inverse_dynamics_force(
                self.model, self.state, mass_matrix=self.mass, joint_qdd=self.qdd,
                coriolis_force=self.coriolis, gravity_force=self.gravity, joint_f=self.torque,
            )
            # The official inverse API returns M*qdd+C+G. Motor effort also
            # offsets the source model's passive viscous damping.
            result[start:stop] = (self.torque.numpy() + self.damping * padded_velocity)[:count]
        return result
