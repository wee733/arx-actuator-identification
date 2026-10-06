"""Official Newton/MuJoCo replay of the coupled arm with per-tick IdealPD effort.

This mirrors the deployed Isaac Lab 3.0 contract: Newton + MuJoCo implicitfast,
one 5 ms solver step per actuator tick, explicit PD effort from the current
state, command delay in whole physics ticks (DelayedPDActuator), and joint
armature / Coulomb friction (``frictionloss``) written into the plant.
"""

from __future__ import annotations

from pathlib import Path

import newton
from newton.solvers import SolverMuJoCo
import numpy as np

from .data import ARM_JOINTS, ArmCapture
from .greybox import Inertia

# MuJoCo's mjMINIMP / mjMAXIMP.
MIN_IMPEDANCE = 0.0001
MAX_IMPEDANCE = 0.9999
from .plant import arm_xml, make_builder, mass_matrix


def coupled_arm_xml(source: str | Path, params: dict, pose: list[float], physics_dt: float) -> str:
    """Six free joints with zero passive damping, as in the deployed USD."""
    free = {j + 1: {"damping": 0.0, "armature": params["armature"][j],
                    "frictionloss": params["friction"][j]} for j in range(ARM_JOINTS)}
    return arm_xml(source, free, pose, physics_dt, name="R5a_coupled_identification")


def set_friction_impedance(model: newton.Model, solver: SolverMuJoCo, impedance) -> None:
    """Set the frictionloss constraint's solimp d0, per joint or for all joints.

    MuJoCo's frictionloss rows are regularised by their own solref/solimp, so the
    default d0 = 0.9 lets a loaded joint creep towards its target. d0 -> 1 removes
    the regularisation and the constraint becomes close to hard Coulomb friction.
    """
    impedance = np.broadcast_to(np.asarray(impedance, dtype=np.float64), (ARM_JOINTS,))
    if np.any(impedance < 0.0) or np.any(impedance >= 1.0):
        raise ValueError("friction impedance d0 must lie in [0, 1)")
    values = model.mujoco.solimpfriction.numpy()
    # Zero marks a friction-free joint and keeps MuJoCo's default; MuJoCo clamps
    # solimp to [mjMINIMP, mjMAXIMP], so an identified value is clamped the same way.
    values[:, 0] = np.where(impedance > 0.0, np.clip(impedance, MIN_IMPEDANCE, MAX_IMPEDANCE), values[:, 0])
    model.mujoco.solimpfriction.assign(values)
    solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)


def arm_inertia(source: str | Path, pose: list[float], physics_dt: float, step: float = 0.02) -> Inertia:
    """Central-difference first-order inertia model about the capture pose."""
    zero = {"armature": [0.0] * ARM_JOINTS, "friction": [0.0] * ARM_JOINTS}
    model = make_builder(coupled_arm_xml(source, zero, pose, physics_dt), dofs=ARM_JOINTS).finalize(device="cpu")
    pose = np.asarray(pose, dtype=np.float64)
    gradient = []
    for k in range(ARM_JOINTS):
        offset = np.zeros(ARM_JOINTS)
        offset[k] = step
        gradient.append((mass_matrix(model, pose + offset) - mass_matrix(model, pose - offset)) / (2 * step))
    return Inertia(pose, mass_matrix(model, pose), np.stack(gradient), mass_matrix(model, np.zeros(ARM_JOINTS)))


def replay(capture: ArmCapture, params: dict, source: str | Path, pose: list[float],
           physics_dt: float) -> dict:
    """Closed-loop replay from the capture's first measured state."""
    xml = coupled_arm_xml(source, params, pose, physics_dt)
    model = make_builder(xml, dofs=ARM_JOINTS).finalize(device="cpu")
    solver = SolverMuJoCo(model, use_mujoco_cpu=True, integrator="implicitfast",
                          disable_contacts=True, disable_sensors=True)
    set_friction_impedance(model, solver, params["friction_impedance"])
    state, following = model.state(), model.state()
    control = model.control()
    state.joint_q.assign(capture.q[0].astype(np.float32))
    state.joint_qd.assign(capture.velocity[0].astype(np.float32))
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    sample_dt = float(capture.t[1] - capture.t[0])
    substeps = round(sample_dt / physics_dt)
    fine_t = np.arange(len(capture.t) * substeps) * physics_dt
    command = np.stack([np.interp(fine_t, capture.t, capture.command[:, j])
                        for j in range(ARM_JOINTS)], axis=1) + np.asarray(params["bias"])
    delay = np.asarray(params["delay_ticks"], dtype=int)
    kp, kd = np.asarray(params["kp"]), np.asarray(params["kd"])
    effort = np.asarray(params["effort"])
    columns = np.arange(ARM_JOINTS)
    q = np.full_like(capture.q, np.nan)
    torque = np.full_like(capture.q, np.nan)
    q[0] = capture.q[0]
    diverged = False
    for tick in range(len(command) - substeps):
        position = state.joint_q.numpy().astype(np.float64)
        velocity = state.joint_qd.numpy().astype(np.float64)
        if not np.all(np.isfinite(position)) or np.max(np.abs(velocity)) > 50:
            diverged = True
            break
        target = command[np.maximum(tick - delay, 0), columns]
        effort_now = np.clip(kp * (target - position) - kd * velocity, -effort, effort)
        control.joint_f.assign(effort_now.astype(np.float32))
        state.clear_forces()
        solver.step(state, following, control, None, physics_dt)
        state, following = following, state
        if (tick + 1) % substeps == 0:
            sample = (tick + 1) // substeps
            q[sample] = state.joint_q.numpy()
            torque[sample - 1] = effort_now
    return {"q": q, "torque": torque, "diverged": diverged}


def rmse(capture: ArmCapture, result: dict, mask: np.ndarray) -> np.ndarray:
    """Per-joint position RMSE over masked samples; inf after divergence."""
    error = result["q"][mask] - capture.q[mask]
    if result["diverged"] or not np.all(np.isfinite(error)):
        return np.full(ARM_JOINTS, np.inf)
    return np.sqrt(np.mean(error ** 2, axis=0))
