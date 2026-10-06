"""Static current-to-torque calibration from `profile=static` captures.

At rest the motor torque balances gravity and static friction:

    k_j * i_j = g_j(q) + f_j,   |f_j| <= friction_j

g_j comes from the model (link masses and centres of mass), which a CAD model
gets far more reliably than rotor inertia. Each joint has an unknown torque
constant k_j and a current offset b_j; a least-squares fit over many poses with
varied gravity load gives both. The friction band shows up as residual scatter.
The ARX firmware gravity compensation does not matter here: the measured
current is whatever the drive applies, compensation included.

The base must be level during capture, with gravity along -z of the arm base.
"""

from __future__ import annotations

from pathlib import Path

import newton
import numpy as np
import warp as wp
import yaml

from .data import ARM_JOINTS
from .plant import arm_xml

GRAVITY = 9.81


def gravity_torques(source: str | Path, poses: np.ndarray, physics_dt: float) -> np.ndarray:
    """Joint gravity torques g(q) (N*m) of the six-joint arm, shape (poses, 6)."""
    free = {j + 1: {"damping": 0.0} for j in range(ARM_JOINTS)}
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -GRAVITY))
    builder.add_mjcf(arm_xml(source, free, [0.0] * ARM_JOINTS, physics_dt), floating=False,
                     parse_meshes=False, parse_visuals=False, parse_sites=False, collapse_fixed_joints=True)
    model = builder.finalize(device="cpu")
    state = model.state()
    size = model.max_dofs_per_articulation
    mass = wp.zeros((model.articulation_count, size, size), device="cpu")
    gravity = wp.zeros(model.joint_dof_count, device="cpu")
    coriolis = wp.zeros_like(gravity)
    result = []
    for pose in poses:
        state.joint_q.assign(np.asarray(pose, dtype=np.float32))
        state.joint_qd.zero_()
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        newton.eval_inverse_dynamics_passive(model, state, mass_matrix=mass,
                                             coriolis_force=coriolis, gravity_force=gravity)
        # Newton returns the generalised gravity force; the motor holds its negative.
        result.append(-gravity.numpy().astype(np.float64))
    return np.array(result)


def static_samples(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Mean measured pose and current of each recorded static pose."""
    rows = np.genfromtxt(path, delimiter=",", names=True)
    if "recording" not in rows.dtype.names or "cur0" not in rows.dtype.names:
        raise ValueError(f"{path} is not a static capture with current columns")
    poses, currents = [], []
    for index in np.unique(rows["pose_index"][rows["recording"] == 1]):
        held = (rows["pose_index"] == index) & (rows["recording"] == 1)
        q = np.stack([rows[f"q_real{j}"][held] for j in range(ARM_JOINTS)], axis=1)
        i = np.stack([rows[f"cur{j}"][held] for j in range(ARM_JOINTS)], axis=1)
        if not np.all(np.isfinite(i)):
            raise ValueError(f"{path}: missing current at pose {int(index)}")
        poses.append(q.mean(0))
        currents.append(i.mean(0))
    return np.array(poses), np.array(currents)


def fit_torque_constants(current: np.ndarray, torque: np.ndarray, min_load_span: float = 0.5) -> dict:
    """Per-joint least squares torque = k * current + c, with fit diagnostics.

    Joints whose gravity load varies by less than `min_load_span` N*m across
    the poses are reported as unresolved instead of fitted.
    """
    joints = []
    for j in range(ARM_JOINTS):
        span = float(np.ptp(torque[:, j]))
        if span < min_load_span:
            joints.append({"resolved": False, "gravity_span_nm": span})
            continue
        design = np.column_stack([current[:, j], np.ones(len(current))])
        (k, c), *_ = np.linalg.lstsq(design, torque[:, j], rcond=None)
        residual = torque[:, j] - design @ np.array([k, c])
        joints.append({"resolved": True, "torque_constant_nm_per_a": float(k), "offset_nm": float(c),
                       "residual_rms_nm": float(np.sqrt(np.mean(residual ** 2))),
                       "residual_max_nm": float(np.max(np.abs(residual))), "gravity_span_nm": span})
    return {"joints": joints}


def calibrate(cfg: dict, captures: list[Path], output: Path) -> dict:
    poses, currents = zip(*(static_samples(path) for path in captures))
    poses, currents = np.concatenate(poses), np.concatenate(currents)
    torque = gravity_torques(cfg["plant"]["source_xml"], poses, cfg["plant"]["physics_dt"])
    result = fit_torque_constants(currents, torque)
    result.update({
        "captures": [str(p) for p in captures], "poses": len(poses),
        "model": cfg["plant"]["source_xml"],
        "note": "torque = torque_constant * current + offset; gravity from the model, base level",
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(result, sort_keys=False))
    for j, joint in enumerate(result["joints"], start=1):
        if joint["resolved"]:
            print(f"joint {j}: k={joint['torque_constant_nm_per_a']:.4f} N*m/A "
                  f"offset={joint['offset_nm']:+.3f} N*m residual rms={joint['residual_rms_nm']:.3f} N*m")
        else:
            print(f"joint {j}: gravity load span {joint['gravity_span_nm']:.3f} N*m is too small to calibrate")
    return result
