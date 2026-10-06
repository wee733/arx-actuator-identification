"""Coupled six-joint grey-box actuator identification.

The deployed simulator runs an Isaac Lab IdealPD/DelayedPD actuator on every
5 ms physics tick of a fixed-pose, gravity-neutral arm (Newton + MuJoCo
implicitfast, zero passive damping). Single-joint captures also move the other
joints through the arm's inertia coupling, so all six joints are simulated
together and scored against all six encoders.

Each tick of the differentiable reduced model is

    tau = clip(kp * (q_cmd(t - delay) - q) - kd * qd, +-effort)
    M(q) qdd = tau - c(q, qd) + f,    |f_j| <= friction_j
    qd' = qd + dt * qdd,  q' = q + dt * qd'

with M(q) expanded to first order about the capture pose plus armature, c the
matching Coriolis/centrifugal terms, and f MuJoCo's soft frictionloss
constraint (same regulariser and reference acceleration as the solver, with the
solimp d0 identified). q_cmd carries an optional constant per-joint bias. It is
only used for gradients; official Newton/MuJoCo replays score the result.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .data import ARM_JOINTS, ArmCapture

PARAMETERS = ("kp", "kd", "delay_s", "armature", "friction", "friction_impedance", "bias")


@dataclass
class Windows:
    """Batched simulation windows on the physics grid.

    command: (B, history + ticks, 6) smooth command, tick 0 at index `history`.
    q0, v0: (B, 6) measured start state. target: (B, samples, 6) encoder
    positions at every `substeps`-th tick after the start. weight: (B, samples).
    """

    command: torch.Tensor
    q0: torch.Tensor
    v0: torch.Tensor
    target: torch.Tensor
    weight: torch.Tensor
    history: int
    substeps: int


def make_windows(captures: list[ArmCapture], mask_name: str, length_s: float | None,
                 stride_s: float, physics_dt: float, history_s: float,
                 dtype: torch.dtype = torch.float64) -> Windows:
    """Cut windows whose samples all lie in a mask; None length means whole captures.

    Whole-capture windows start at sample zero and weight only masked samples,
    matching the closed-loop replay used for validation.
    """
    sample_dt = float(captures[0].t[1] - captures[0].t[0])
    substeps = round(sample_dt / physics_dt)
    history = int(np.ceil(history_s / physics_dt)) + 1
    commands, q0, v0, targets, weights = [], [], [], [], []
    for capture in captures:
        mask = getattr(capture, mask_name)
        fine_t = np.arange(len(capture.t) * substeps) * physics_dt
        # The deployed EMA output is smooth at 1 kHz; interpolate the 100 Hz log.
        fine = np.stack([np.interp(fine_t, capture.t, capture.command[:, j])
                         for j in range(ARM_JOINTS)], axis=1)
        padded = np.concatenate([np.repeat(fine[:1], history, axis=0), fine])
        if length_s is None:
            starts, samples = [0], len(capture.t) - 1
        else:
            samples = round(length_s / sample_dt)
            stride = max(1, round(stride_s / sample_dt))
            valid = np.convolve(mask.astype(int), np.ones(samples + 1, dtype=int), "valid") == samples + 1
            starts = np.flatnonzero(valid)[::stride]
        for start in starts:
            first = start * substeps
            commands.append(padded[first:first + history + samples * substeps])
            q0.append(capture.q[start])
            v0.append(capture.velocity[start])
            targets.append(capture.q[start + 1:start + samples + 1])
            weights.append(mask[start + 1:start + samples + 1].astype(float))
    if not commands:
        raise ValueError(f"No {mask_name} windows of {length_s} s")
    width = max(len(c) for c in commands)
    span = max(len(t) for t in targets)
    pad = lambda rows, size: np.concatenate([rows, np.repeat(rows[-1:], size - len(rows), axis=0)])
    tensor = lambda arrays: torch.tensor(np.stack(arrays), dtype=dtype)
    return Windows(
        command=tensor([pad(c, width) for c in commands]), q0=tensor(q0), v0=tensor(v0),
        target=tensor([pad(t, span) for t in targets]),
        weight=tensor([np.concatenate([w, np.zeros(span - len(w))]) for w in weights]),
        history=history, substeps=substeps,
    )


@dataclass
class Inertia:
    """Arm inertia without armature: M at the pose, dM/dq_k at the pose, M at q = 0."""

    pose: np.ndarray
    at_pose: np.ndarray
    gradient: np.ndarray
    at_zero: np.ndarray


# MuJoCo defaults for a dof frictionloss row: solref timeconst, solimp d0 (the
# impedance at zero violation, identified as friction_impedance) and its width
# limit d1. Only d0 is fitted, since it is what sets the stiction band.
FRICTION_TIMECONST = 0.02
FRICTION_IMPEDANCE = 0.9
FRICTION_IMPEDANCE_MAX = 0.95


class GreyBoxArm(nn.Module):
    """Per-joint IdealPD + command delay and bias + armature + soft Coulomb friction on a coupled arm."""

    def __init__(self, inertia: Inertia, kp, kd, delay_s, armature, friction, effort,
                 physics_dt: float, max_delay_s: float, trainable: set[str],
                 friction_iterations: int = 4, compile: bool = False):
        super().__init__()
        # torch.compile fuses the per-tick graph; it pays a one-off compile per batch shape.
        self.compile = compile
        self.compiled_tick = torch.compile(physics_tick, dynamic=False) if compile else None
        as_tensor = lambda x: torch.as_tensor(np.asarray(x, dtype=np.float64))
        self.register_buffer("pose", as_tensor(inertia.pose))
        self.register_buffer("inertia", as_tensor(inertia.at_pose))
        self.register_buffer("inertia_gradient", as_tensor(inertia.gradient))
        self.register_buffer("inertia_zero", as_tensor(inertia.at_zero))
        self.register_buffer("effort", as_tensor(effort))
        self.physics_dt = physics_dt
        self.max_delay_s = max_delay_s
        self.friction_iterations = friction_iterations
        # Positive parameters are optimised in log space; a small floor keeps
        # armature/friction differentiable when they are near zero.
        self.floor = {"armature": 1e-6, "friction": 1e-5}
        initial = {"kp": kp, "kd": kd, "armature": armature, "friction": friction}
        for name, value in initial.items():
            raw = torch.log(as_tensor(value) + self.floor.get(name, 0.0))
            self.register_parameter(f"log_{name}", nn.Parameter(raw, requires_grad=name in trainable))
        # d0 = 1 would be a perfectly hard constraint; keep it strictly below.
        self.impedance_logit = nn.Parameter(
            torch.logit(as_tensor(np.full(ARM_JOINTS, FRICTION_IMPEDANCE))),
            requires_grad="friction_impedance" in trainable)
        fraction = np.clip(np.asarray(delay_s, dtype=np.float64) / max_delay_s, 1e-4, 1 - 1e-4)
        self.delay_logit = nn.Parameter(torch.logit(as_tensor(fraction)),
                                        requires_grad="delay_s" in trainable)
        self.use_friction = "friction" in trainable or bool(np.any(np.asarray(friction) > 0))
        # Constant offset between the commanded and the held joint position
        # (encoder zero or residual gravity compensation), added to the target.
        self.bias = nn.Parameter(torch.zeros(ARM_JOINTS, dtype=torch.float64), requires_grad="bias" in trainable)

    def value(self, name: str) -> torch.Tensor:
        if name == "delay_s":
            return torch.sigmoid(self.delay_logit) * self.max_delay_s
        if name == "friction_impedance":
            # MuJoCo clamps solimp to at most mjMAXIMP = 0.9999; mirror it.
            return torch.sigmoid(self.impedance_logit).clamp(max=0.9999)
        if name == "bias":
            return self.bias
        return torch.exp(getattr(self, f"log_{name}")) - self.floor.get(name, 0.0)

    def parameters_dict(self) -> dict[str, list[float]]:
        with torch.no_grad():
            return {name: self.value(name).tolist() for name in PARAMETERS}

    def delayed_command(self, windows: Windows, ticks: int) -> torch.Tensor:
        """Fractional per-joint delay by linear interpolation on the physics grid."""
        delay = self.value("delay_s") / self.physics_dt
        position = windows.history + torch.arange(ticks, dtype=delay.dtype)[:, None] - delay[None, :]
        lower = torch.floor(position).detach()
        fraction = position - lower
        lower = lower.long().clamp(0, windows.command.shape[1] - 2)
        gather = lambda index: torch.gather(windows.command, 1, index.expand(windows.command.shape[0], -1, -1))
        return gather(lower[None]) * (1 - fraction) + gather(lower[None] + 1) * fraction

    def forward(self, windows: Windows) -> torch.Tensor:
        """Simulated encoder positions at the target sample times, (B, samples, 6)."""
        samples = windows.target.shape[1]
        ticks = samples * windows.substeps
        command = self.delayed_command(windows, ticks) + self.bias
        armature = self.value("armature")
        friction = self.value("friction") if self.use_friction else torch.zeros_like(armature)
        invweight0 = torch.diagonal(torch.linalg.inv(self.inertia_zero + torch.diag(armature)))
        constants = dict(
            kp=self.value("kp"), kd=self.value("kd"), effort=self.effort,
            inertia=self.inertia + torch.diag(armature), gradient=self.inertia_gradient, pose=self.pose,
            friction=friction,
            regulariser=(1 / self.value("friction_impedance") - 1) * invweight0,
            reference_gain=2 / (FRICTION_IMPEDANCE_MAX * max(FRICTION_TIMECONST, 2 * self.physics_dt)),
            dt=self.physics_dt, iterations=self.friction_iterations if self.use_friction else 0,
        )
        step = self.compiled_tick if self.compile else physics_tick
        q, v = windows.q0, windows.v0
        outputs = []
        for tick in range(ticks):
            q, v = step(q, v, command[:, tick], **constants)
            if (tick + 1) % windows.substeps == 0:
                outputs.append(q)
        return torch.stack(outputs, 1)


def physics_tick(q, v, target, kp, kd, effort, inertia, gradient, pose, friction, regulariser,
                 reference_gain: float, dt: float, iterations: int):
    """One 5 ms tick: IdealPD effort, first-order M(q) with Coriolis, soft frictionloss."""
    torque = kp * (target - q) - kd * v
    torque = torch.maximum(torch.minimum(torque, effort), -effort)
    mass = inertia + torch.einsum("kij,bk->bij", gradient, q - pose)
    coriolis = (torch.einsum("kij,bj,bk->bi", gradient, v, v)
                - 0.5 * torch.einsum("ijk,bj,bk->bi", gradient, v, v))
    inverse = torch.linalg.inv(mass)
    acceleration = torch.einsum("bij,bj->bi", inverse, torque - coriolis)
    if iterations:
        # MuJoCo's soft frictionloss rows by projected Gauss-Seidel: minimise
        # 0.5 f'(A + R)f + f'(a0 - aref) over |f_j| <= friction_j with A = M^-1
        # and aref = -b qd.
        force = torch.zeros_like(v)
        diagonal = torch.diagonal(inverse, dim1=1, dim2=2)
        unit = torch.eye(ARM_JOINTS, dtype=v.dtype, device=v.device)
        for _ in range(iterations):
            for j in range(ARM_JOINTS):
                row = acceleration[:, j] + (inverse[:, j] * force).sum(1)
                residual = row + reference_gain * v[:, j] + regulariser[j] * force[:, j]
                update = force[:, j] - residual / (diagonal[:, j] + regulariser[j])
                clipped = torch.maximum(torch.minimum(update, friction[j]), -friction[j])
                force = force + (clipped - force[:, j])[:, None] * unit[j]
        acceleration = acceleration + torch.einsum("bij,bj->bi", inverse, force)
    v = v + dt * acceleration
    return q + dt * v, v


def position_rmse(prediction: torch.Tensor, windows: Windows) -> torch.Tensor:
    """Per-joint weighted position RMSE in rad, (6,)."""
    weight = windows.weight[..., None]
    return torch.sqrt(((prediction - windows.target) ** 2 * weight).sum((0, 1)) / weight.sum())
