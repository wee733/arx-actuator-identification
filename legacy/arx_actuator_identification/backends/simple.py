from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from arx_actuator_identification.actuator import JointActuatorParams, SimplePdActuator
from arx_actuator_identification.backends import ReplayResult
from arx_actuator_identification.dataset import SweepDataset


@dataclass
class SingleJointPlant:
    inertia: float = 1.0
    viscous: float = 0.2

    def __post_init__(self) -> None:
        if self.inertia <= 0.0:
            raise ValueError("inertia must be positive")
        if self.viscous < 0.0:
            raise ValueError("viscous must be non-negative")

    def reset(self, q0: float = 0.0, qd0: float = 0.0) -> None:
        self.q = float(q0)
        self.qd = float(qd0)

    def step(self, tau: float, dt: float) -> tuple[float, float]:
        qdd = (float(tau) - self.viscous * self.qd) / self.inertia
        self.qd += qdd * dt
        self.q += self.qd * dt
        return self.q, self.qd


def simulate_joint_response(
    q_cmd: np.ndarray,
    params: JointActuatorParams,
    plant: SingleJointPlant,
    dt: float,
    q0: float = 0.0,
    qd0: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    result = SimpleJointReplayBackend(plant=plant).replay_arrays(q_cmd, params, dt, q0=q0, qd0=qd0)
    return result.q, result.qd


class SimpleJointReplayBackend:
    def __init__(self, plant: SingleJointPlant | None = None):
        self.plant = SingleJointPlant() if plant is None else plant

    def replay(self, dataset: SweepDataset, params: JointActuatorParams) -> ReplayResult:
        dataset.validate()
        joint = dataset.active_joint
        dt = float(np.median(np.diff(dataset.t)))
        return self.replay_arrays(
            dataset.q_cmd[:, joint],
            params,
            dt,
            q0=float(dataset.q[0, joint]),
            qd0=float(dataset.qd[0, joint]),
            t=dataset.t,
        )

    def replay_arrays(
        self,
        q_cmd: np.ndarray,
        params: JointActuatorParams,
        dt: float,
        q0: float = 0.0,
        qd0: float = 0.0,
        t: np.ndarray | None = None,
    ) -> ReplayResult:
        q_cmd = np.asarray(q_cmd, dtype=float)
        actuator = SimplePdActuator([params])
        actuator.reset(np.array([q_cmd[0]]))
        plant = SingleJointPlant(inertia=self.plant.inertia, viscous=self.plant.viscous)
        plant.reset(q0=q0, qd0=qd0)
        q_values = []
        qd_values = []
        tau_values = []
        for command in q_cmd:
            q_values.append(float(plant.q))
            qd_values.append(float(plant.qd))
            tau = actuator.step(np.array([command]), np.array([plant.q]), np.array([plant.qd]))[0]
            tau_values.append(tau)
            plant.step(tau, dt)
        t_values = np.arange(len(q_cmd), dtype=float) * dt if t is None else np.asarray(t, dtype=float)
        return ReplayResult(
            t=t_values,
            q=np.asarray(q_values),
            qd=np.asarray(qd_values),
            tau=np.asarray(tau_values),
        )
