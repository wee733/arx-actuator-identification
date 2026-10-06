"""Simulation replay backends used by offline fitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from arx_actuator_identification.actuator import JointActuatorParams
from arx_actuator_identification.dataset import SweepDataset


@dataclass
class ReplayResult:
    t: np.ndarray
    q: np.ndarray
    qd: np.ndarray
    tau: np.ndarray


class JointReplayBackend(Protocol):
    def replay(self, dataset: SweepDataset, params: JointActuatorParams) -> ReplayResult:
        """Replay one active-joint dataset with candidate actuator parameters."""


class BatchJointReplayBackend(Protocol):
    def replay_batch(self, dataset: SweepDataset, candidates: list[JointActuatorParams]):
        """Replay one active-joint dataset for a batch of candidate actuator parameters."""
