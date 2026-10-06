from __future__ import annotations

from arx_actuator_identification.config import IdentificationConfig
from arx_actuator_identification.dataset import SweepDataset
from arx_actuator_identification.fitting.search import FitResult, fit_joint_with_backend
from arx_actuator_identification.backends.simple import SimpleJointReplayBackend, SingleJointPlant


def fit_joint_parameters(dataset: SweepDataset, cfg: IdentificationConfig, plant: SingleJointPlant | None = None) -> FitResult:
    backend = SimpleJointReplayBackend(plant=plant)
    return fit_joint_with_backend(dataset, cfg, backend)


__all__ = ["FitResult", "fit_joint_parameters"]
