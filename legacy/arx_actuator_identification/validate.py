from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ValidationMetrics:
    q_rmse: float
    qd_rmse: float
    peak_error: float


def compute_validation_metrics(q_real: np.ndarray, q_sim: np.ndarray, qd_real: np.ndarray, qd_sim: np.ndarray) -> ValidationMetrics:
    q_error = np.asarray(q_sim) - np.asarray(q_real)
    qd_error = np.asarray(qd_sim) - np.asarray(qd_real)
    return ValidationMetrics(
        q_rmse=float(np.sqrt(np.mean(np.square(q_error)))),
        qd_rmse=float(np.sqrt(np.mean(np.square(qd_error)))),
        peak_error=float(np.max(np.abs(q_error))),
    )
