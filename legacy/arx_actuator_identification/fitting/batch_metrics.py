from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from arx_actuator_identification.actuator import JointActuatorParams
from arx_actuator_identification.config import IdentificationConfig


@dataclass(frozen=True)
class BatchMetricResult:
    loss: np.ndarray
    q_rmse: np.ndarray
    qd_rmse: np.ndarray
    kp_regularization: np.ndarray
    tau_limit_regularization: np.ndarray
    gravity_scale_regularization: np.ndarray
    best_index: int


def compute_batch_metrics_np(
    q_sim: np.ndarray,
    qd_sim: np.ndarray,
    q_real: np.ndarray,
    qd_real: np.ndarray,
    params: list[JointActuatorParams],
    cfg: IdentificationConfig,
    time_shift_steps: np.ndarray | list[int] | None = None,
) -> BatchMetricResult:
    q_sim = np.asarray(q_sim, dtype=float)
    qd_sim = np.asarray(qd_sim, dtype=float)
    q_real = np.asarray(q_real, dtype=float)
    qd_real = np.asarray(qd_real, dtype=float)
    if q_sim.ndim != 2 or qd_sim.ndim != 2:
        raise ValueError("q_sim and qd_sim must have shape (B, N)")
    if q_sim.shape != qd_sim.shape:
        raise ValueError("q_sim and qd_sim must have the same shape")
    batch_size, sample_count = q_sim.shape
    if q_real.shape != (sample_count,) or qd_real.shape != (sample_count,):
        raise ValueError("q_real and qd_real must have shape (N,)")
    if len(params) != batch_size:
        raise ValueError("params length must match q_sim batch size")
    if time_shift_steps is None:
        shifts = np.zeros(batch_size, dtype=np.int64)
    else:
        shifts = np.asarray(time_shift_steps, dtype=np.int64)
        if shifts.shape != (batch_size,):
            raise ValueError("time_shift_steps must have shape (B,)")

    q_rmse = np.zeros(batch_size, dtype=float)
    qd_rmse = np.zeros(batch_size, dtype=float)
    for idx, shift in enumerate(shifts):
        if shift < 0:
            raise ValueError("time_shift_steps must be non-negative")
        if shift >= sample_count:
            raise ValueError("time_shift_steps must be smaller than replay length")
        if shift == 0:
            q_error = q_sim[idx] - q_real
            qd_error = qd_sim[idx] - qd_real
        else:
            q_error = q_sim[idx, :-shift] - q_real[shift:]
            qd_error = qd_sim[idx, :-shift] - qd_real[shift:]
        q_rmse[idx] = _rmse(q_error)
        qd_rmse[idx] = _rmse(qd_error)

    kp = np.asarray([param.kp for param in params], dtype=float)
    tau_limit = np.asarray([param.tau_limit for param in params], dtype=float)
    gravity_scale = np.asarray([param.gravity_scale for param in params], dtype=float)
    kp_reg = float(cfg.w_reg_kp) * np.square(kp)
    tau_reg = float(cfg.w_reg_tau) * np.square(tau_limit)
    # 与 MuJoCo search 的目标函数对齐：开重力时拉向理论值 1.0；
    # 关重力时拉向 0.0，避免 no-gravity ablation 被一个无意义常数污染 loss。
    gravity_target = 1.0 if bool(cfg.gravity_compensation) else 0.0
    gravity_reg = float(cfg.w_reg_gravity_scale) * np.square(gravity_scale - gravity_target)
    loss = float(cfg.w_q) * q_rmse + float(cfg.w_qd) * qd_rmse + kp_reg + tau_reg + gravity_reg
    best_index = int(np.argmin(loss))
    return BatchMetricResult(
        loss=loss,
        q_rmse=q_rmse,
        qd_rmse=qd_rmse,
        kp_regularization=kp_reg,
        tau_limit_regularization=tau_reg,
        gravity_scale_regularization=gravity_reg,
        best_index=best_index,
    )


def _rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))
