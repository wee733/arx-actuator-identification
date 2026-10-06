from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from arx_actuator_identification.actuator import JointActuatorParams
from arx_actuator_identification.backends import JointReplayBackend
from arx_actuator_identification.config import IdentificationConfig
from arx_actuator_identification.dataset import SweepDataset


@dataclass
class FitMetrics:
    loss: float
    q_rmse: float
    qd_rmse: float
    kp_regularization: float
    tau_limit_regularization: float
    gravity_scale_regularization: float = 0.0


@dataclass
class FitResult:
    params: JointActuatorParams
    loss: float
    q_rmse: float
    qd_rmse: float
    time_shift_steps: int = 0
    kp_regularization: float = 0.0
    tau_limit_regularization: float = 0.0
    gravity_scale_regularization: float = 0.0
    trace: list[dict[str, Any]] | None = None


def _sample_uniform(rng: np.random.Generator, bounds: tuple[float, float]) -> float:
    return float(rng.uniform(bounds[0], bounds[1]))


def _rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def compute_metrics(
    q_sim: np.ndarray,
    qd_sim: np.ndarray,
    q_real: np.ndarray,
    qd_real: np.ndarray,
    params: JointActuatorParams,
    cfg: IdentificationConfig,
    time_shift_steps: int = 0,
) -> FitMetrics:
    q_sim = np.asarray(q_sim, dtype=float)
    qd_sim = np.asarray(qd_sim, dtype=float)
    q_real = np.asarray(q_real, dtype=float)
    qd_real = np.asarray(qd_real, dtype=float)
    shift = int(time_shift_steps)
    if shift < 0:
        raise ValueError("time_shift_steps must be non-negative")
    if shift > 0:
        if shift >= len(q_real) or shift >= len(q_sim):
            raise ValueError("time_shift_steps must be smaller than the replay length")
        q_sim = q_sim[:-shift]
        qd_sim = qd_sim[:-shift]
        q_real = q_real[shift:]
        qd_real = qd_real[shift:]

    q_error = _rmse(q_sim - q_real)
    qd_error = _rmse(qd_sim - qd_real)
    kp_regularization = float(cfg.w_reg_kp) * float(params.kp) ** 2
    tau_regularization = float(cfg.w_reg_tau) * float(params.tau_limit) ** 2
    # 轻正则把 gravity_scale 拉向当前模型语义：开重力时拉向理论值 1.0；
    # 关重力时拉向 0.0，避免 no-gravity ablation 被一个无意义常数污染 loss。
    gravity_target = 1.0 if bool(cfg.gravity_compensation) else 0.0
    gravity_reg = float(cfg.w_reg_gravity_scale) * (float(params.gravity_scale) - gravity_target) ** 2
    return FitMetrics(
        loss=cfg.w_q * q_error + cfg.w_qd * qd_error + kp_regularization + tau_regularization + gravity_reg,
        q_rmse=q_error,
        qd_rmse=qd_error,
        kp_regularization=kp_regularization,
        tau_limit_regularization=tau_regularization,
        gravity_scale_regularization=gravity_reg,
    )


ProgressCallback = Callable[[int, int, FitResult], None]


def fit_joint_with_backend(
    dataset: SweepDataset,
    cfg: IdentificationConfig,
    backend: JointReplayBackend,
    progress_callback: ProgressCallback | None = None,
    progress_interval: int = 0,
) -> FitResult:
    dataset.validate()
    joint = dataset.active_joint
    q_real = dataset.q[:, joint]
    qd_real = np.gradient(q_real, dataset.t)
    rng = np.random.default_rng(cfg.random_seed)
    best: FitResult | None = None
    trace: list[dict[str, Any]] = []
    candidate_index = 0
    # 重力补偿关闭时退化为单一 gravity_scale=0.0(即纯 PD,旧行为)。
    gravity_scale_candidates = (
        list(cfg.gravity_scale_candidates) if cfg.gravity_compensation else [0.0]
    )
    total_candidates = (
        len(cfg.delay_steps_candidates)
        * len(cfg.filter_alpha_candidates)
        * len(cfg.time_shift_candidates)
        * len(cfg.tau_lag_alpha_candidates)
        * len(gravity_scale_candidates)
        * int(cfg.random_samples)
    )

    for delay_steps in cfg.delay_steps_candidates:
        for filter_alpha in cfg.filter_alpha_candidates:
            for time_shift_steps in cfg.time_shift_candidates:
                for tau_lag_alpha in cfg.tau_lag_alpha_candidates:
                    for gravity_scale in gravity_scale_candidates:
                        low_kp, high_kp = cfg.kp_bounds
                        low_kd, high_kd = cfg.kd_bounds
                        low_tau, high_tau = cfg.tau_limit_bounds
                        for _ in range(cfg.random_samples):
                            kp = _sample_uniform(rng, (low_kp, high_kp))
                            # kd/kp 比值约束与 Isaac 精修(candidates.py)用同一个 cfg.max_kd_over_kp，
                            # 保证两阶段在相同搜索空间里找谷，避免 MuJoCo best 在 Isaac 被拒。
                            kd_hi = min(high_kd, max(low_kd, kp * float(cfg.max_kd_over_kp)))
                            params = JointActuatorParams(
                                kp=kp,
                                kd=_sample_uniform(rng, (low_kd, kd_hi)),
                                delay_steps=int(delay_steps),
                                filter_alpha=float(filter_alpha),
                                tau_limit=_sample_uniform(rng, (low_tau, high_tau)),
                                tau_lag_alpha=float(tau_lag_alpha),
                                gravity_scale=float(gravity_scale),
                            )
                            replay = backend.replay(dataset, params)
                            metrics = compute_metrics(
                                replay.q,
                                replay.qd,
                                q_real,
                                qd_real,
                                params,
                                cfg,
                                time_shift_steps=int(time_shift_steps),
                            )
                            if best is None or metrics.loss < best.loss:
                                best = FitResult(
                                    params=params,
                                    loss=metrics.loss,
                                    q_rmse=metrics.q_rmse,
                                    qd_rmse=metrics.qd_rmse,
                                    time_shift_steps=int(time_shift_steps),
                                    kp_regularization=metrics.kp_regularization,
                                    tau_limit_regularization=metrics.tau_limit_regularization,
                                    gravity_scale_regularization=metrics.gravity_scale_regularization,
                                )
                            candidate_index += 1
                            trace.append(
                                {
                                    "candidate_index": candidate_index,
                                    "delay_steps": int(delay_steps),
                                    "filter_alpha": float(filter_alpha),
                                    "time_shift_steps": int(time_shift_steps),
                                    "tau_lag_alpha": float(tau_lag_alpha),
                                    "gravity_scale": float(gravity_scale),
                                    "kp": float(params.kp),
                                    "kd": float(params.kd),
                                    "tau_limit": float(params.tau_limit),
                                    "loss": float(metrics.loss),
                                    "q_rmse": float(metrics.q_rmse),
                                    "qd_rmse": float(metrics.qd_rmse),
                                    "kp_regularization": float(metrics.kp_regularization),
                                    "tau_limit_regularization": float(metrics.tau_limit_regularization),
                                    "gravity_scale_regularization": float(metrics.gravity_scale_regularization),
                                    "best_loss": float(best.loss),
                                }
                            )
                            if progress_callback is not None and best is not None and progress_interval > 0:
                                if candidate_index == 1 or candidate_index % progress_interval == 0 or candidate_index == total_candidates:
                                    progress_callback(candidate_index, total_candidates, best)

    assert best is not None
    best.trace = trace
    return best
