from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class JointActuatorParams:
    kp: float
    kd: float
    delay_steps: int
    filter_alpha: float
    tau_limit: float
    tau_lag_alpha: float = 1.0
    # 重力补偿前馈缩放。当前 ARX 流程默认纯 PD：真机固件已补重力，辨识侧不再叠加 +G。
    # 只有显式辨识会下垂的执行器时，才把该值设为正数。
    gravity_scale: float = 0.0

    def __post_init__(self) -> None:
        if self.kp <= 0.0:
            raise ValueError("kp must be positive")
        if self.kd < 0.0:
            raise ValueError("kd must be non-negative")
        if self.delay_steps < 0:
            raise ValueError("delay_steps must be non-negative")
        if not 0.0 < self.filter_alpha <= 1.0:
            raise ValueError("filter_alpha must be in (0, 1]")
        if self.tau_limit <= 0.0:
            raise ValueError("tau_limit must be positive")
        if not 0.0 < self.tau_lag_alpha <= 1.0:
            raise ValueError("tau_lag_alpha must be in (0, 1]")
        if self.gravity_scale < 0.0:
            raise ValueError("gravity_scale must be non-negative")


class DelayFilter:
    def __init__(self, size: int, delay_steps: int, alpha: float, initial: np.ndarray | None = None):
        if size <= 0:
            raise ValueError("size must be positive")
        if delay_steps < 0:
            raise ValueError("delay_steps must be non-negative")
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.size = size
        self.delay_steps = delay_steps
        self.alpha = alpha
        self.filtered = np.zeros(size, dtype=float) if initial is None else np.asarray(initial, dtype=float).copy()
        self.queue: list[np.ndarray] = [self.filtered.copy() for _ in range(delay_steps)]

    def reset(self, value: np.ndarray | None = None) -> None:
        self.filtered = np.zeros(self.size, dtype=float) if value is None else np.asarray(value, dtype=float).copy()
        self.queue = [self.filtered.copy() for _ in range(self.delay_steps)]

    def update(self, raw: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw, dtype=float)
        if raw.shape != (self.size,):
            raise ValueError(f"raw must have shape ({self.size},)")
        if self.delay_steps == 0:
            delayed = raw.copy()
        else:
            self.queue.append(raw.copy())
            delayed = self.queue.pop(0)
        self.filtered = self.alpha * delayed + (1.0 - self.alpha) * self.filtered
        return self.filtered.copy()


class ScalarLagFilter:
    def __init__(self, alpha: float):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self.filtered = 0.0
        self.initialized = False

    def reset(self, value: float | None = None) -> None:
        if value is None:
            self.filtered = 0.0
            self.initialized = False
            return
        self.filtered = float(value)
        self.initialized = True

    def update(self, raw: float) -> float:
        raw = float(raw)
        if not self.initialized:
            self.filtered = raw
            self.initialized = True
        else:
            self.filtered = self.alpha * raw + (1.0 - self.alpha) * self.filtered
        return float(self.filtered)


class SimplePdActuator:
    def __init__(self, params: list[JointActuatorParams]):
        if not params:
            raise ValueError("params must not be empty")
        self.params = params
        self.size = len(params)
        max_delay = max(param.delay_steps for param in params)
        self.filters = [
            DelayFilter(size=1, delay_steps=param.delay_steps, alpha=param.filter_alpha)
            for param in params
        ]
        self.torque_filters = [
            ScalarLagFilter(alpha=param.tau_lag_alpha)
            for param in params
        ]
        self.max_delay = max_delay

    def reset(self, q_des_raw: np.ndarray | None = None) -> None:
        if q_des_raw is None:
            for filt in self.filters:
                filt.reset()
            for filt in self.torque_filters:
                filt.reset(0.0)
            return
        q_des_raw = np.asarray(q_des_raw, dtype=float)
        if q_des_raw.shape != (self.size,):
            raise ValueError(f"q_des_raw must have shape ({self.size},)")
        for idx, filt in enumerate(self.filters):
            filt.reset(np.array([q_des_raw[idx]], dtype=float))
        for filt in self.torque_filters:
            filt.reset(0.0)

    def filter_command(self, q_des_raw: np.ndarray) -> np.ndarray:
        q_des_raw = np.asarray(q_des_raw, dtype=float)
        if q_des_raw.shape != (self.size,):
            raise ValueError(f"q_des_raw must have shape ({self.size},)")
        return np.array([self.filters[idx].update(np.array([q_des_raw[idx]]))[0] for idx in range(self.size)])

    def step(self, q_des_raw: np.ndarray, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        # 无前馈版本：等价于 tau_ff=0，保持旧行为/调用方兼容。
        return self.step_with_feedforward(q_des_raw, q, qd, np.zeros(self.size, dtype=float))

    def step_with_feedforward(
        self,
        q_des_raw: np.ndarray,
        q: np.ndarray,
        qd: np.ndarray,
        tau_ff: np.ndarray,
    ) -> np.ndarray:
        # 力矩合成顺序(关键)：
        #   tau_raw = kp*(q_des - q) - kd*qd + tau_ff   # 前馈(重力)在 clip 之前
        #   tau     = torque_lag(clip(tau_raw, ±tau_limit))
        # 重力项必须进 clip 之前：真实电机力矩上限限制的是“总输出力矩”，
        # 而非仅 PD 部分；若把重力放到 clip 之后会人为突破力矩限幅。
        q = np.asarray(q, dtype=float)
        qd = np.asarray(qd, dtype=float)
        tau_ff = np.asarray(tau_ff, dtype=float)
        if q.shape != (self.size,) or qd.shape != (self.size,):
            raise ValueError(f"q and qd must have shape ({self.size},)")
        if tau_ff.shape != (self.size,):
            raise ValueError(f"tau_ff must have shape ({self.size},)")
        q_des = self.filter_command(q_des_raw)
        kp = np.array([param.kp for param in self.params], dtype=float)
        kd = np.array([param.kd for param in self.params], dtype=float)
        limit = np.array([param.tau_limit for param in self.params], dtype=float)
        tau = kp * (q_des - q) - kd * qd + tau_ff
        tau = np.clip(tau, -limit, limit)
        return np.array(
            [self.torque_filters[idx].update(float(tau[idx])) for idx in range(self.size)],
            dtype=float,
        )
