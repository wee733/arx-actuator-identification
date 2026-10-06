from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class IdentificationConfig:
    delay_steps_candidates: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])
    filter_alpha_candidates: list[float] = field(default_factory=lambda: [1.0])
    time_shift_candidates: list[int] = field(default_factory=lambda: [0])
    tau_lag_alpha_candidates: list[float] = field(default_factory=lambda: [1.0])
    kp_bounds: tuple[float, float] = (0.1, 500.0)
    kd_bounds: tuple[float, float] = (0.0, 50.0)
    tau_limit_bounds: tuple[float, float] = (0.1, 200.0)
    # kd/kp 比值上界(物理过阻尼先验)。MuJoCo 粗拟合与 Isaac 精修必须用同一个值，
    # 否则两阶段在不同搜索空间里找谷——MuJoCo 拟出的 best 可能在 Isaac 第 0 轮被拒，
    # 导致系统性低估 kd。设得过紧会剪掉真实 kd，过松失去过阻尼保护。
    max_kd_over_kp: float = 0.25
    w_q: float = 1.0
    w_qd: float = 1.0
    w_reg_kp: float = 0.0
    w_reg_tau: float = 0.0
    # 重力补偿辨识开关。默认 False(纯 PD)：实测(2026-06-05)真机 ARX 固件已内置重力补偿，
    # 采集的 q_real 是重力中性轨迹，Isaac/MuJoCo 都用纯 PD 拟合更优(j2/j3/j4 好 6~17 倍)。
    # 仅当辨识一个真正会因重力下垂的执行器时才设 True 并放开 gravity_scale_candidates。
    gravity_compensation: bool = False
    gravity_scale_candidates: list[float] = field(default_factory=lambda: [0.0])
    w_reg_gravity_scale: float = 0.01
    random_seed: int = 7
    random_samples: int = 120
    # Isaac coarse-to-fine 迭代精修：每轮用上轮 best 当中心、按 refine_shrink 收缩搜索半径。
    # refinement_rounds=1 退化为旧的单轮随机搜索。refine_candidates_per_round=None 时
    # 每轮候选数 = candidates_per_joint // refinement_rounds，保证总 GPU 预算不变。
    refinement_rounds: int = 4
    refine_shrink: float = 0.5
    refine_candidates_per_round: int | None = None

    def __post_init__(self) -> None:
        if self.refinement_rounds < 1:
            raise ValueError("refinement_rounds must be >= 1")
        if self.max_kd_over_kp <= 0.0:
            raise ValueError("max_kd_over_kp must be positive")
        if not 0.0 < self.refine_shrink <= 1.0:
            raise ValueError("refine_shrink must be in (0, 1]")
        if self.refine_candidates_per_round is not None and self.refine_candidates_per_round <= 0:
            raise ValueError("refine_candidates_per_round must be positive when set")


def _load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def load_identification_config(path: str | Path) -> IdentificationConfig:
    data = _load_yaml_mapping(path)
    return IdentificationConfig(**data)
