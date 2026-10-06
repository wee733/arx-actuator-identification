from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from arx_actuator_identification.actuator import JointActuatorParams


@dataclass(frozen=True)
class CandidateRow:
    kp: float
    kd: float
    delay_steps: int
    filter_alpha: float
    tau_limit: float
    tau_lag_alpha: float = 1.0
    gravity_scale: float = 0.0
    time_shift_steps: int = 0
    source: str = "generated"
    prior_loss: float | None = None
    prior_q_rmse: float | None = None
    prior_qd_rmse: float | None = None

    @property
    def params(self) -> JointActuatorParams:
        return JointActuatorParams(
            kp=float(self.kp),
            kd=float(self.kd),
            delay_steps=int(self.delay_steps),
            filter_alpha=float(self.filter_alpha),
            tau_limit=float(self.tau_limit),
            tau_lag_alpha=float(self.tau_lag_alpha),
            gravity_scale=float(self.gravity_scale),
        )

    def key(self, ndigits: int = 9) -> tuple[Any, ...]:
        return (
            round(float(self.kp), ndigits),
            round(float(self.kd), ndigits),
            int(self.delay_steps),
            round(float(self.filter_alpha), ndigits),
            round(float(self.tau_limit), ndigits),
            round(float(self.tau_lag_alpha), ndigits),
            round(float(self.gravity_scale), ndigits),
            int(self.time_shift_steps),
        )


@dataclass(frozen=True)
class MujocoJointPriors:
    joint_number: int
    best: CandidateRow
    trace: list[CandidateRow]
    report_path: str
    trace_path: str | None


@dataclass(frozen=True)
class CandidateGenerationConfig:
    candidates_per_joint: int = 196_608
    random_seed: int = 7
    exploration_fraction: float = 0.08
    kp_multiplier_bounds: tuple[float, float] = (0.5, 1.8)
    kd_multiplier_bounds: tuple[float, float] = (0.4, 2.2)
    tau_limit_multiplier_bounds: tuple[float, float] = (0.6, 2.0)
    delay_radius: int = 2
    filter_alpha_choices: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0)
    tau_lag_alpha_choices: tuple[float, ...] = (0.1, 0.2, 0.4, 0.7, 1.0)
    time_shift_radius: int = 2

    def shrunk(self, factor: float) -> "CandidateGenerationConfig":
        # coarse-to-fine：把搜索半径朝中心收缩 factor 倍(factor∈(0,1])。
        # 乘性半径(kp/kd/tau)朝 1.0 收缩：例如 (0.5,1.8) 在 factor=0.5 时 -> (0.707,1.342)。
        # 加性半径(delay/time_shift)朝 0 收缩并向上取整,但 round 后至少保留原非零半径的 1。
        # factor=1.0 时原样返回(等价不收缩)。被锁死的离散维度(filter/tau_lag choices)不动。
        f = float(factor)
        if not 0.0 < f <= 1.0:
            raise ValueError("shrink factor must be in (0, 1]")

        def shrink_mult(bounds: tuple[float, float]) -> tuple[float, float]:
            lo, hi = float(bounds[0]), float(bounds[1])
            return (1.0 - (1.0 - lo) * f, 1.0 + (hi - 1.0) * f)

        def shrink_radius(radius: int) -> int:
            r = int(radius)
            if r <= 0:
                return 0
            return max(1, int(round(r * f)))

        return replace(
            self,
            kp_multiplier_bounds=shrink_mult(self.kp_multiplier_bounds),
            kd_multiplier_bounds=shrink_mult(self.kd_multiplier_bounds),
            tau_limit_multiplier_bounds=shrink_mult(self.tau_limit_multiplier_bounds),
            delay_radius=shrink_radius(self.delay_radius),
            time_shift_radius=shrink_radius(self.time_shift_radius),
        )


@dataclass(frozen=True)
class CandidateFilterConfig:
    kp_bounds: tuple[float, float] = (0.5, 200.0)
    kd_bounds: tuple[float, float] = (0.0, 20.0)
    tau_limit_bounds: tuple[float, float] = (0.3, 64.0)
    delay_steps_bounds: tuple[int, int] = (0, 30)
    max_kd_over_kp: float = 0.25


@dataclass(frozen=True)
class CandidateGenerationResult:
    candidates: list[CandidateRow]
    requested_count: int
    rejected_count: int
    duplicate_count: int
    source_count: int


def read_mujoco_joint_priors(
    report_path: str | Path,
    trace_dir: str | Path | None,
    joint_number: int,
    prior_top_k: int = 512,
) -> MujocoJointPriors:
    report_path = Path(report_path)
    with report_path.open("r", encoding="utf-8") as handle:
        report = yaml.safe_load(handle) or {}
    if not isinstance(report, dict):
        raise ValueError(f"{report_path} must contain a YAML mapping")

    best_record = _find_report_joint(report, joint_number)
    best = _candidate_from_mapping(best_record, source="mujoco_best")

    trace_path: Path | None = None
    trace_rows: list[CandidateRow] = []
    if trace_dir is not None:
        trace_path = Path(trace_dir) / f"joint_{joint_number}_fit_trace.csv"
        if trace_path.exists():
            trace_rows = _read_trace_rows(trace_path, prior_top_k=prior_top_k)

    return MujocoJointPriors(
        joint_number=int(joint_number),
        best=best,
        trace=trace_rows,
        report_path=str(report_path),
        trace_path=str(trace_path) if trace_path is not None else None,
    )


def generate_candidates_for_joint(
    priors: MujocoJointPriors,
    generation: CandidateGenerationConfig | None = None,
    filtering: CandidateFilterConfig | None = None,
) -> CandidateGenerationResult:
    generation = CandidateGenerationConfig() if generation is None else generation
    filtering = CandidateFilterConfig() if filtering is None else filtering
    if generation.candidates_per_joint <= 0:
        raise ValueError("candidates_per_joint must be positive")
    if not 0.0 <= generation.exploration_fraction < 1.0:
        raise ValueError("exploration_fraction must be in [0, 1)")

    rng = np.random.default_rng(int(generation.random_seed) + int(priors.joint_number) - 1)
    seeds = _dedupe([priors.best, *priors.trace])
    accepted: list[CandidateRow] = []
    seen: set[tuple[Any, ...]] = set()
    rejected_count = 0
    duplicate_count = 0

    def add_candidate(row: CandidateRow) -> None:
        nonlocal duplicate_count, rejected_count
        if not is_candidate_reasonable(row, filtering):
            rejected_count += 1
            return
        key = row.key()
        if key in seen:
            duplicate_count += 1
            return
        accepted.append(row)
        seen.add(key)

    for seed in seeds:
        add_candidate(seed)
        if len(accepted) >= generation.candidates_per_joint:
            break

    max_attempts = max(10_000, generation.candidates_per_joint * 100)
    attempts = 0
    while len(accepted) < generation.candidates_per_joint and attempts < max_attempts:
        attempts += 1
        if rng.random() < generation.exploration_fraction:
            row = _sample_exploration_candidate(rng, priors.best, filtering, generation)
        else:
            seed = seeds[int(rng.integers(0, len(seeds)))]
            row = _sample_local_candidate(rng, seed, generation)
        add_candidate(row)

    if len(accepted) < generation.candidates_per_joint:
        raise RuntimeError(
            f"only generated {len(accepted)} candidates after {attempts} attempts; "
            "relax filters or reduce candidates_per_joint"
        )

    return CandidateGenerationResult(
        candidates=accepted[: generation.candidates_per_joint],
        requested_count=int(generation.candidates_per_joint),
        rejected_count=int(rejected_count),
        duplicate_count=int(duplicate_count),
        source_count=len(seeds),
    )


def is_candidate_reasonable(row: CandidateRow, cfg: CandidateFilterConfig | None = None) -> bool:
    cfg = CandidateFilterConfig() if cfg is None else cfg
    if row.kp <= 0.0 or row.kd < 0.0 or row.tau_limit <= 0.0:
        return False
    if row.delay_steps < 0:
        return False
    if not 0.0 < row.filter_alpha <= 1.0:
        return False
    if not 0.0 < row.tau_lag_alpha <= 1.0:
        return False
    if not _in_bounds(row.kp, cfg.kp_bounds):
        return False
    if not _in_bounds(row.kd, cfg.kd_bounds):
        return False
    if not _in_bounds(row.tau_limit, cfg.tau_limit_bounds):
        return False
    delay_lo, delay_hi = cfg.delay_steps_bounds
    if not int(delay_lo) <= int(row.delay_steps) <= int(delay_hi):
        return False
    if row.kd / max(row.kp, 1e-9) > cfg.max_kd_over_kp:
        return False
    return True


def candidates_to_params(candidates: list[CandidateRow]) -> list[JointActuatorParams]:
    return [candidate.params for candidate in candidates]


def candidate_to_record(candidate: CandidateRow) -> dict[str, Any]:
    return {
        "kp": float(candidate.kp),
        "kd": float(candidate.kd),
        "delay_steps": int(candidate.delay_steps),
        "filter_alpha": float(candidate.filter_alpha),
        "time_shift_steps": int(candidate.time_shift_steps),
        "tau_limit": float(candidate.tau_limit),
        "tau_lag_alpha": float(candidate.tau_lag_alpha),
        "gravity_scale": float(candidate.gravity_scale),
        "source": candidate.source,
        "prior_loss": None if candidate.prior_loss is None else float(candidate.prior_loss),
        "prior_q_rmse": None if candidate.prior_q_rmse is None else float(candidate.prior_q_rmse),
        "prior_qd_rmse": None if candidate.prior_qd_rmse is None else float(candidate.prior_qd_rmse),
    }


def _find_report_joint(report: dict[str, Any], joint_number: int) -> dict[str, Any]:
    joints = report.get("joints")
    if not isinstance(joints, list):
        raise ValueError("fit report must contain a joints list")
    for row in joints:
        if isinstance(row, dict) and int(row.get("joint", -1)) == int(joint_number):
            return row
    raise ValueError(f"fit report does not contain joint {joint_number}")


def _read_trace_rows(trace_path: Path, prior_top_k: int) -> list[CandidateRow]:
    if prior_top_k <= 0:
        return []
    rows: list[CandidateRow] = []
    with trace_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            rows.append(_candidate_from_mapping(raw, source="mujoco_trace"))
    rows.sort(key=lambda row: float("inf") if row.prior_loss is None else row.prior_loss)
    return _dedupe(rows)[:prior_top_k]


def _candidate_from_mapping(row: dict[str, Any], source: str) -> CandidateRow:
    return CandidateRow(
        kp=_float(row, "kp"),
        kd=_float(row, "kd"),
        delay_steps=_int(row, "delay_steps"),
        filter_alpha=_float(row, "filter_alpha"),
        tau_limit=_float(row, "tau_limit"),
        tau_lag_alpha=_float(row, "tau_lag_alpha", default=1.0),
        gravity_scale=_float(row, "gravity_scale", default=0.0),
        time_shift_steps=_int(row, "time_shift_steps", default=0),
        source=source,
        prior_loss=_optional_float(row, "loss"),
        prior_q_rmse=_optional_float(row, "q_rmse"),
        prior_qd_rmse=_optional_float(row, "qd_rmse"),
    )


def _sample_local_candidate(
    rng: np.random.Generator,
    seed: CandidateRow,
    cfg: CandidateGenerationConfig,
) -> CandidateRow:
    kp = _multiplicative_sample(rng, seed.kp, cfg.kp_multiplier_bounds)
    kd_seed = max(float(seed.kd), 1e-6)
    kd = _multiplicative_sample(rng, kd_seed, cfg.kd_multiplier_bounds)
    if seed.kd <= 1e-9 and rng.random() < 0.3:
        kd = 0.0
    tau_limit = _multiplicative_sample(rng, seed.tau_limit, cfg.tau_limit_multiplier_bounds)

    delay_lo = max(0, int(seed.delay_steps) - int(cfg.delay_radius))
    delay_hi = int(seed.delay_steps) + int(cfg.delay_radius)
    delay_steps = int(rng.integers(delay_lo, delay_hi + 1))
    filter_alpha = _choose_near(rng, cfg.filter_alpha_choices, seed.filter_alpha)
    tau_lag_alpha = _choose_near(rng, cfg.tau_lag_alpha_choices, seed.tau_lag_alpha)
    time_shift_lo = max(0, int(seed.time_shift_steps) - int(cfg.time_shift_radius))
    time_shift_hi = int(seed.time_shift_steps) + int(cfg.time_shift_radius)
    time_shift_steps = int(rng.integers(time_shift_lo, time_shift_hi + 1))

    return CandidateRow(
        kp=kp,
        kd=kd,
        delay_steps=delay_steps,
        filter_alpha=filter_alpha,
        tau_limit=tau_limit,
        tau_lag_alpha=tau_lag_alpha,
        gravity_scale=float(seed.gravity_scale),
        time_shift_steps=time_shift_steps,
        source="local_sample",
        prior_loss=seed.prior_loss,
        prior_q_rmse=seed.prior_q_rmse,
        prior_qd_rmse=seed.prior_qd_rmse,
    )


def _sample_exploration_candidate(
    rng: np.random.Generator,
    seed: CandidateRow,
    filter_cfg: CandidateFilterConfig,
    generation_cfg: CandidateGenerationConfig,
) -> CandidateRow:
    kp = float(rng.uniform(*filter_cfg.kp_bounds))
    kd_hi = min(float(filter_cfg.kd_bounds[1]), max(float(filter_cfg.kd_bounds[0]), kp * filter_cfg.max_kd_over_kp))
    kd = float(rng.uniform(float(filter_cfg.kd_bounds[0]), kd_hi))
    tau_limit = float(rng.uniform(*filter_cfg.tau_limit_bounds))
    delay_steps = int(rng.integers(int(filter_cfg.delay_steps_bounds[0]), int(filter_cfg.delay_steps_bounds[1]) + 1))
    time_shift_radius = int(generation_cfg.time_shift_radius)
    return CandidateRow(
        kp=kp,
        kd=kd,
        delay_steps=delay_steps,
        filter_alpha=float(rng.choice(generation_cfg.filter_alpha_choices)),
        tau_limit=tau_limit,
        tau_lag_alpha=float(rng.choice(generation_cfg.tau_lag_alpha_choices)),
        gravity_scale=float(seed.gravity_scale),
        time_shift_steps=max(
            0,
            int(seed.time_shift_steps) + int(rng.integers(-time_shift_radius, time_shift_radius + 1)),
        ),
        source="exploration_sample",
    )


def _dedupe(rows: list[CandidateRow]) -> list[CandidateRow]:
    out: list[CandidateRow] = []
    seen: set[tuple[Any, ...]] = set()
    for row in rows:
        key = row.key()
        if key in seen:
            continue
        out.append(row)
        seen.add(key)
    return out


def _multiplicative_sample(
    rng: np.random.Generator,
    center: float,
    multiplier_bounds: tuple[float, float],
) -> float:
    lo, hi = multiplier_bounds
    if lo <= 0.0 or hi < lo:
        raise ValueError("multiplier bounds must be positive and non-decreasing")
    if hi == lo:
        return float(center * lo)
    return float(center * np.exp(rng.uniform(np.log(lo), np.log(hi))))


def _choose_near(rng: np.random.Generator, choices: tuple[float, ...], center: float) -> float:
    values = np.asarray(choices, dtype=float)
    if values.size == 0:
        raise ValueError("choices must not be empty")
    distances = np.abs(values - float(center))
    weights = 1.0 / (distances + 0.05)
    weights = weights / np.sum(weights)
    return float(rng.choice(values, p=weights))


def _in_bounds(value: float, bounds: tuple[float, float]) -> bool:
    return float(bounds[0]) <= float(value) <= float(bounds[1])


def _float(row: dict[str, Any], key: str, default: float | None = None) -> float:
    value = row.get(key, default)
    if value is None or value == "":
        raise ValueError(f"missing required field {key}")
    return float(value)


def _optional_float(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value is None or value == "":
        return None
    return float(value)


def _int(row: dict[str, Any], key: str, default: int | None = None) -> int:
    value = row.get(key, default)
    if value is None or value == "":
        raise ValueError(f"missing required field {key}")
    return int(float(value))
