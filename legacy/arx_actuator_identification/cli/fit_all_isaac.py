from __future__ import annotations

import argparse
import atexit
import csv
from dataclasses import dataclass, replace
import os
from pathlib import Path
import time
from typing import Any, Callable, Sequence

import numpy as np
import yaml

from arx_actuator_identification.actuator import JointActuatorParams
from arx_actuator_identification.backends.isaac import IsaacBatchReplayBackend, IsaacReplayConfig
from arx_actuator_identification.cli.common import infer_metadata_path
from arx_actuator_identification.config import IdentificationConfig, load_identification_config
from arx_actuator_identification.dataset import load_sweep_dataset
from arx_actuator_identification.export import JOINT_NAMES, export_actuator_yaml
from arx_actuator_identification.fitting.batch_metrics import BatchMetricResult, compute_batch_metrics_np
from arx_actuator_identification.fitting.candidates import (
    CandidateFilterConfig,
    CandidateGenerationConfig,
    CandidateGenerationResult,
    CandidateRow,
    candidate_to_record,
    generate_candidates_for_joint,
    read_mujoco_joint_priors,
)


BackendFactory = Callable[[IsaacReplayConfig], Any]


def _bounds_locked(bounds: tuple[float, float]) -> bool:
    return float(bounds[0]) == float(bounds[1])


def _locked_value(candidates: Sequence[float] | None) -> float | None:
    # 候选集若退化为单一取值，返回该值(表示“被锁死”)，否则返回 None。
    # 用于把 config 里 time_shift_candidates=[0] 这种固定意图传递给 Isaac 候选。
    if not candidates:
        return None
    unique = {float(value) for value in candidates}
    if len(unique) == 1:
        return float(next(iter(unique)))
    return None


@dataclass(frozen=True)
class IsaacJointFitOutput:
    joint_number: int
    params: JointActuatorParams
    record: dict[str, Any]
    trace: list[dict[str, Any]]


@dataclass(frozen=True)
class _AggCounts:
    # 跨迭代轮聚合的候选计数，供 _result_record 复用(它读 .requested_count 等属性)。
    requested_count: int
    rejected_count: int
    duplicate_count: int
    source_count: int


def _joint_npz_path(data_dir: Path, joint_number: int) -> Path:
    return data_dir / f"joint_{joint_number}.npz"


def _joint_metadata_path(npz_path: Path) -> Path:
    return Path(infer_metadata_path(str(npz_path)))


def _require_joint_files(data_dir: Path, joints: Sequence[int]) -> None:
    missing: list[str] = []
    for joint_number in joints:
        npz_path = _joint_npz_path(data_dir, joint_number)
        metadata_path = _joint_metadata_path(npz_path)
        if not npz_path.exists():
            missing.append(str(npz_path))
        if not metadata_path.exists():
            missing.append(str(metadata_path))
    if missing:
        joined = "\n  ".join(missing)
        raise FileNotFoundError(f"missing required sweep files:\n  {joined}")


def _warn_cmd_source_mismatch(data_dir: Path, joints: Sequence[int], expected: str) -> None:
    # 防 footgun：若 NPZ metadata 标了 cmd_source，且与 --cmd-source 不符则报错，
    # 避免把 raw 数据导出成 q_cmd_smooth 契约。旧数据(无该键)不阻断，仅静默放行。
    import yaml as _yaml
    for joint_number in joints:
        meta_path = _joint_metadata_path(_joint_npz_path(data_dir, joint_number))
        try:
            meta = _yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
        except OSError:
            continue
        actual = meta.get("cmd_source")
        if actual is not None and str(actual) != str(expected):
            raise ValueError(
                f"joint_{joint_number} 数据的 cmd_source='{actual}' 与 --cmd-source='{expected}' 不符；"
                f"导出契约会撒谎。请改 --cmd-source {actual} 或换用对应数据目录。"
            )


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(report, handle, sort_keys=False, allow_unicode=True)


def _write_trace_csv(path: Path, trace: list[dict[str, Any]]) -> None:
    if not trace:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(trace[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trace)


def _trace_float(row: dict[str, Any], key: str, default: float | None = None) -> float:
    value = row.get(key, default)
    if value is None or value == "":
        raise ValueError(f"trace row missing required field {key}")
    return float(value)


def _trace_int(row: dict[str, Any], key: str, default: int | None = None) -> int:
    value = row.get(key, default)
    if value is None or value == "":
        raise ValueError(f"trace row missing required field {key}")
    return int(float(value))


def _trace_params(row: dict[str, Any]) -> JointActuatorParams:
    return JointActuatorParams(
        kp=_trace_float(row, "kp"),
        kd=_trace_float(row, "kd"),
        delay_steps=_trace_int(row, "delay_steps"),
        filter_alpha=_trace_float(row, "filter_alpha"),
        tau_limit=_trace_float(row, "tau_limit"),
        tau_lag_alpha=_trace_float(row, "tau_lag_alpha", 1.0),
        gravity_scale=_trace_float(row, "gravity_scale", 0.0),
    )


def _load_existing_isaac_output(
    joint_number: int,
    trace_dir: Path,
    trace_top_n: int,
) -> IsaacJointFitOutput:
    trace_path = trace_dir / f"joint_{joint_number}_isaac_top_trace.csv"
    if not trace_path.exists():
        raise FileNotFoundError(
            f"--start-joint {joint_number + 1} needs existing Isaac trace for joint {joint_number}: {trace_path}"
        )
    rows: list[dict[str, Any]] = []
    with trace_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            rows.append(dict(raw))
    if not rows:
        raise ValueError(f"existing Isaac trace is empty: {trace_path}")
    rows.sort(key=lambda row: _trace_float(row, "loss"))
    best = rows[0]
    params = _trace_params(best)
    evaluated = _trace_int(best, "candidate_index", 0)
    record = {
        "joint": int(joint_number),
        "joint_name": JOINT_NAMES[joint_number - 1],
        "kp": float(params.kp),
        "kd": float(params.kd),
        "delay_steps": int(params.delay_steps),
        "filter_alpha": float(params.filter_alpha),
        "time_shift_steps": _trace_int(best, "time_shift_steps", 0),
        "tau_limit": float(params.tau_limit),
        "tau_lag_alpha": float(params.tau_lag_alpha),
        "gravity_scale": float(params.gravity_scale),
        "loss": _trace_float(best, "loss"),
        "q_rmse": _trace_float(best, "q_rmse"),
        "qd_rmse": _trace_float(best, "qd_rmse"),
        "kp_regularization": _trace_float(best, "kp_regularization", 0.0),
        "tau_limit_regularization": _trace_float(best, "tau_limit_regularization", 0.0),
        "gravity_scale_regularization": _trace_float(best, "gravity_scale_regularization", 0.0),
        "candidate_source": str(best.get("source", "isaac_trace")),
        "requested_candidates": int(evaluated),
        "evaluated_candidates": int(evaluated),
        "rejected_candidates": 0,
        "duplicate_candidates": 0,
        "prior_source_count": 0,
        "trace_csv": str(trace_path),
        "status": "resumed_from_isaac_trace",
    }
    print(
        f"[fit-isaac] joint {joint_number}/6 resumed from {trace_path} "
        f"best_loss={record['loss']:.6f}",
        flush=True,
    )
    return IsaacJointFitOutput(
        joint_number=joint_number,
        params=params,
        record=record,
        trace=rows[: max(1, int(trace_top_n))],
    )


def _result_record(
    joint_number: int,
    result_candidate: CandidateRow,
    metrics: BatchMetricResult,
    metric_index: int,
    generated: CandidateGenerationResult | _AggCounts,
    evaluated_candidates: int,
    trace_path: Path | None,
) -> dict[str, Any]:
    return {
        "joint": int(joint_number),
        "joint_name": JOINT_NAMES[joint_number - 1],
        "kp": float(result_candidate.kp),
        "kd": float(result_candidate.kd),
        "delay_steps": int(result_candidate.delay_steps),
        "filter_alpha": float(result_candidate.filter_alpha),
        "time_shift_steps": int(result_candidate.time_shift_steps),
        "tau_limit": float(result_candidate.tau_limit),
        "tau_lag_alpha": float(result_candidate.tau_lag_alpha),
        "gravity_scale": float(result_candidate.gravity_scale),
        "loss": float(metrics.loss[metric_index]),
        "q_rmse": float(metrics.q_rmse[metric_index]),
        "qd_rmse": float(metrics.qd_rmse[metric_index]),
        "kp_regularization": float(metrics.kp_regularization[metric_index]),
        "tau_limit_regularization": float(metrics.tau_limit_regularization[metric_index]),
        "gravity_scale_regularization": float(metrics.gravity_scale_regularization[metric_index]),
        "candidate_source": result_candidate.source,
        "requested_candidates": int(generated.requested_count),
        "evaluated_candidates": int(evaluated_candidates),
        "rejected_candidates": int(generated.rejected_count),
        "duplicate_candidates": int(generated.duplicate_count),
        "prior_source_count": int(generated.source_count),
        "trace_csv": str(trace_path) if trace_path is not None else None,
    }


def _evaluate_candidates(
    backend,
    dataset,
    candidates: list[CandidateRow],
    num_envs: int,
    identify_cfg: IdentificationConfig,
    q_real: np.ndarray,
    qd_real: np.ndarray,
    trace_top_n: int,
    candidate_offset: int,
    joint_number: int,
    round_label: str,
) -> tuple[CandidateRow | None, BatchMetricResult | None, int, list[dict[str, Any]], int]:
    # 单轮评估：把一批候选分成 num_envs 大小的 chunk 在 GPU 上 batch replay，
    # 返回(本轮 best 候选, 其所在 chunk 的 metrics, chunk 内索引, top_trace, 评估数)。
    # 无副作用，供迭代精修每轮重复调用。
    best_candidate: CandidateRow | None = None
    best_metrics: BatchMetricResult | None = None
    best_metric_index = 0
    top_trace: list[dict[str, Any]] = []
    evaluated = 0
    total = len(candidates)
    chunk_count = (total + num_envs - 1) // num_envs
    for start in range(0, total, num_envs):
        chunk = candidates[start : start + num_envs]
        params = [candidate.params for candidate in chunk]
        chunk_number = start // num_envs + 1
        chunk_t0 = time.perf_counter()
        print(
            f"[fit-isaac] joint {joint_number}/6 {round_label} "
            f"chunk {chunk_number}/{chunk_count} start size={len(chunk)}",
            flush=True,
        )
        replay = backend.replay_batch(dataset, params)
        metrics = compute_batch_metrics_np(
            q_sim=replay.q,
            qd_sim=replay.qd,
            q_real=q_real,
            qd_real=qd_real,
            params=params,
            cfg=identify_cfg,
            time_shift_steps=np.asarray([candidate.time_shift_steps for candidate in chunk], dtype=np.int64),
        )
        for local_index, candidate in enumerate(chunk):
            row = {
                "candidate_index": candidate_offset + evaluated + local_index + 1,
                **candidate_to_record(candidate),
                "loss": float(metrics.loss[local_index]),
                "q_rmse": float(metrics.q_rmse[local_index]),
                "qd_rmse": float(metrics.qd_rmse[local_index]),
                "kp_regularization": float(metrics.kp_regularization[local_index]),
                "tau_limit_regularization": float(metrics.tau_limit_regularization[local_index]),
                "gravity_scale_regularization": float(metrics.gravity_scale_regularization[local_index]),
            }
            _push_top_trace(top_trace, row, trace_top_n)
        local_best_index = metrics.best_index
        if best_metrics is None or metrics.loss[local_best_index] < best_metrics.loss[best_metric_index]:
            best_candidate = chunk[local_best_index]
            best_metrics = metrics
            best_metric_index = local_best_index
        evaluated += len(chunk)
        if best_metrics is not None:
            chunk_s = time.perf_counter() - chunk_t0
            print(
                f"[fit-isaac] joint {joint_number}/6 {round_label} candidates {evaluated}/{total} "
                f"best_loss={best_metrics.loss[best_metric_index]:.6f} chunk_s={chunk_s:.1f}",
                flush=True,
            )
    return best_candidate, best_metrics, best_metric_index, top_trace, evaluated


def _fit_joint_isaac(
    joint_number: int,
    data_dir: Path,
    mujoco_report: Path,
    mujoco_trace_dir: Path | None,
    identify_cfg: IdentificationConfig,
    generation_cfg: CandidateGenerationConfig,
    filter_cfg: CandidateFilterConfig,
    backend,
    num_envs: int,
    prior_top_k: int,
    trace_top_n: int,
    trace_dir: Path | None,
) -> IsaacJointFitOutput:
    joint_t0 = time.perf_counter()
    npz_path = _joint_npz_path(data_dir, joint_number)
    metadata_path = _joint_metadata_path(npz_path)
    print(
        f"[fit-isaac] joint {joint_number}/6 preparing dataset={npz_path} metadata={metadata_path}",
        flush=True,
    )
    dataset = load_sweep_dataset(npz_path, metadata_path)
    if dataset.active_joint != joint_number - 1:
        raise ValueError(f"{npz_path} contains active_joint={dataset.active_joint}, expected {joint_number - 1}")

    priors = read_mujoco_joint_priors(
        mujoco_report,
        mujoco_trace_dir,
        joint_number=joint_number,
        prior_top_k=prior_top_k,
    )
    print(
        f"[fit-isaac] joint {joint_number}/6 loaded priors "
        f"best_loss={priors.best.prior_loss} trace_rows={len(priors.trace)}",
        flush=True,
    )
    q_real = dataset.q[:, joint_number - 1]
    qd_real = np.gradient(q_real, dataset.t)

    # time_shift 只是 loss 对齐旋钮，不会导出到 actuator YAML。当 config 把它锁成单一值
    # (本流程固定 [0]) 时，强制覆写所有候选——否则 MuJoCo prior 里若带非零 time_shift，
    # Isaac 会沿用它，靠一个不可部署的时钟偏移偷偷压低 loss。time_shift_radius=0 只保证
    # “不再扰动”，并不保证“归零”，故每轮生成后都显式 clamp。
    forced_time_shift = _locked_value(identify_cfg.time_shift_candidates)
    # gravity_scale：Isaac 不消费它(纯 PD)，但 MuJoCo prior 可能带 gravity_scale=1.0(旧/有重力
    # 数据)，会进 report/trace 且被 batch_metrics 的 gravity 正则按 cfg.gravity_target 惩罚，污染
    # loss。故按 config 强制覆写:关重力→0.0；开重力且锁单值→该值；开重力且多候选→None(允许辨识)。
    if not identify_cfg.gravity_compensation:
        forced_gravity_scale: float | None = 0.0
    else:
        forced_gravity_scale = _locked_value(identify_cfg.gravity_scale_candidates)

    def _clamp(result: CandidateGenerationResult) -> CandidateGenerationResult:
        if forced_time_shift is None and forced_gravity_scale is None:
            return result
        new_candidates = []
        for candidate in result.candidates:
            changes: dict[str, Any] = {}
            if forced_time_shift is not None:
                changes["time_shift_steps"] = int(forced_time_shift)
            if forced_gravity_scale is not None:
                changes["gravity_scale"] = float(forced_gravity_scale)
            new_candidates.append(replace(candidate, **changes) if changes else candidate)
        return replace(result, candidates=new_candidates)

    # coarse-to-fine：第 0 轮用 MuJoCo prior 的邻域撒点(原行为)；后续每轮以上一轮 best 为
    # 唯一种子、半径按 refine_shrink^round 收缩重撒。总预算切分到各轮，best 单调不退化。
    rounds = max(1, int(identify_cfg.refinement_rounds))
    if identify_cfg.refine_candidates_per_round is not None:
        per_round = int(identify_cfg.refine_candidates_per_round)
        # 显式设值时 --candidates-per-joint 不再决定每轮数，总预算 = per_round * rounds。
        print(
            f"[fit-isaac] joint {joint_number}/6 refine_candidates_per_round={per_round} "
            f"已显式设定，总预算={per_round * rounds}(--candidates-per-joint 被忽略)",
            flush=True,
        )
    else:
        per_round = max(1, int(generation_cfg.candidates_per_joint) // rounds)
    print(
        f"[fit-isaac] joint {joint_number}/6 refinement rounds={rounds} "
        f"candidates_per_round={per_round} total_budget={per_round * rounds}",
        flush=True,
    )

    best_candidate: CandidateRow | None = None
    best_metrics: BatchMetricResult | None = None
    best_metric_index = 0
    evaluated = 0
    top_trace: list[dict[str, Any]] = []
    round_best_losses: list[float] = []
    agg_requested = 0
    agg_rejected = 0
    agg_duplicate = 0
    source_count = 0

    for round_idx in range(rounds):
        round_label = f"round {round_idx + 1}/{rounds}"
        round_t0 = time.perf_counter()
        print(f"[fit-isaac] joint {joint_number}/6 {round_label} generating candidates", flush=True)
        if round_idx == 0:
            round_gen_cfg = replace(generation_cfg, candidates_per_joint=per_round)
            round_priors = priors
        else:
            assert best_candidate is not None
            factor = float(identify_cfg.refine_shrink) ** round_idx
            round_gen_cfg = replace(
                generation_cfg.shrunk(factor),
                candidates_per_joint=per_round,
                exploration_fraction=0.0,  # 精修轮不再全局探索，专注 best 邻域
                # 种子正交化：candidates 内部会再加 (joint_number-1)，故 round 用 *1000 间隔，
                # 避免 (round=1,joint=1) 与 (round=0,joint=2) 撞到同一 RNG 流。
                random_seed=int(generation_cfg.random_seed) + round_idx * 1000,
            )
            # 用上一轮 best 当唯一 prior(best 当 prior.best、trace 清空)
            round_priors = replace(priors, best=replace(best_candidate, source="refine_seed"), trace=[])

        gen_t0 = time.perf_counter()
        generated = _clamp(
            generate_candidates_for_joint(round_priors, generation=round_gen_cfg, filtering=filter_cfg)
        )
        gen_s = time.perf_counter() - gen_t0
        print(
            f"[fit-isaac] joint {joint_number}/6 {round_label} generated "
            f"{len(generated.candidates)} candidates rejected={generated.rejected_count} "
            f"duplicates={generated.duplicate_count} sources={generated.source_count} gen_s={gen_s:.1f}",
            flush=True,
        )
        agg_requested += int(generated.requested_count)
        agg_rejected += int(generated.rejected_count)
        agg_duplicate += int(generated.duplicate_count)
        source_count = max(source_count, int(generated.source_count))

        r_candidate, r_metrics, r_index, r_trace, r_evaluated = _evaluate_candidates(
            backend, dataset, generated.candidates, num_envs, identify_cfg,
            q_real, qd_real, trace_top_n, evaluated, joint_number, round_label,
        )
        evaluated += r_evaluated
        for row in r_trace:
            _push_top_trace(top_trace, row, trace_top_n)
        if r_metrics is not None and (
            best_metrics is None or r_metrics.loss[r_index] < best_metrics.loss[best_metric_index]
        ):
            best_candidate = r_candidate
            best_metrics = r_metrics
            best_metric_index = r_index
        assert best_metrics is not None
        round_best_losses.append(float(best_metrics.loss[best_metric_index]))
        round_loss = float(r_metrics.loss[r_index]) if r_metrics is not None else float("nan")
        print(
            f"[fit-isaac] joint {joint_number}/6 {round_label} done "
            f"round_best_loss={round_loss:.6f} global_best_loss={best_metrics.loss[best_metric_index]:.6f} "
            f"round_s={time.perf_counter() - round_t0:.1f}",
            flush=True,
        )

    assert best_candidate is not None and best_metrics is not None
    top_trace.sort(key=lambda row: float(row["loss"]))
    del top_trace[trace_top_n:]
    trace_path = None
    if trace_dir is not None:
        trace_path = trace_dir / f"joint_{joint_number}_isaac_top_trace.csv"
        write_t0 = time.perf_counter()
        print(
            f"[fit-isaac] joint {joint_number}/6 writing top trace rows={len(top_trace)} path={trace_path}",
            flush=True,
        )
        _write_trace_csv(trace_path, top_trace)
        print(
            f"[fit-isaac] joint {joint_number}/6 wrote top trace write_s={time.perf_counter() - write_t0:.1f}",
            flush=True,
        )
    record = _result_record(
        joint_number,
        best_candidate,
        best_metrics,
        best_metric_index,
        _AggCounts(agg_requested, agg_rejected, agg_duplicate, source_count),
        evaluated,
        trace_path,
    )
    record["refinement_rounds_run"] = rounds
    record["round_best_losses"] = round_best_losses
    record["candidates_per_round"] = per_round
    print(
        f"[fit-isaac] joint {joint_number}/6 done evaluated={evaluated} "
        f"best_loss={best_metrics.loss[best_metric_index]:.6f} "
        f"round_best_losses={round_best_losses} joint_s={time.perf_counter() - joint_t0:.1f}",
        flush=True,
    )
    return IsaacJointFitOutput(
        joint_number=joint_number,
        params=best_candidate.params,
        record=record,
        trace=top_trace,
    )


def _push_top_trace(rows: list[dict[str, Any]], row: dict[str, Any], limit: int) -> None:
    if limit <= 0:
        return
    rows.append(row)
    if len(rows) > limit * 2:
        rows.sort(key=lambda item: float(item["loss"]))
        del rows[limit:]


def main(argv: Sequence[str] | None = None, backend_factory: BackendFactory | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fit all 6 ARX actuator joints using Isaac batch replay.")
    parser.add_argument("--data-dir", required=True, help="Directory containing joint_1.npz ... joint_6.npz.")
    parser.add_argument("--mujoco-report", required=True, help="MuJoCo fit report used as candidate prior.")
    parser.add_argument("--mujoco-trace-dir", default=None, help="Directory containing MuJoCo fit trace CSV files.")
    parser.add_argument("--output", required=True, help="Output actuator parameter YAML path.")
    parser.add_argument("--report", required=False, help="Output Isaac fit report YAML path.")
    parser.add_argument("--trace-dir", required=False, help="Directory for per-joint Isaac top trace CSV files.")
    parser.add_argument("--config", required=False, help="Identification YAML config for loss weights.")
    parser.add_argument("--num-envs", type=int, default=6144)
    parser.add_argument("--candidates-per-joint", type=int, default=49_152)
    parser.add_argument(
        "--refinement-rounds",
        type=int,
        default=None,
        help="Override config refinement_rounds without editing YAML.",
    )
    parser.add_argument(
        "--refine-candidates-per-round",
        type=int,
        default=None,
        help="Override config refine_candidates_per_round; when set, --candidates-per-joint is ignored.",
    )
    parser.add_argument("--prior-top-k", type=int, default=512)
    parser.add_argument("--trace-top-n", type=int, default=4096)
    parser.add_argument("--joint", type=int, choices=range(1, 7), default=None, help="Fit only one joint for smoke tests.")
    parser.add_argument(
        "--start-joint",
        type=int,
        choices=range(1, 7),
        default=1,
        help="Resume a full 6-joint fit from this joint; earlier joints are restored from --trace-dir.",
    )
    parser.add_argument("--usd-path", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--backend-mode", choices=("isaac", "fake"), default="isaac", help=argparse.SUPPRESS)
    parser.add_argument("--action-scale", type=float, default=0.25)
    parser.add_argument(
        "--default-joint-pos",
        nargs=6,
        type=float,
        default=[0.0, 1.0, 0.8, 0.0, 0.0, 0.0],
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
    )
    parser.add_argument("--cmd-source", choices=("smooth", "raw"), default="smooth",
                        help="此次数据命令源，写入导出 YAML 的 command_input(smooth=A方案)")
    parser.add_argument("--ema-alpha", type=float, default=0.05,
                        help="真机 1kHz 软件 EMA alpha，写入导出 YAML 的 software_ema")
    args = parser.parse_args(argv)

    if args.num_envs < 1:
        raise ValueError("--num-envs must be >= 1")
    if args.candidates_per_joint < 1:
        raise ValueError("--candidates-per-joint must be >= 1")
    if args.prior_top_k < 1:
        raise ValueError("--prior-top-k must be >= 1")

    data_dir = Path(args.data_dir)
    output_path = Path(args.output)
    report_path = Path(args.report) if args.report else output_path.with_name(f"{output_path.stem}_fit_report.yaml")
    trace_dir = Path(args.trace_dir) if args.trace_dir else None
    mujoco_trace_dir = Path(args.mujoco_trace_dir) if args.mujoco_trace_dir else None
    mujoco_report = Path(args.mujoco_report)
    identify_cfg = load_identification_config(args.config) if args.config else IdentificationConfig()
    if args.refinement_rounds is not None:
        if args.refinement_rounds < 1:
            raise ValueError("--refinement-rounds must be >= 1")
        identify_cfg = replace(identify_cfg, refinement_rounds=int(args.refinement_rounds))
    if args.refine_candidates_per_round is not None:
        if args.refine_candidates_per_round < 1:
            raise ValueError("--refine-candidates-per-round must be >= 1")
        identify_cfg = replace(identify_cfg, refine_candidates_per_round=int(args.refine_candidates_per_round))
    if args.joint is not None and int(args.start_joint) != 1:
        raise ValueError("--joint and --start-joint cannot be used together")
    joints = [int(args.joint)] if args.joint is not None else list(range(int(args.start_joint), 7))
    resumed_joints = [] if args.joint is not None else list(range(1, int(args.start_joint)))
    if resumed_joints and trace_dir is None:
        raise ValueError("--start-joint > 1 requires --trace-dir so earlier Isaac joints can be restored")
    _require_joint_files(data_dir, joints)
    _warn_cmd_source_mismatch(data_dir, joints, args.cmd_source)

    # Isaac 辨识为纯 IdealPD：本套 PhysX/IsaacLab articulation 臂已重力中性(实测 2026-06-05
    # probe，松手不掉)，训练侧 b2arx_env_cfg 也是纯 IdealPD、无前馈。故 Isaac 侧不加任何重力
    # 前馈。MuJoCo 侧同样纯 PD(gravity_scale=0)：实测真机数据是重力中性轨迹(固件已补)，
    # 纯 PD 拟 j2/j3/j4 比加 +G 好 6~17 倍。三方一致，不手动加 get_gravity_compensation_forces。
    isaac_cfg = IsaacReplayConfig(
        num_envs=int(args.num_envs),
        usd_path=args.usd_path,
        device=str(args.device),
        headless=bool(args.headless),
        effort_limit_sim=64.0,
    )
    if backend_factory is None:
        backend_factory = lambda cfg: IsaacBatchReplayBackend(cfg)
    backend = backend_factory(isaac_cfg)

    def _close_backend_once() -> None:
        close = getattr(backend, "close", None)
        if callable(close):
            close()

    atexit.register(_close_backend_once)
    default_generation_cfg = CandidateGenerationConfig()
    tau_limit_multiplier_bounds = (
        (1.0, 1.0)
        if _bounds_locked(identify_cfg.tau_limit_bounds)
        else default_generation_cfg.tau_limit_multiplier_bounds
    )
    generation_cfg = CandidateGenerationConfig(
        candidates_per_joint=int(args.candidates_per_joint),
        random_seed=int(identify_cfg.random_seed),
        tau_limit_multiplier_bounds=tau_limit_multiplier_bounds,
        filter_alpha_choices=tuple(float(value) for value in identify_cfg.filter_alpha_candidates),
        tau_lag_alpha_choices=tuple(float(value) for value in identify_cfg.tau_lag_alpha_candidates),
        time_shift_radius=0,
    )
    filter_cfg = CandidateFilterConfig(
        kp_bounds=tuple(float(value) for value in identify_cfg.kp_bounds),
        kd_bounds=tuple(float(value) for value in identify_cfg.kd_bounds),
        tau_limit_bounds=tuple(float(value) for value in identify_cfg.tau_limit_bounds),
        delay_steps_bounds=(
            min(int(value) for value in identify_cfg.delay_steps_candidates),
            max(int(value) for value in identify_cfg.delay_steps_candidates),
        ),
        max_kd_over_kp=float(identify_cfg.max_kd_over_kp),
    )

    print(
        f"[fit-isaac] data_dir={data_dir} num_envs={args.num_envs} "
        f"candidates_per_joint={args.candidates_per_joint} joints={joints} "
        f"resumed_joints={resumed_joints} "
        f"device={args.device} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}",
        flush=True,
    )

    outputs_by_joint: dict[int, IsaacJointFitOutput] = {}
    for joint_number in resumed_joints:
        assert trace_dir is not None
        outputs_by_joint[joint_number] = _load_existing_isaac_output(
            joint_number=joint_number,
            trace_dir=trace_dir,
            trace_top_n=int(args.trace_top_n),
        )
    for joint_number in joints:
        print(f"[fit-isaac] starting joint {joint_number}/6", flush=True)
        output = _fit_joint_isaac(
            joint_number=joint_number,
            data_dir=data_dir,
            mujoco_report=mujoco_report,
            mujoco_trace_dir=mujoco_trace_dir,
            identify_cfg=identify_cfg,
            generation_cfg=generation_cfg,
            filter_cfg=filter_cfg,
            backend=backend,
            num_envs=int(args.num_envs),
            prior_top_k=int(args.prior_top_k),
            trace_top_n=int(args.trace_top_n),
            trace_dir=trace_dir,
        )
        outputs_by_joint[joint_number] = output
        print(
            f"[fit-isaac] finished joint {joint_number}/6 "
            f"best_loss={output.record['loss']:.6f}",
            flush=True,
        )
        if joint_number != joints[-1]:
            reset_scene = getattr(backend, "reset_scene", None)
            if callable(reset_scene):
                print(
                    f"[fit-isaac] resetting Isaac scene after joint {joint_number}/6 "
                    f"before starting joint {joints[joints.index(joint_number) + 1]}/6",
                    flush=True,
                )
                reset_scene()

    refinement_rounds = max(1, int(identify_cfg.refinement_rounds))
    refinement_candidates_per_round = (
        int(identify_cfg.refine_candidates_per_round)
        if identify_cfg.refine_candidates_per_round is not None
        else max(1, int(args.candidates_per_joint) // refinement_rounds)
    )
    export_params: list[JointActuatorParams] = []
    records: list[dict[str, Any]] = []
    for joint_number in range(1, 7):
        if joint_number in outputs_by_joint:
            output = outputs_by_joint[joint_number]
            export_params.append(output.params)
            records.append(output.record)
        else:
            prior = read_mujoco_joint_priors(mujoco_report, mujoco_trace_dir, joint_number=joint_number, prior_top_k=1)
            prior_best = prior.best
            export_params.append(prior_best.params)
            records.append(
                {
                    "joint": joint_number,
                    "joint_name": JOINT_NAMES[joint_number - 1],
                    "status": "not_refit",
                    **candidate_to_record(prior_best),
                }
            )

    export_actuator_yaml(
        output_path,
        params=export_params,
        action_scale=args.action_scale,
        default_joint_pos=args.default_joint_pos,
        command_input=("q_cmd_smooth" if args.cmd_source == "smooth" else "q_cmd_raw"),
        software_ema_alpha_1khz=float(args.ema_alpha),
        # 跟随 config:本流程 gravity_compensation=False(纯 PD)。Isaac 从不施加重力前馈，
        # 此标志只如实记录辨识所用模型,供部署侧读 gravity_contract。
        gravity_compensation=bool(identify_cfg.gravity_compensation),
    )
    _write_report(
        report_path,
        {
            "data_dir": str(data_dir),
            "backend": "isaac",
            "backend_mode": args.backend_mode,
            "mujoco_prior_report": str(mujoco_report),
            "mujoco_trace_dir": str(mujoco_trace_dir) if mujoco_trace_dir else None,
            "output": str(output_path),
            "num_envs": int(args.num_envs),
            "candidates_per_joint": int(args.candidates_per_joint),
            "prior_top_k": int(args.prior_top_k),
            "trace_top_n": int(args.trace_top_n),
            "start_joint": int(args.start_joint),
            "resumed_joints": resumed_joints,
            "refinement": {
                "rounds": refinement_rounds,
                "shrink": float(identify_cfg.refine_shrink),
                "candidates_per_round": refinement_candidates_per_round,
                "configured_candidates_per_round": identify_cfg.refine_candidates_per_round,
            },
            "config": str(args.config) if args.config else None,
            "isaac": {
                "usd_path": args.usd_path,
                "device": args.device,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "headless": bool(args.headless),
                "gravity_feedforward": False,  # 纯 IdealPD，臂已重力中性(实测)
                "effort_limit_sim": float(isaac_cfg.effort_limit_sim),
            },
            "command_input": ("q_cmd_smooth" if args.cmd_source == "smooth" else "q_cmd_raw"),
            "ema_alpha": float(args.ema_alpha),
            "action_scale": float(args.action_scale),
            "default_joint_pos": [float(value) for value in args.default_joint_pos],
            "joints": records,
        },
    )
    print(f"[fit-isaac] wrote actuator parameters to {output_path}", flush=True)
    print(f"[fit-isaac] wrote fit report to {report_path}", flush=True)
    close = getattr(backend, "close", None)
    if callable(close):
        close()


if __name__ == "__main__":
    main()
