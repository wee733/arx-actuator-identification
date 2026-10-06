from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from arx_actuator_identification.analysis.plots.loss import plot_loss_trace_svg
from arx_actuator_identification.actuator import JointActuatorParams
from arx_actuator_identification.cli.common import infer_metadata_path, make_backend
from arx_actuator_identification.config import IdentificationConfig, load_identification_config
from arx_actuator_identification.dataset import load_sweep_dataset
from arx_actuator_identification.export import JOINT_NAMES, export_actuator_yaml
from arx_actuator_identification.fitting.search import FitResult, fit_joint_with_backend


@dataclass(frozen=True)
class JointFitTask:
    joint_number: int
    data_dir: str
    cfg: IdentificationConfig
    backend_name: str
    b2arx_config: str | None


@dataclass(frozen=True)
class JointFitOutput:
    joint_number: int
    params: JointActuatorParams
    record: dict[str, Any]
    trace: list[dict[str, Any]] | None


def _joint_npz_path(data_dir: Path, joint_number: int) -> Path:
    return data_dir / f"joint_{joint_number}.npz"


def _joint_metadata_path(npz_path: Path) -> Path:
    return Path(infer_metadata_path(str(npz_path)))


def _require_joint_files(data_dir: Path) -> None:
    missing: list[str] = []
    for joint_number in range(1, 7):
        npz_path = _joint_npz_path(data_dir, joint_number)
        metadata_path = _joint_metadata_path(npz_path)
        if not npz_path.exists():
            missing.append(str(npz_path))
        if not metadata_path.exists():
            missing.append(str(metadata_path))
    if missing:
        joined = "\n  ".join(missing)
        raise FileNotFoundError(f"missing required sweep files:\n  {joined}")


def _warn_cmd_source_mismatch(data_dir: Path, expected: str) -> None:
    # 防 footgun：NPZ metadata 标了 cmd_source 且与 --cmd-source 不符则报错；旧数据无键则放行。
    for joint_number in range(1, 7):
        meta_path = _joint_metadata_path(_joint_npz_path(data_dir, joint_number))
        try:
            meta = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
        except OSError:
            continue
        actual = meta.get("cmd_source")
        if actual is not None and str(actual) != str(expected):
            raise ValueError(
                f"joint_{joint_number} 数据 cmd_source='{actual}' 与 --cmd-source='{expected}' 不符；"
                f"导出契约会撒谎。请改 --cmd-source {actual} 或换数据目录。"
            )


def _result_record(joint_number: int, random_seed: int, result: FitResult) -> dict[str, Any]:
    return {
        "joint": joint_number,
        "joint_name": JOINT_NAMES[joint_number - 1],
        "random_seed": int(random_seed),
        "kp": float(result.params.kp),
        "kd": float(result.params.kd),
        "delay_steps": int(result.params.delay_steps),
        "filter_alpha": float(result.params.filter_alpha),
        "time_shift_steps": int(result.time_shift_steps),
        "tau_limit": float(result.params.tau_limit),
        "tau_lag_alpha": float(result.params.tau_lag_alpha),
        "gravity_scale": float(result.params.gravity_scale),
        "loss": float(result.loss),
        "q_rmse": float(result.q_rmse),
        "qd_rmse": float(result.qd_rmse),
        "kp_regularization": float(result.kp_regularization),
        "tau_limit_regularization": float(result.tau_limit_regularization),
        "gravity_scale_regularization": float(result.gravity_scale_regularization),
    }


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(report, handle, sort_keys=False, allow_unicode=True)


def _write_trace_csv(path: Path, trace: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "candidate_index",
        "delay_steps",
        "filter_alpha",
        "time_shift_steps",
        "kp",
        "kd",
        "tau_limit",
        "tau_lag_alpha",
        "gravity_scale",
        "loss",
        "q_rmse",
        "qd_rmse",
        "kp_regularization",
        "tau_limit_regularization",
        "gravity_scale_regularization",
        "best_loss",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trace)


def _write_trace_outputs(trace_dir: Path, joint_number: int, trace: list[dict[str, Any]]) -> dict[str, str]:
    csv_path = trace_dir / f"joint_{joint_number}_fit_trace.csv"
    svg_path = trace_dir / f"joint_{joint_number}_loss.svg"
    _write_trace_csv(csv_path, trace)
    plot_loss_trace_svg(
        svg_path,
        candidate_index=np.asarray([row["candidate_index"] for row in trace], dtype=float),
        loss=np.asarray([row["loss"] for row in trace], dtype=float),
        best_loss=np.asarray([row["best_loss"] for row in trace], dtype=float),
    )
    return {"trace_csv": str(csv_path), "loss_plot": str(svg_path)}


def _candidate_count(cfg: IdentificationConfig) -> int:
    gravity_scale_candidates = cfg.gravity_scale_candidates if cfg.gravity_compensation else [0.0]
    return (
        len(cfg.delay_steps_candidates)
        * len(cfg.filter_alpha_candidates)
        * len(cfg.time_shift_candidates)
        * len(cfg.tau_lag_alpha_candidates)
        * len(gravity_scale_candidates)
        * int(cfg.random_samples)
    )


def _fit_joint_task(task: JointFitTask) -> JointFitOutput:
    data_dir = Path(task.data_dir)
    joint_number = task.joint_number
    npz_path = _joint_npz_path(data_dir, joint_number)
    metadata_path = _joint_metadata_path(npz_path)
    print(f"[fit-all] joint {joint_number}/6 loading {npz_path}", flush=True)
    dataset = load_sweep_dataset(npz_path, metadata_path)
    if dataset.active_joint != joint_number - 1:
        raise ValueError(
            f"{npz_path} contains active_joint={dataset.active_joint}, expected {joint_number - 1}"
        )
    joint_seed = int(task.cfg.random_seed) + joint_number - 1
    joint_cfg = replace(task.cfg, random_seed=joint_seed)
    progress_interval = max(1, _candidate_count(joint_cfg) // 10)

    def _progress(done: int, total: int, best: FitResult) -> None:
        print(
            f"[fit-all] joint {joint_number}/6 candidates {done}/{total} "
            f"best_loss={best.loss:.6f} best_q_rmse={best.q_rmse:.6f} "
            f"best_qd_rmse={best.qd_rmse:.6f}",
            flush=True,
        )

    print(
        f"[fit-all] joint {joint_number}/6 fitting seed={joint_seed} "
        f"samples={len(dataset.t)} active_joint={dataset.active_joint}",
        flush=True,
    )
    backend = make_backend(task.backend_name, task.b2arx_config)
    result = fit_joint_with_backend(
        dataset,
        joint_cfg,
        backend,
        progress_callback=_progress,
        progress_interval=progress_interval,
    )
    record = _result_record(joint_number, joint_seed, result)
    print(
        f"[fit-all] joint {joint_number}/6 done backend={task.backend_name} "
        f"kp={record['kp']:.6f} kd={record['kd']:.6f} "
        f"delay_steps={record['delay_steps']} filter_alpha={record['filter_alpha']:.6f} "
        f"time_shift_steps={record['time_shift_steps']} tau_limit={record['tau_limit']:.6f} "
        f"gravity_scale={record['gravity_scale']:.6f} "
        f"q_rmse={record['q_rmse']:.6f} qd_rmse={record['qd_rmse']:.6f} "
        f"kp_reg={record['kp_regularization']:.6f} tau_reg={record['tau_limit_regularization']:.6f} "
        f"gravity_reg={record['gravity_scale_regularization']:.6f}",
        flush=True,
    )
    return JointFitOutput(
        joint_number=joint_number,
        params=result.params,
        record=record,
        trace=result.trace,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit all 6 ARX actuator joints and export actuator YAML.")
    parser.add_argument("--data-dir", required=True, help="Directory containing joint_1.npz ... joint_6.npz.")
    parser.add_argument("--output", required=True, help="Output actuator parameter YAML path.")
    parser.add_argument(
        "--report",
        required=False,
        help="Output fit report YAML path. Defaults to <output stem>_fit_report.yaml.",
    )
    parser.add_argument(
        "--trace-dir",
        required=False,
        help="Directory for per-joint fit trace CSV files and loss SVG plots.",
    )
    parser.add_argument("--config", required=False, help="Identification YAML config.")
    parser.add_argument("--backend", choices=("simple", "mujoco"), default="mujoco")
    parser.add_argument(
        "--b2arx-config",
        default=None,
        help="Path to b2arx_mj.yaml for --backend mujoco.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of joints to fit in parallel. Default: 1.",
    )
    parser.add_argument(
        "--random-samples",
        type=int,
        default=None,
        help="Override config random_samples. Total candidates = discrete grid size * random_samples.",
    )
    parser.add_argument("--action-scale", type=float, default=0.25)
    parser.add_argument(
        "--default-joint-pos",
        nargs=6,
        type=float,
        default=[0.0, 1.0, 0.8, 0.0, 0.0, 0.0],
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
    )
    # 导出契约：声明这套参数是给哪一路命令(EMA 后 q_cmd_smooth / EMA 前 q_cmd_raw)拟的，
    # 以及真机 1kHz EMA alpha。默认 smooth(A 方案)。务必与 csv_to_npz 的 --cmd-source 一致，
    # 否则导出的 YAML 会“撒谎”。
    parser.add_argument("--cmd-source", choices=("smooth", "raw"), default="smooth",
                        help="此次数据的命令源(smooth=EMA后/raw=EMA前)，写入导出 YAML 的 command_input")
    parser.add_argument("--ema-alpha", type=float, default=0.05,
                        help="真机 1kHz 软件 EMA alpha，写入导出 YAML 的 software_ema")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_path = Path(args.output)
    report_path = Path(args.report) if args.report else output_path.with_name(f"{output_path.stem}_fit_report.yaml")
    trace_dir = Path(args.trace_dir) if args.trace_dir else None
    cfg = load_identification_config(args.config) if args.config else IdentificationConfig()
    if args.random_samples is not None:
        if args.random_samples < 1:
            raise ValueError("--random-samples must be >= 1")
        cfg = replace(cfg, random_samples=int(args.random_samples))
    if args.jobs < 1:
        raise ValueError("--jobs must be >= 1")
    _require_joint_files(data_dir)
    _warn_cmd_source_mismatch(data_dir, args.cmd_source)
    jobs = min(int(args.jobs), 6)

    print(
        f"[fit-all] data_dir={data_dir} backend={args.backend} "
        f"config={args.config or '<defaults>'} output={output_path} jobs={jobs}",
        flush=True,
    )
    print(
        f"[fit-all] delay_candidates={cfg.delay_steps_candidates} "
        f"alpha_candidates={cfg.filter_alpha_candidates} "
        f"time_shift_candidates={cfg.time_shift_candidates} random_samples={cfg.random_samples} "
        f"tau_lag_alpha_candidates={cfg.tau_lag_alpha_candidates} "
        f"gravity_compensation={cfg.gravity_compensation} "
        f"gravity_scale_candidates={cfg.gravity_scale_candidates if cfg.gravity_compensation else [0.0]} "
        f"candidates_per_joint={_candidate_count(cfg)}",
        flush=True,
    )
    print(
        f"[fit-all] loss=w_q*q_rmse + w_qd*qd_rmse_from_q_gradient "
        f"+ w_reg_kp*kp^2 + w_reg_tau*tau_limit^2 "
        f"+ w_reg_gravity_scale*(gravity_scale-{1.0 if cfg.gravity_compensation else 0.0})^2 "
        f"(w_q={cfg.w_q}, w_qd={cfg.w_qd}, w_reg_kp={cfg.w_reg_kp}, "
        f"w_reg_tau={cfg.w_reg_tau}, w_reg_gravity_scale={cfg.w_reg_gravity_scale})",
        flush=True,
    )
    if trace_dir is not None:
        print(f"[fit-all] trace_dir={trace_dir}", flush=True)

    tasks = [
        JointFitTask(
            joint_number=joint_number,
            data_dir=str(data_dir),
            cfg=cfg,
            backend_name=args.backend,
            b2arx_config=args.b2arx_config,
        )
        for joint_number in range(1, 7)
    ]
    outputs_by_joint: dict[int, JointFitOutput] = {}
    if jobs == 1:
        for task in tasks:
            output = _fit_joint_task(task)
            outputs_by_joint[output.joint_number] = output
    else:
        print(f"[fit-all] fitting joints in parallel with {jobs} worker processes", flush=True)
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            futures = {executor.submit(_fit_joint_task, task): task.joint_number for task in tasks}
            for future in as_completed(futures):
                joint_number = futures[future]
                output = future.result()
                outputs_by_joint[output.joint_number] = output
                print(f"[fit-all] collected joint {joint_number}/6 result", flush=True)

    params = []
    records = []
    for joint_number in range(1, 7):
        output = outputs_by_joint[joint_number]
        record = dict(output.record)
        if trace_dir is not None and output.trace is not None:
            record.update(_write_trace_outputs(trace_dir, joint_number, output.trace))
            print(
                f"[fit-all] joint {joint_number}/6 wrote trace {record['trace_csv']} "
                f"and {record['loss_plot']}",
                flush=True,
            )
        params.append(output.params)
        records.append(record)

    export_actuator_yaml(
        output_path,
        params=params,
        action_scale=args.action_scale,
        default_joint_pos=args.default_joint_pos,
        command_input=("q_cmd_smooth" if args.cmd_source == "smooth" else "q_cmd_raw"),
        software_ema_alpha_1khz=float(args.ema_alpha),
        gravity_compensation=bool(cfg.gravity_compensation),
    )
    _write_report(
        report_path,
        {
            "data_dir": str(data_dir),
            "backend": args.backend,
            "config": str(args.config) if args.config else None,
            "b2arx_config": str(args.b2arx_config) if args.backend == "mujoco" else None,
            "jobs": jobs,
            "output": str(output_path),
            "command_input": ("q_cmd_smooth" if args.cmd_source == "smooth" else "q_cmd_raw"),
            "ema_alpha": float(args.ema_alpha),
            "gravity_compensation": bool(cfg.gravity_compensation),
            "action_scale": float(args.action_scale),
            "default_joint_pos": [float(value) for value in args.default_joint_pos],
            "joints": records,
        },
    )
    print(f"[fit-all] wrote actuator parameters to {output_path}", flush=True)
    print(f"[fit-all] wrote fit report to {report_path}", flush=True)


if __name__ == "__main__":
    main()
