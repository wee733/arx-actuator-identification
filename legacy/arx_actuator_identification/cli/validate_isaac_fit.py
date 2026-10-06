from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import yaml

from arx_actuator_identification.analysis.plots.loss import plot_loss_trace_svg
from arx_actuator_identification.analysis.plots.response import plot_joint_response_svg
from arx_actuator_identification.backends.isaac import IsaacBatchReplayBackend, IsaacReplayConfig
from arx_actuator_identification.cli.common import infer_metadata_path
from arx_actuator_identification.dataset import load_sweep_dataset
from arx_actuator_identification.export import JOINT_NAMES, load_actuator_yaml
from arx_actuator_identification.validate import compute_validation_metrics


BackendFactory = Callable[[IsaacReplayConfig], Any]


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


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(report, handle, sort_keys=False, allow_unicode=True)


def _load_yaml(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def _fit_report_joint(fit_report: dict[str, Any], joint_number: int) -> dict[str, Any] | None:
    joints = fit_report.get("joints", [])
    if not isinstance(joints, list):
        return None
    for row in joints:
        if isinstance(row, dict) and int(row.get("joint", -1)) == joint_number:
            return row
    return None


def _resolve_trace_path(
    joint_number: int,
    fit_report: dict[str, Any],
    fit_report_path: Path | None,
    trace_dir: Path | None,
) -> Path | None:
    joint_record = _fit_report_joint(fit_report, joint_number)
    if joint_record is not None and joint_record.get("trace_csv"):
        trace_path = Path(str(joint_record["trace_csv"]))
        if trace_path.exists():
            return trace_path
        if fit_report_path is not None:
            relative_to_report = fit_report_path.parent / trace_path
            if relative_to_report.exists():
                return relative_to_report
        return trace_path
    if trace_dir is not None:
        return trace_dir / f"joint_{joint_number}_isaac_top_trace.csv"
    return None


def _read_loss_trace(path: Path) -> tuple[np.ndarray, np.ndarray]:
    candidate_index: list[int] = []
    loss: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} is empty")
        required = {"candidate_index", "loss"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"{path} missing required columns: {sorted(missing)}")
        for row in reader:
            candidate_index.append(int(row["candidate_index"]))
            loss.append(float(row["loss"]))
    if not candidate_index:
        raise ValueError(f"{path} does not contain any loss rows")
    x = np.asarray(candidate_index, dtype=np.int64)
    y = np.asarray(loss, dtype=float)
    order = np.argsort(x)
    return x[order], y[order]


def _plot_loss_trace_if_available(
    trace_path: Path | None,
    output_path: Path,
) -> tuple[Path | None, str | None]:
    if trace_path is None:
        return None, None
    if not trace_path.exists():
        return None, f"trace CSV not found: {trace_path}"
    candidate_index, loss = _read_loss_trace(trace_path)
    best_loss = np.minimum.accumulate(loss)
    plot_loss_trace_svg(output_path, candidate_index=candidate_index, loss=loss, best_loss=best_loss)
    return output_path, None


def _first_batch_row(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        return values
    if values.ndim == 2 and values.shape[0] >= 1:
        return values[0]
    raise ValueError(f"expected replay array with shape (N,) or (B, N), got {values.shape}")


def main(argv: Sequence[str] | None = None, backend_factory: BackendFactory | None = None) -> None:
    parser = argparse.ArgumentParser(description="Replay fitted actuator parameters in Isaac and plot validation curves.")
    parser.add_argument("--data-dir", required=True, help="Directory containing joint_1.npz ... joint_6.npz.")
    parser.add_argument("--params", required=True, help="Actuator parameter YAML from Isaac fitting.")
    parser.add_argument(
        "--fit-report",
        default=None,
        help="Isaac fit report YAML. Used to locate joint_N_isaac_top_trace.csv files.",
    )
    parser.add_argument(
        "--trace-dir",
        default=None,
        help="Fallback directory containing joint_N_isaac_top_trace.csv files.",
    )
    parser.add_argument("--output-dir", default="results_plot/isaac_validate", help="Directory for SVG plots and report.")
    parser.add_argument("--joint", type=int, choices=range(1, 7), default=None, help="Validate only one joint.")
    parser.add_argument("--usd-path", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--backend-mode", choices=("isaac", "fake"), default="isaac", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    params_path = Path(args.params)
    fit_report_path = Path(args.fit_report) if args.fit_report else None
    trace_dir = Path(args.trace_dir) if args.trace_dir else None
    output_dir = Path(args.output_dir)
    report_path = output_dir / "validation_report.yaml"
    joints = [int(args.joint)] if args.joint is not None else list(range(1, 7))

    _require_joint_files(data_dir, joints)
    params, actuator_data = load_actuator_yaml(params_path)
    fit_report = _load_yaml(fit_report_path)
    isaac_cfg = IsaacReplayConfig(
        num_envs=1,
        usd_path=args.usd_path,
        device=str(args.device),
        headless=bool(args.headless),
    )
    if backend_factory is None:
        backend_factory = lambda cfg: IsaacBatchReplayBackend(cfg)
    backend = backend_factory(isaac_cfg)

    records: list[dict[str, Any]] = []
    try:
        for joint_number in joints:
            npz_path = _joint_npz_path(data_dir, joint_number)
            metadata_path = _joint_metadata_path(npz_path)
            dataset = load_sweep_dataset(npz_path, metadata_path)
            if dataset.active_joint != joint_number - 1:
                raise ValueError(
                    f"{npz_path} contains active_joint={dataset.active_joint}, expected {joint_number - 1}"
                )

            joint_idx = joint_number - 1
            replay = backend.replay_batch(dataset, [params[joint_idx]])
            q_sim = _first_batch_row(replay.q)
            qd_sim = _first_batch_row(replay.qd)
            q_real = dataset.q[:, joint_idx]
            qd_real = np.gradient(q_real, dataset.t)
            metrics = compute_validation_metrics(
                q_real=q_real,
                q_sim=q_sim,
                qd_real=qd_real,
                qd_sim=qd_sim,
            )

            response_plot = output_dir / f"joint_{joint_number}_isaac_response.svg"
            plot_joint_response_svg(
                response_plot,
                t=dataset.t,
                q_cmd=dataset.q_cmd[:, joint_idx],
                q_real=q_real,
                q_sim=q_sim,
                qd_real=qd_real,
                qd_sim=qd_sim,
            )

            trace_path = _resolve_trace_path(joint_number, fit_report, fit_report_path, trace_dir)
            loss_plot, loss_plot_warning = _plot_loss_trace_if_available(
                trace_path,
                output_dir / f"joint_{joint_number}_isaac_loss.svg",
            )

            record = {
                "joint": joint_number,
                "joint_name": JOINT_NAMES[joint_idx],
                "response_plot": str(response_plot),
                "loss_plot": str(loss_plot) if loss_plot is not None else None,
                "loss_trace_csv": str(trace_path) if trace_path is not None else None,
                "loss_plot_warning": loss_plot_warning,
                "q_rmse": float(metrics.q_rmse),
                "qd_rmse": float(metrics.qd_rmse),
                "qd_real_source": "gradient(q_real, t)",
                "peak_error": float(metrics.peak_error),
            }
            records.append(record)
            print(
                f"joint={joint_number} q_rmse={record['q_rmse']:.6f} "
                f"qd_rmse={record['qd_rmse']:.6f} peak_error={record['peak_error']:.6f} "
                f"response_plot={response_plot} loss_plot={loss_plot}",
                flush=True,
            )
    finally:
        close = getattr(backend, "close", None)
        if callable(close):
            close()

    _write_report(
        report_path,
        {
            "data_dir": str(data_dir),
            "params": str(params_path),
            "fit_report": str(fit_report_path) if fit_report_path else None,
            "trace_dir": str(trace_dir) if trace_dir else None,
            "backend": "isaac",
            "backend_mode": args.backend_mode,
            "actuator_model": actuator_data.get("model"),
            "action_scale": actuator_data.get("action_scale"),
            "default_joint_pos": actuator_data.get("default_joint_pos"),
            "isaac": {
                "num_envs": 1,
                "usd_path": args.usd_path,
                "device": args.device,
                "headless": bool(args.headless),
                "gravity_feedforward": False,  # 纯 IdealPD，臂已重力中性(实测)
            },
            "qd_real_source": "gradient(q_real, t)",
            "loss_trace_note": "Isaac fitting currently stores top trace rows, not the full evaluated candidate history.",
            "joints": records,
        },
    )
    print(f"wrote Isaac validation report to {report_path}", flush=True)


if __name__ == "__main__":
    main()
