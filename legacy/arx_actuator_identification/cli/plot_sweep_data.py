from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

from arx_actuator_identification.analysis.plots.sweep_data import plot_sweep_data_svg
from arx_actuator_identification.cli.common import infer_metadata_path
from arx_actuator_identification.dataset import load_sweep_dataset
from arx_actuator_identification.export import JOINT_NAMES


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


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plot collected ARX sweep datasets.")
    parser.add_argument("--data-dir", required=True, help="Directory containing joint_1.npz ... joint_6.npz.")
    parser.add_argument("--output-dir", default="results_plot/sweep_data", help="Directory for sweep data SVG plots.")
    parser.add_argument("--joint", type=int, choices=range(1, 7), default=None, help="Plot only one joint.")
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    joints = [int(args.joint)] if args.joint is not None else list(range(1, 7))
    _require_joint_files(data_dir, joints)

    records: list[dict[str, Any]] = []
    for joint_number in joints:
        npz_path = _joint_npz_path(data_dir, joint_number)
        metadata_path = _joint_metadata_path(npz_path)
        dataset = load_sweep_dataset(npz_path, metadata_path)
        expected_active_joint = joint_number - 1
        if dataset.active_joint != expected_active_joint:
            raise ValueError(
                f"{npz_path} contains active_joint={dataset.active_joint}, expected {expected_active_joint}"
            )

        joint_idx = joint_number - 1
        q_cmd = dataset.q_cmd[:, joint_idx]
        q_real = dataset.q[:, joint_idx]
        qd_real = dataset.qd[:, joint_idx]
        tracking_error = q_real - q_cmd
        plot_path = output_dir / f"joint_{joint_number}_sweep_data.svg"
        plot_sweep_data_svg(
            plot_path,
            t=dataset.t,
            q_cmd=q_cmd,
            q_real=q_real,
            qd_real=qd_real,
            tracking_error=tracking_error,
        )

        record = {
            "joint": joint_number,
            "joint_name": JOINT_NAMES[joint_idx],
            "plot": str(plot_path),
            "sample_count": int(len(dataset.t)),
            "duration_s": float(dataset.t[-1] - dataset.t[0]) if len(dataset.t) > 1 else 0.0,
            "q_cmd_min": float(np.min(q_cmd)),
            "q_cmd_max": float(np.max(q_cmd)),
            "q_real_min": float(np.min(q_real)),
            "q_real_max": float(np.max(q_real)),
            "tracking_error_rmse": float(np.sqrt(np.mean(np.square(tracking_error)))),
            "tracking_error_peak": float(np.max(np.abs(tracking_error))),
            "metadata": {
                "sweep_profile": dataset.metadata.get("sweep_profile"),
                "freq_start_hz": dataset.metadata.get("freq_start_hz"),
                "freq_end_hz": dataset.metadata.get("freq_end_hz"),
                "duration_s": dataset.metadata.get("duration_s"),
                "high_freq_hold_s": dataset.metadata.get("high_freq_hold_s"),
                "total_duration_s": dataset.metadata.get("total_duration_s"),
            },
        }
        records.append(record)
        print(
            f"joint={joint_number} samples={record['sample_count']} "
            f"tracking_rmse={record['tracking_error_rmse']:.6f} "
            f"tracking_peak={record['tracking_error_peak']:.6f} plot={plot_path}",
            flush=True,
        )

    report_path = output_dir / "sweep_data_report.yaml"
    _write_report(
        report_path,
        {
            "data_dir": str(data_dir),
            "output_dir": str(output_dir),
            "joints": records,
        },
    )
    print(f"wrote sweep data report to {report_path}", flush=True)


if __name__ == "__main__":
    main()
