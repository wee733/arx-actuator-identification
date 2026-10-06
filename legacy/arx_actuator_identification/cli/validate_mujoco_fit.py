from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from arx_actuator_identification.analysis.plots.response import plot_joint_response_svg
from arx_actuator_identification.backends.mujoco import B2arxMujocoReplayBackend
from arx_actuator_identification.b2arx.config import load_b2arx_mujoco_config
from arx_actuator_identification.cli.common import infer_metadata_path
from arx_actuator_identification.dataset import load_sweep_dataset
from arx_actuator_identification.export import JOINT_NAMES, load_actuator_yaml
from arx_actuator_identification.validate import compute_validation_metrics


def _joint_npz_path(data_dir: Path, joint_number: int) -> Path:
    return data_dir / f"joint_{joint_number}.npz"


def _joint_metadata_path(npz_path: Path) -> Path:
    return Path(infer_metadata_path(str(npz_path)))


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(report, handle, sort_keys=False, allow_unicode=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay fitted actuator parameters in MuJoCo and plot validation curves.")
    parser.add_argument("--data-dir", required=True, help="Directory containing joint_1.npz ... joint_6.npz.")
    parser.add_argument("--params", required=True, help="Actuator parameter YAML from arx_fit_all_actuators.")
    parser.add_argument("--output-dir", default="results_plot", help="Directory for response plots and report.")
    parser.add_argument(
        "--b2arx-config",
        default=None,
        help="Path to b2arx_mj.yaml. Defaults to bundled model config.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    params_path = Path(args.params)
    output_dir = Path(args.output_dir)
    params, actuator_data = load_actuator_yaml(params_path)
    backend = B2arxMujocoReplayBackend(load_b2arx_mujoco_config(args.b2arx_config))

    records = []
    for joint_number, param in enumerate(params, start=1):
        npz_path = _joint_npz_path(data_dir, joint_number)
        metadata_path = _joint_metadata_path(npz_path)
        dataset = load_sweep_dataset(npz_path, metadata_path)
        if dataset.active_joint != joint_number - 1:
            raise ValueError(
                f"{npz_path} contains active_joint={dataset.active_joint}, expected {joint_number - 1}"
            )

        replay = backend.replay(dataset, param)
        joint_idx = joint_number - 1
        qd_real = np.gradient(dataset.q[:, joint_idx], dataset.t)
        metrics = compute_validation_metrics(
            q_real=dataset.q[:, joint_idx],
            q_sim=replay.q,
            qd_real=qd_real,
            qd_sim=replay.qd,
        )
        plot_path = output_dir / f"joint_{joint_number}_response.svg"
        plot_joint_response_svg(
            plot_path,
            t=dataset.t,
            q_cmd=dataset.q_cmd[:, joint_idx],
            q_real=dataset.q[:, joint_idx],
            q_sim=replay.q,
            qd_real=qd_real,
            qd_sim=replay.qd,
        )
        record = {
            "joint": joint_number,
            "joint_name": JOINT_NAMES[joint_idx],
            "plot": str(plot_path),
            "q_rmse": float(metrics.q_rmse),
            "qd_rmse": float(metrics.qd_rmse),
            "qd_real_source": "gradient(q_real, t)",
            "peak_error": float(metrics.peak_error),
        }
        records.append(record)
        print(
            f"joint={joint_number} q_rmse={record['q_rmse']:.6f} "
            f"qd_rmse={record['qd_rmse']:.6f} peak_error={record['peak_error']:.6f} "
            f"plot={plot_path}"
        )

    report_path = output_dir / "validation_report.yaml"
    _write_report(
        report_path,
        {
            "data_dir": str(data_dir),
            "params": str(params_path),
            "b2arx_config": str(args.b2arx_config) if args.b2arx_config else None,
            "actuator_model": actuator_data.get("model"),
            "action_scale": actuator_data.get("action_scale"),
            "default_joint_pos": actuator_data.get("default_joint_pos"),
            "qd_real_source": "gradient(q_real, t)",
            "joints": records,
        },
    )
    print(f"wrote validation report to {report_path}")


if __name__ == "__main__":
    main()
