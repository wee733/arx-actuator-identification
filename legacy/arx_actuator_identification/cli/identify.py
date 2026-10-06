from __future__ import annotations

import argparse

from arx_actuator_identification.cli.common import infer_metadata_path, make_backend
from arx_actuator_identification.config import IdentificationConfig, load_identification_config
from arx_actuator_identification.dataset import load_sweep_dataset
from arx_actuator_identification.fitting.search import fit_joint_with_backend


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit ARX actuator parameters from one sweep dataset.")
    parser.add_argument("--npz", required=True)
    parser.add_argument("--metadata", required=False, help="Defaults to <npz stem>_metadata.yaml.")
    parser.add_argument("--config", required=False)
    parser.add_argument("--backend", choices=("simple", "mujoco"), default="simple")
    parser.add_argument(
        "--b2arx-config",
        default=None,
        help="Path to b2arx_mj.yaml for --backend mujoco.",
    )
    args = parser.parse_args()
    cfg = load_identification_config(args.config) if args.config else IdentificationConfig()
    metadata_path = args.metadata or infer_metadata_path(args.npz)
    dataset = load_sweep_dataset(args.npz, metadata_path)
    backend = make_backend(args.backend, args.b2arx_config)
    result = fit_joint_with_backend(dataset, cfg, backend)
    print(
        f"backend={args.backend} joint={dataset.active_joint + 1} "
        f"kp={result.params.kp:.6f} kd={result.params.kd:.6f} "
        f"delay_steps={result.params.delay_steps} filter_alpha={result.params.filter_alpha:.6f} "
        f"time_shift_steps={result.time_shift_steps} tau_limit={result.params.tau_limit:.6f} "
        f"tau_lag_alpha={result.params.tau_lag_alpha:.6f} "
        f"q_rmse={result.q_rmse:.6f} qd_rmse={result.qd_rmse:.6f} "
        f"kp_reg={result.kp_regularization:.6f} tau_reg={result.tau_limit_regularization:.6f}"
    )
