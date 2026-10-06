from __future__ import annotations

import argparse
from pathlib import Path

from arx_actuator_identification.actuator import JointActuatorParams
from arx_actuator_identification.export import export_actuator_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Export ARX actuator parameters to YAML.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--action-scale", type=float, default=0.25)
    args = parser.parse_args()
    params = [
        JointActuatorParams(kp=1.0, kd=0.0, delay_steps=0, filter_alpha=1.0, tau_limit=1.0, tau_lag_alpha=1.0)
        for _ in range(6)
    ]
    export_actuator_yaml(
        Path(args.output),
        params=params,
        action_scale=args.action_scale,
        default_joint_pos=[0.0, 1.0, 0.8, 0.0, 0.0, 0.0],
    )
