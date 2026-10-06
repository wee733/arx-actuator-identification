from __future__ import annotations

from pathlib import Path

from newton.actuators import Actuator, ClampingMaxEffort, Delay, DriveNeuralMLP, DrivePD
import warp as wp
import yaml


def make_newton_actuator(record: dict, device: str = "cpu", neural: bool = True,
                         with_delay: bool = True) -> Actuator:
    """Official Newton 1.6 pipeline: delay -> neural/PD drive -> effort clamp."""
    if neural:
        drive = DriveNeuralMLP(record["exports"]["pt2"])
    else:
        drive = DrivePD(kp=wp.array([record["baseline"]["kp"]], dtype=wp.float32, device=device),
                        kd=wp.array([record["baseline"]["kd"]], dtype=wp.float32, device=device))
    steps = int(record["metadata"]["command_delay_steps"])
    delay = None
    if with_delay and steps:
        delay = Delay(delay_steps=wp.array([steps], dtype=wp.int32, device=device), max_delay=steps)
    return Actuator(
        indices=wp.array([0], dtype=wp.uint32, device=device), drive=drive, delay=delay,
        clamping=[ClampingMaxEffort(max_effort=wp.array(
            [record["metadata"]["effort_limit"]], dtype=wp.float32, device=device))],
        control_feedforward_attr=None,
    )


def make_isaaclab_configs(run_dir: str | Path, inference_dt: float,
                           commands_preprocessed: bool = False) -> dict:
    """Build official Lab 3.0 configs after the caller supplies EMA and delay.

    ActuatorNetMLPCfg does not contain a command delay/EMA stage. The caller
    must tick at the trained inference interval, delay q_cmd_smooth with the
    manifest's per-joint values, and hold effort between ticks. This function
    deliberately rejects an unconfirmed timing/command contract.
    """
    from isaaclab.actuators import ActuatorNetMLPCfg

    run_dir = Path(run_dir).resolve()
    manifest = yaml.safe_load((run_dir / "manifest.yaml").read_text())
    if abs(inference_dt - manifest["sample_dt"]) > 1e-8:
        raise ValueError(f"NetMLP history was trained at {manifest['sample_dt']} s; supply that inference_dt")
    if not commands_preprocessed:
        raise ValueError("Apply the deployment EMA and each joint's manifest delay before the MLP")
    configs = {}
    for record in manifest["joints"]:
        meta = record["metadata"]
        configs[record["name"]] = ActuatorNetMLPCfg(
            joint_names_expr=[record["name"]],
            network_file=str(run_dir / record["name"] / "actuator.pt"),
            input_idx=meta["input_idx"], input_order=meta["input_order"],
            pos_scale=meta["pos_scale"], vel_scale=meta["vel_scale"], torque_scale=meta["torque_scale"],
            actuator_effort_limit=meta["effort_limit"], joint_effort_limit=meta["effort_limit"],
            saturation_effort=meta["effort_limit"], actuator_velocity_limit=float("inf"),
            stiffness=None, damping=None,
        )
    return configs
