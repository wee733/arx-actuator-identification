from __future__ import annotations

from pathlib import Path

import yaml

from arx_actuator_identification.actuator import JointActuatorParams


JOINT_NAMES = [
    "R5a_joint1",
    "R5a_joint2",
    "R5a_joint3",
    "R5a_joint4",
    "R5a_joint5",
    "R5a_joint6",
]


def export_actuator_yaml(
    path: str | Path,
    params: list[JointActuatorParams],
    action_scale: float,
    default_joint_pos: list[float],
    command_input: str = "q_cmd_smooth",
    software_ema_alpha_1khz: float = 0.05,
    gravity_compensation: bool = False,
) -> None:
    if len(params) != 6:
        raise ValueError("params must contain 6 joint parameter sets")
    if len(default_joint_pos) != 6:
        raise ValueError("default_joint_pos must contain 6 values")
    # EMA 时间常数: alpha = 1 - exp(-dt/tau) @ dt=1ms -> tau = -dt / ln(1-alpha)
    import math
    tau_s = (0.001 / -math.log(1.0 - software_ema_alpha_1khz)) if 0.0 < software_ema_alpha_1khz < 1.0 else 0.0
    data = {
        "model": "simple_pd_delay_filter_saturation",
        # 关键契约：这套 kp/kd/tau_limit 是给“EMA 之后的目标”(q_cmd_smooth)用的。
        # 训练侧必须先过同一个软件 EMA(raw->smooth)，再喂给本 actuator，否则 EMA 会算两遍。
        "command_input": command_input,
        "software_ema": {
            "enabled": True,
            "alpha_1khz_real": float(software_ema_alpha_1khz),
            "tau_s": float(tau_s),
        },
        # 重力契约(文档性，目前无代码消费，写清防误读)。
        # 目标物理行为：臂“重力中性”(松手不下垂)，与真机 ARX 固件内置重力补偿一致。
        # 实测结论(2026-06-05)：真机固件已补重力，采集的 q_real 是重力中性轨迹；
        #   - Isaac articulation 实测也重力中性(probe) → 不加前馈。
        #   - MuJoCo 用纯 PD 拟 j2/j3/j4 比加 +G 好 6~17 倍 → 也不加前馈(gravity_scale=0)。
        # 三方一致：辨识/部署都按“重力中性臂 + 纯 PD”，不手动加 get_gravity_compensation_forces。
        "gravity_compensation": bool(gravity_compensation),
        "gravity_contract": {
            "target": "gravity_neutral_arm",
            "isaac_feedforward": False,   # PhysX/articulation 实测中性
            "mujoco_feedforward": False,  # 纯 PD 拟真机中性轨迹更优(实测)
            "note": "real firmware already compensates gravity; kp/kd identified with pure PD, no feedforward.",
        },
        "action_scale": float(action_scale),
        "default_joint_pos": [float(value) for value in default_joint_pos],
        "joints": {},
    }
    exported_gravity_scale = None if gravity_compensation else 0.0
    for name, param in zip(JOINT_NAMES, params):
        data["joints"][name] = {
            "kp": float(param.kp),
            "kd": float(param.kd),
            "delay_steps": int(param.delay_steps),
            "filter_alpha": float(param.filter_alpha),
            "tau_limit": float(param.tau_limit),
            "tau_lag_alpha": float(param.tau_lag_alpha),
            "gravity_scale": float(param.gravity_scale if exported_gravity_scale is None else exported_gravity_scale),
        }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)


def load_actuator_yaml(path: str | Path) -> tuple[list[JointActuatorParams], dict]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if data.get("model") != "simple_pd_delay_filter_saturation":
        raise ValueError(f"unsupported actuator model: {data.get('model')}")
    joints = data.get("joints")
    if not isinstance(joints, dict):
        raise ValueError("actuator YAML must contain a joints mapping")
    default_gravity_scale = 1.0 if bool(data.get("gravity_compensation", False)) else 0.0
    params: list[JointActuatorParams] = []
    for name in JOINT_NAMES:
        if name not in joints:
            raise ValueError(f"actuator YAML missing joint {name}")
        values = joints[name]
        params.append(
            JointActuatorParams(
                kp=float(values["kp"]),
                kd=float(values["kd"]),
                delay_steps=int(values["delay_steps"]),
                filter_alpha=float(values["filter_alpha"]),
                tau_limit=float(values["tau_limit"]),
                tau_lag_alpha=float(values.get("tau_lag_alpha", 1.0)),
                gravity_scale=float(values.get("gravity_scale", default_gravity_scale)),
            )
        )
    return params, data
