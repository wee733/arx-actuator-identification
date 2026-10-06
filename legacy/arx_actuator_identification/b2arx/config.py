from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from arx_actuator_identification.b2arx.joint_map import CONTROLLED_JOINT_NAMES, controlled_defaults_with_down_base


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_B2ARX_MJ_CONFIG = PACKAGE_ROOT / "models" / "b2arx_mj" / "b2arx_mj.yaml"


@dataclass
class B2arxMujocoConfig:
    path: Path
    xml_path: Path
    simulation_dt: float
    default_angles: np.ndarray
    kps: np.ndarray
    kds: np.ndarray
    effort_limits: np.ndarray
    init_base_height: float
    joint_armature: float
    raw: dict[str, Any]


def _as_vector(data: dict[str, Any], key: str, length: int) -> np.ndarray:
    values = np.asarray(data[key], dtype=float)
    if values.shape != (length,):
        raise ValueError(f"{key} must contain {length} values")
    return values


def default_b2arx_mujoco_config_path() -> Path:
    if DEFAULT_B2ARX_MJ_CONFIG.exists():
        return DEFAULT_B2ARX_MJ_CONFIG
    try:
        from ament_index_python.packages import get_package_share_directory
    except ImportError as exc:
        raise FileNotFoundError(f"default MuJoCo config not found: {DEFAULT_B2ARX_MJ_CONFIG}") from exc
    share_dir = Path(get_package_share_directory("arx_actuator_identification"))
    return share_dir / "models" / "b2arx_mj" / "b2arx_mj.yaml"


def load_b2arx_mujoco_config(path: str | Path | None = None) -> B2arxMujocoConfig:
    path = default_b2arx_mujoco_config_path() if path is None else Path(path).expanduser()
    path = path.resolve()
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    xml_path = (path.parent / data["xml_path"]).resolve()
    default_angles = _as_vector(data, "default_angles", len(CONTROLLED_JOINT_NAMES))
    kps = _as_vector(data, "kps", len(CONTROLLED_JOINT_NAMES))
    kds = _as_vector(data, "kds", len(CONTROLLED_JOINT_NAMES))
    effort_limits = _as_vector(data, "effort_limits", len(CONTROLLED_JOINT_NAMES))
    return B2arxMujocoConfig(
        path=path,
        xml_path=xml_path,
        simulation_dt=float(data["simulation_dt"]),
        default_angles=default_angles,
        kps=kps,
        kds=kds,
        effort_limits=effort_limits,
        init_base_height=float(data["init_base_height"]),
        joint_armature=float(data.get("joint_armature", 0.0)),
        raw=data,
    )


def down_base_default_angles(arm_pos: np.ndarray | None = None) -> np.ndarray:
    return controlled_defaults_with_down_base(arm_pos)
