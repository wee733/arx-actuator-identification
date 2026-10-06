from __future__ import annotations

from pathlib import Path

from arx_actuator_identification.b2arx.config import load_b2arx_mujoco_config
from arx_actuator_identification.backends.mujoco import B2arxMujocoReplayBackend
from arx_actuator_identification.backends.simple import SimpleJointReplayBackend, SingleJointPlant


def infer_metadata_path(npz_path: str) -> str:
    path = Path(npz_path)
    if path.suffix != ".npz":
        raise ValueError("--npz must point to a .npz file when --metadata is omitted")
    name = path.name.removesuffix(".npz")
    return str(path.with_name(f"{name}_metadata.yaml"))


def make_backend(name: str, b2arx_config: str | None):
    if name == "simple":
        return SimpleJointReplayBackend(SingleJointPlant())
    if name == "mujoco":
        cfg = load_b2arx_mujoco_config(b2arx_config)
        return B2arxMujocoReplayBackend(cfg)
    raise ValueError(f"unsupported backend: {name}")
