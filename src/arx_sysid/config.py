from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec
from pathlib import Path
import platform
import subprocess

import yaml


def load_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text())
    if cfg.get("schema_version") != 1:
        raise ValueError("Unsupported config schema")
    for name in ("source_xml", "baseline"):
        cfg["plant"][name] = str((path.parent / cfg["plant"][name]).resolve())
    cfg["data"]["runs"] = [str((path.parent / p).resolve()) for p in cfg["data"]["runs"]]
    if cfg["data"]["command_source"] != "smooth":
        raise ValueError("This experiment identifies the post-EMA actuator: use smooth commands")
    if "network" in cfg:
        lags = cfg["network"]["input_idx"]
        if not lags or lags[0] != 0 or len(set(lags)) != len(lags) or min(lags) < 0:
            raise ValueError("input_idx must begin with 0 and contain distinct nonnegative lags")
        if cfg["network"]["input_order"] != "pos_vel":
            raise ValueError("This trainer exports the official pos_vel contract")
    if any(x != 0 for x in cfg["plant"]["gravity"]):
        raise ValueError("These captures require a gravity-neutral identification plant")
    ratio = cfg["data"]["sample_dt"] / cfg["plant"]["physics_dt"]
    if abs(ratio - round(ratio)) > 1e-6 or ratio < 1:
        raise ValueError("sample_dt must be an integer multiple of physics_dt")
    window = cfg["data"]["derivative_window"]
    if window % 2 != 1 or window <= cfg["data"]["derivative_polyorder"]:
        raise ValueError("Savitzky-Golay window must be odd and larger than polyorder")
    if not 0 < cfg["data"]["train_fraction"] < 1:
        raise ValueError("train_fraction must lie in (0, 1)")
    return cfg


def physics_substeps(cfg: dict) -> int:
    """Physics steps per actuator tick; load_config checks the ratio is integral."""
    return round(cfg["data"]["sample_dt"] / cfg["plant"]["physics_dt"])


def isaaclab_root() -> Path | None:
    """Source checkout of the isaaclab package this interpreter imports, if any."""
    spec = find_spec("isaaclab")
    if spec is None or spec.origin is None:
        return None
    for parent in Path(spec.origin).resolve().parents:
        if (parent / "VERSION").is_file() and (parent / "isaaclab.sh").is_file():
            return parent
    return None


def runtime_versions(expected: dict) -> dict:
    """Record the simulator stack and check the API families listed in the config.

    Identification needs Newton, Warp, MuJoCo and PyTorch. Isaac Sim / Isaac Lab
    are optional; when the config names them they must be importable.
    """
    result = {"python": platform.python_version()}
    packages = {"newton": "newton", "torch": "torch", "warp": "warp-lang", "mujoco": "mujoco",
                "isaac_sim": "isaacsim"}
    for key, package in packages.items():
        try:
            result[key] = version(package)
        except PackageNotFoundError:
            pass
    lab = isaaclab_root()
    if lab is not None:
        result["isaac_lab"] = (lab / "VERSION").read_text().strip()
        result["isaac_lab_source"] = str(lab)
        revision = subprocess.run(["git", "-C", str(lab), "rev-parse", "HEAD"],
                                  capture_output=True, text=True)
        if revision.returncode == 0:
            result["isaac_lab_revision"] = revision.stdout.strip()
    for key, prefix in expected.items():
        if key not in result:
            raise RuntimeError(f"{key} {prefix}.x is required by the config but not installed")
        if not str(result[key]).startswith(str(prefix) + "."):
            raise RuntimeError(f"{key}: expected {prefix}.x, found {result[key]}")
    return result
