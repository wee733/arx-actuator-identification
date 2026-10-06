from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter
import yaml


ARM_JOINTS = 6


@dataclass
class ArmCapture:
    """All six arm joints of one single-joint excitation, on a uniform grid.

    Arrays are (samples, 6), column j holding model joint j + 1. Masks select
    samples of the active joint's chirp/step protocol.
    """

    path: Path
    joint: int
    profile: str
    t: np.ndarray
    command: np.ndarray
    q: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    train_mask: np.ndarray
    val_mask: np.ndarray
    score_mask: np.ndarray


@dataclass
class Trace:
    path: Path
    joint: int
    profile: str
    t: np.ndarray
    command: np.ndarray
    q: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    train_mask: np.ndarray
    val_mask: np.ndarray
    score_mask: np.ndarray


def trace_path(run: str | Path, joint: int, profile: str) -> Path:
    """CSV files use zero-based joints; `joint` is the one-based model joint."""
    return Path(run) / f"joint{joint - 1}_{profile}.csv"


def load_capture(run: str | Path, joint: int, profile: str, cfg: dict) -> ArmCapture:
    path = trace_path(run, joint, profile)
    meta = yaml.safe_load(Path(str(path) + ".meta.yaml").read_text())
    if int(meta["active_joint"]) != joint - 1 or meta["profile"] != profile:
        raise ValueError(f"Metadata does not match joint/profile: {path}")
    columns = [f"q_real{j}" for j in range(ARM_JOINTS)] + [f"q_cmd_smooth{j}" for j in range(ARM_JOINTS)]
    rows = np.genfromtxt(path, delimiter=",", names=True, usecols=["t"] + columns)
    raw_t = rows["t"]
    dt = float(cfg["sample_dt"])
    if not np.all(np.diff(raw_t) > 0) or abs(np.median(np.diff(raw_t)) - dt) > dt * 0.05:
        raise ValueError(f"Unexpected sampling interval: {path}")
    if np.max(np.diff(raw_t)) > 3 * dt:
        raise ValueError(f"Capture has a large timestamp gap: {path}")
    t = np.arange(0, float(raw_t[-1] - raw_t[0]) + dt * 0.25, dt)
    resample = lambda name: np.interp(t + raw_t[0], raw_t, rows[name])
    q = np.stack([resample(f"q_real{j}") for j in range(ARM_JOINTS)], axis=1)
    command = np.stack([resample(f"q_cmd_smooth{j}") for j in range(ARM_JOINTS)], axis=1)
    if not np.all(np.isfinite(q)) or not np.all(np.isfinite(command)):
        raise ValueError(f"Non-finite data: {path}")
    window = int(cfg["derivative_window"])
    order = int(cfg["derivative_polyorder"])
    if len(t) <= 2 * window:
        raise ValueError(f"Capture too short for derivative filter: {path}")
    # The SDK velocity is not calibrated in these captures. Use position derivatives.
    velocity = savgol_filter(q, window, order, deriv=1, delta=dt, axis=0)
    acceleration = savgol_filter(q, window, order, deriv=2, delta=dt, axis=0)
    edge = (window // 2) * dt
    score_mask = (t >= cfg["discard_start_s"]) & (t <= t[-1] - edge)
    train_mask = np.zeros(len(t), dtype=bool)
    val_mask = train_mask.copy()
    if profile == "chirp":
        split = cfg["train_fraction"] * t[-1]
        # A gap prevents derivative windows and feature histories crossing the split.
        gap = max(float(cfg["split_gap_s"]), edge)
        train_mask = score_mask & (t < split - gap)
        val_mask = score_mask & (t > split + gap)
    return ArmCapture(path, joint, profile, t, command, q, velocity, acceleration,
                      train_mask, val_mask, score_mask)


def load_trace(run: str | Path, joint: int, profile: str, cfg: dict) -> Trace:
    """The active joint of a capture, for single-joint models."""
    c = load_capture(run, joint, profile, cfg)
    j = joint - 1
    return Trace(c.path, joint, profile, c.t, c.command[:, j], c.q[:, j], c.velocity[:, j],
                 c.acceleration[:, j], c.train_mask, c.val_mask, c.score_mask)


def history_features(error: np.ndarray, velocity: np.ndarray, lags: list[int]) -> np.ndarray:
    """Official pos_vel order: [e(t-i)..., v(t-i)...], newest lag is zero."""
    def past(values: np.ndarray, lag: int) -> np.ndarray:
        if lag == 0:
            return values
        return np.concatenate([np.zeros(lag), values[:-lag]])
    if max(lags) >= len(error):
        raise ValueError("History is longer than the capture")
    return np.stack([past(error, i) for i in lags] + [past(velocity, i) for i in lags], axis=1)


def delay_command(command: np.ndarray, steps: int) -> np.ndarray:
    if steps < 0 or steps >= len(command):
        raise ValueError("Invalid command delay")
    if steps == 0:
        return command.copy()
    return np.concatenate([np.full(steps, command[0]), command[:-steps]])
