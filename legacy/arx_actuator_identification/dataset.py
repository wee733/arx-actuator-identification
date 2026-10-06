from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass
class SweepDataset:
    t: np.ndarray
    q_cmd: np.ndarray
    q: np.ndarray
    qd: np.ndarray
    joint_cur: np.ndarray
    active_joint: int
    metadata: dict[str, Any]

    def validate(self) -> None:
        n = len(self.t)
        if self.q_cmd.shape != (n, 6):
            raise ValueError("q_cmd must have shape (N, 6)")
        if self.q.shape != (n, 7):
            raise ValueError("q must have shape (N, 7)")
        if self.qd.shape != (n, 7):
            raise ValueError("qd must have shape (N, 7)")
        if self.joint_cur.shape != (n, 7):
            raise ValueError("joint_cur must have shape (N, 7)")
        if not 0 <= self.active_joint < 6:
            raise ValueError("active_joint must be in [0, 5]")


def save_sweep_dataset(directory: str | Path, stem: str, dataset: SweepDataset) -> tuple[Path, Path]:
    dataset.validate()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    npz_path = directory / f"{stem}.npz"
    metadata_path = directory / f"{stem}_metadata.yaml"
    np.savez_compressed(
        npz_path,
        t=dataset.t,
        q_cmd=dataset.q_cmd,
        q=dataset.q,
        qd=dataset.qd,
        joint_cur=dataset.joint_cur,
        active_joint=np.array(dataset.active_joint, dtype=np.int64),
    )
    metadata = dict(dataset.metadata)
    metadata["active_joint"] = int(dataset.active_joint)
    metadata["sample_count"] = int(len(dataset.t))
    with metadata_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(metadata, handle, sort_keys=True, allow_unicode=True)
    return npz_path, metadata_path


def load_sweep_dataset(npz_path: str | Path, metadata_path: str | Path) -> SweepDataset:
    npz = np.load(npz_path)
    with Path(metadata_path).open("r", encoding="utf-8") as handle:
        metadata = yaml.safe_load(handle) or {}
    dataset = SweepDataset(
        t=npz["t"],
        q_cmd=npz["q_cmd"],
        q=npz["q"],
        qd=npz["qd"],
        joint_cur=npz["joint_cur"],
        active_joint=int(npz["active_joint"]),
        metadata=metadata,
    )
    dataset.validate()
    return dataset
