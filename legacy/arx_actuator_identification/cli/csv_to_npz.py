#!/usr/bin/env python3
"""把 arm_id_collect (C++) 产出的富 CSV 转成现有 fit 流程期望的 NPZ + metadata.yaml。

背景：C++ 采集器复用 2real 的 ArmInterface，同时记录两路命令——
  - q_cmd_raw    : set_target 写入值（EMA 之前，= 策略 action 解码后的目标）
  - q_cmd_smooth : 真正下发给 setJointPositions 的值（EMA 之后）
本脚本按 --cmd-source 选择其中一路作为 fit 的 `q_cmd`：
  - smooth -> A 方案：辨识 EMA 之后的固件等效 actuator（EMA 当作我们已知的软件层单独建模）
  - raw    -> B 方案：端到端辨识（含 EMA）
同一批 CSV 可转出两个 NPZ，离线对比哪路在 held-out 上更准。

输出严格对齐 dataset.SweepDataset 契约：
  t (N,), q_cmd (N,6), q (N,7), qd (N,7), joint_cur (N,7), active_joint int
其中 q/qd 的第 7 列(夹爪)与 joint_cur 全列我们未采集，按 0 填充（fit 不使用 joint_cur，
夹爪不在辨识关节范围内）。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

ARM_DOF = 6


def _read_meta(csv_path: Path) -> dict:
    meta_path = Path(str(csv_path) + ".meta.yaml")
    if not meta_path.exists():
        return {}
    with meta_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def convert(csv_path: Path, cmd_source: str, out_dir: Path, stem: str | None) -> Path:
    rows = np.genfromtxt(csv_path, delimiter=",", names=True)
    n = rows.shape[0]
    meta = _read_meta(csv_path)

    t = np.asarray(rows["t"], dtype=np.float64)
    active_joint = int(rows["active_joint"][0]) if n else int(meta.get("active_joint", 0))

    prefix = "q_cmd_smooth" if cmd_source == "smooth" else "q_cmd_raw"
    q_cmd = np.stack([rows[f"{prefix}{j}"] for j in range(ARM_DOF)], axis=1).astype(np.float64)
    q_arm = np.stack([rows[f"q_real{j}"] for j in range(ARM_DOF)], axis=1).astype(np.float64)
    qd_sdk = np.stack([rows[f"qd_real{j}"] for j in range(ARM_DOF)], axis=1).astype(np.float64)

    # 点4：dataset.qd 写从位置求导的速度，而非 SDK 原始速度。
    # 已确认 ARX SDK getJointVelocities() 数值不可靠(P1，偏大 1.6~4.8x)；
    # 而 fit 主 loss 的 qd ground truth 本就用 np.gradient(q,t)，simple backend 初值/plot 也读 dataset.qd。
    # 这里统一用 gradient，避免任何下游误用原始不可靠速度。原始 SDK 速度另存额外键 qd_sdk_raw 备查。
    qd_grad = np.gradient(q_arm, t, axis=0) if n > 1 else np.zeros_like(q_arm)

    # 补到 7 列（第 7 列夹爪占位 0），满足 fit 端 (N,7) 契约。
    q = np.concatenate([q_arm, np.zeros((n, 1))], axis=1)
    qd = np.concatenate([qd_grad, np.zeros((n, 1))], axis=1)
    qd_sdk_raw = np.concatenate([qd_sdk, np.zeros((n, 1))], axis=1)
    joint_cur = np.zeros((n, 7), dtype=np.float64)  # 未采集，fit 不使用

    # 点3：fit_all.py / fit_all_isaac.py 硬要求 data_dir/joint_<N>.npz(+_metadata.yaml)，N=active_joint+1。
    # 按 <out-dir>/<cmd_source>/joint_<N>.npz 输出，A/B 分目录，可直接整目录喂给 fit。
    joint_number = active_joint + 1
    if stem is None:
        target_dir = out_dir / cmd_source
        stem = f"joint_{joint_number}"
    else:
        target_dir = out_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    npz_path = target_dir / f"{stem}.npz"
    np.savez_compressed(
        npz_path,
        t=t,
        q_cmd=q_cmd,
        q=q,
        qd=qd,
        joint_cur=joint_cur,
        active_joint=np.array(active_joint, dtype=np.int64),
        qd_sdk_raw=qd_sdk_raw,  # 非契约键：原始 SDK 速度，仅供 P1 诊断，fit 不读
    )

    md = dict(meta)
    md["active_joint"] = active_joint
    md["sample_count"] = int(n)
    md["cmd_source"] = cmd_source
    md["qd_source"] = "np.gradient(q_real, t)"  # 标注 qd 来源，防误判
    md["source_csv"] = str(csv_path)
    md_path = target_dir / f"{stem}_metadata.yaml"
    with md_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(md, f, sort_keys=True, allow_unicode=True)

    dt = float(np.median(np.diff(t))) if n > 1 else float("nan")
    print(f"[csv_to_npz] {csv_path.name} -> {npz_path}  rows={n}  active={active_joint}  "
          f"cmd={cmd_source}  median_dt={dt:.5f}s ({1.0/dt:.1f}Hz)" if dt == dt else
          f"[csv_to_npz] {csv_path.name} -> {npz_path}  rows={n}")
    return npz_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", type=Path, nargs="+", help="arm_id_collect 产出的 CSV(可多个)")
    p.add_argument("--cmd-source", choices=["smooth", "raw", "both"], default="both",
                   help="选 q_cmd 来源：smooth=A方案 / raw=B方案 / both=两个都转")
    p.add_argument("--out-dir", type=Path, default=Path("data/converted"))
    p.add_argument("--stem", default=None,
                   help="输出文件名(省略时按 fit 契约用 <out-dir>/<cmd_source>/joint_<N>.npz)")
    args = p.parse_args()

    sources = ["smooth", "raw"] if args.cmd_source == "both" else [args.cmd_source]
    for csv_path in args.csv:
        for src in sources:
            convert(csv_path, src, args.out_dir, args.stem)


if __name__ == "__main__":
    main()
