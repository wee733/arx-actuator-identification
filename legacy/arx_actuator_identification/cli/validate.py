from __future__ import annotations

import argparse

from arx_actuator_identification.dataset import load_sweep_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate ARX actuator dataset loading.")
    parser.add_argument("--npz", required=True)
    parser.add_argument("--metadata", required=True)
    args = parser.parse_args()
    dataset = load_sweep_dataset(args.npz, args.metadata)
    print(f"loaded joint {dataset.active_joint + 1} dataset with {len(dataset.t)} samples")
