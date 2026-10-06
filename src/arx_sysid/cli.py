from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


def main() -> None:
    parser = argparse.ArgumentParser(prog="arx-sysid", description="ARX R5 actuator identification")
    commands = parser.add_subparsers(dest="command", required=True)
    identify = commands.add_parser("identify", help="Coupled six-joint grey-box PD/friction identification")
    identify.add_argument("--config", type=Path, default=CONFIG_DIR / "greybox.yaml")
    identify.add_argument("--run-dir", type=Path, required=True, help="New output directory")
    calibrate = commands.add_parser("calibrate-current",
                                    help="Fit per-joint torque constants from static-pose captures")
    calibrate.add_argument("--config", type=Path, default=CONFIG_DIR / "greybox.yaml")
    calibrate.add_argument("--capture", type=Path, nargs="+", required=True, help="static profile CSV files")
    calibrate.add_argument("--output", type=Path, required=True, help="Torque-constant YAML to write")
    train = commands.add_parser("netmlp-train", help="Experimental per-joint neural actuator")
    train.add_argument("--config", type=Path, default=CONFIG_DIR / "netmlp.yaml")
    train.add_argument("--run-dir", type=Path, required=True, help="New output directory")
    validate = commands.add_parser("netmlp-validate", help="Replay a NetMLP run against PD")
    validate.add_argument("--run-dir", type=Path, required=True, help="Existing netmlp-train run")
    validate.add_argument("--stage", choices=["validation", "test", "all"], default="all")
    args = parser.parse_args()
    if args.command == "identify":
        from .identify import identify as run
        run(load_config(args.config), args.run_dir.resolve())
    elif args.command == "calibrate-current":
        from .current import calibrate
        calibrate(load_config(args.config), [p.resolve() for p in args.capture], args.output.resolve())
    elif args.command == "netmlp-train":
        from .netmlp.train import train as run
        run(load_config(args.config), args.run_dir.resolve())
    else:
        from .netmlp.validate import validate as run
        run(args.run_dir.resolve(), args.stage)


if __name__ == "__main__":
    main()
