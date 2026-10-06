"""Fit, select and test coupled grey-box actuator parameters for the deployed arm.

Model selection uses only the chirp validation blocks; held-out step captures
are replayed once, for the selected candidate and the deployed baseline.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import csv
import multiprocessing
import os
from pathlib import Path

import numpy as np
import torch
import yaml

from .replay import arm_inertia, replay, rmse
from .config import runtime_versions
from .data import ARM_JOINTS, load_capture
from .greybox import GreyBoxArm, make_windows, position_rmse

JOINT_NAMES = [f"R5a_joint{j}" for j in range(1, ARM_JOINTS + 1)]


def baseline_params(cfg: dict) -> dict:
    """The deployed PD contract in the coupled-plant parameter layout."""
    baseline = yaml.safe_load(Path(cfg["plant"]["baseline"]).read_text())
    joints = [baseline["joints"][name] for name in JOINT_NAMES]
    if any(j["filter_alpha"] != 1 or j.get("tau_lag_alpha", 1) != 1 for j in joints):
        raise ValueError("Only an unfiltered post-EMA PD baseline is supported")
    sample_dt, physics_dt = cfg["data"]["sample_dt"], cfg["plant"]["physics_dt"]
    delay_ticks = [round(j["delay_steps"] * sample_dt / physics_dt) for j in joints]
    return {"kp": [float(j["kp"]) for j in joints], "kd": [float(j["kd"]) for j in joints],
            "delay_ticks": delay_ticks, "delay_s": [d * physics_dt for d in delay_ticks],
            "armature": [0.0] * ARM_JOINTS, "friction": [0.0] * ARM_JOINTS,
            "effort": [float(j["tau_limit"]) for j in joints]}


def fit_variant(cfg: dict, name: str, trainable: list[str], runs: list[str]) -> dict:
    """Continuous fit, then round the delay to whole ticks and refit the rest."""
    fit = cfg["fit"]
    torch.set_num_threads(int(fit["threads_per_worker"]))
    physics_dt = cfg["plant"]["physics_dt"]
    captures = [load_capture(run, joint, "chirp", cfg["data"]) for run in runs
                for joint in range(1, ARM_JOINTS + 1)]
    windows = make_windows(captures, "train_mask", fit["window_s"], fit["stride_s"], physics_dt,
                           fit["max_delay_s"])
    start = baseline_params(cfg)
    model = GreyBoxArm(
        arm_inertia(cfg["plant"]["source_xml"], cfg["plant"]["fixed_pose"], physics_dt),
        start["kp"], start["kd"], start["delay_s"],
        [fit["initial_armature"] if "armature" in trainable else 0.0] * ARM_JOINTS,
        [fit["initial_friction"] if "friction" in trainable else 0.0] * ARM_JOINTS,
        start["effort"], physics_dt, fit["max_delay_s"], set(trainable), int(fit["friction_iterations"]),
        compile=True,
    )
    rows = []

    def optimise(stage: str, iterations: int) -> None:
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(parameters, lr=float(fit["learning_rate"]))
        for iteration in range(iterations):
            joint_rmse = position_rmse(model(windows), windows)
            loss = (joint_rmse ** 2).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            rows.append({"stage": stage, "iteration": iteration,
                         "train_rmse_rad": float(loss.detach().sqrt())})

    optimise("continuous", int(fit["iterations"]))
    with torch.no_grad():
        ticks = torch.round(model.value("delay_s") / physics_dt)
        fraction = (ticks * physics_dt / fit["max_delay_s"]).clamp(1e-4, 1 - 1e-4)
        model.delay_logit.copy_(torch.logit(fraction))
    model.delay_logit.requires_grad_(False)
    optimise("whole_tick_delay", int(fit["delay_refit_iterations"]))
    params = model.parameters_dict()
    params["delay_ticks"] = [int(t) for t in ticks.tolist()]
    params["delay_s"] = [t * physics_dt for t in params["delay_ticks"]]
    params["effort"] = start["effort"]
    if "armature" not in trainable:
        params["armature"] = [0.0] * ARM_JOINTS
    if "friction" not in trainable:
        params["friction"] = [0.0] * ARM_JOINTS
    return {"name": name, "runs": [Path(r).parent.name for r in runs], "trainable": trainable,
            "params": params, "log": rows}


def replay_job(cfg: dict, params: dict, run: str, joint: int, profile: str) -> dict:
    capture = load_capture(run, joint, profile, cfg["data"])
    result = replay(capture, params, cfg["plant"]["source_xml"], cfg["plant"]["fixed_pose"],
                    cfg["plant"]["physics_dt"])
    mask = capture.val_mask if profile == "chirp" else capture.score_mask
    return {"run": Path(run).parent.name, "joint": joint, "profile": profile,
            "rmse_rad": rmse(capture, result, mask).tolist(), "diverged": result["diverged"],
            "t": capture.t, "command": capture.command, "real": capture.q, "sim": result["q"],
            "mask": mask}


def pooled_rmse(records: list[dict]) -> dict:
    """RMS over captures of the active joint and of the passive joints, in rad."""
    active = [r["rmse_rad"][r["joint"] - 1] for r in records]
    passive = [x for r in records for j, x in enumerate(r["rmse_rad"]) if j != r["joint"] - 1]
    return {"active_rmse_rad": float(np.sqrt(np.mean(np.square(active)))),
            "passive_rmse_rad": float(np.sqrt(np.mean(np.square(passive))))}


def summary(records: list[dict]) -> list[dict]:
    return [{"run": r["run"], "joint": r["joint"], "profile": r["profile"], "diverged": r["diverged"],
             "active_rmse_rad": r["rmse_rad"][r["joint"] - 1], "all_joint_rmse_rad": r["rmse_rad"]}
            for r in records]


def run_spread(fits: dict, name: str, runs: list[str]) -> dict:
    """Largest per-joint relative difference between single-run fits, per parameter.

    Position-only captures fix kp, kd and friction relative to the joint inertia;
    a large spread marks a direction the data cannot resolve.
    """
    single = [fits[(name, Path(r).parent.name)]["params"] for r in runs]
    spread = {}
    for key in ("kp", "kd", "friction", "armature", "delay_ticks"):
        values = np.array([p[key] for p in single], dtype=float)
        scale = np.maximum(np.abs(values).mean(0), 1e-9)
        spread[key] = np.round(np.ptp(values, axis=0) / scale, 3).tolist()
    return spread


def select(variants: dict, scores: dict, tolerance: float) -> str:
    """Simplest structure (config order) within `tolerance` of the best validation score."""
    best = min(scores[name]["active_rmse_rad"] for name in variants)
    return next(name for name in variants if scores[name]["active_rmse_rad"] <= best * (1 + tolerance))


def isaaclab_record(params: dict, cfg: dict) -> dict:
    """Per-joint values for Isaac Lab 3.0 DelayedPDActuatorCfg (Newton/MJWarp)."""
    physics_dt = cfg["plant"]["physics_dt"]
    return {name: {"stiffness": params["kp"][j], "damping": params["kd"][j],
                   "armature": params["armature"][j], "friction": params["friction"][j],
                   "effort_limit_sim": params["effort"][j],
                   "min_delay": params["delay_ticks"][j], "max_delay": params["delay_ticks"][j],
                   "delay_s": params["delay_ticks"][j] * physics_dt}
            for j, name in enumerate(JOINT_NAMES)}


def write_traces(directory: Path, label: str, records: list[dict]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for r in records:
        path = directory / f"{r['run']}_joint{r['joint']}_{r['profile']}_{label}.csv"
        header = (["t", "scored"] + [f"command{j}" for j in range(1, 7)]
                  + [f"real{j}" for j in range(1, 7)] + [f"sim{j}" for j in range(1, 7)])
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(np.column_stack([r["t"], r["mask"], r["command"], r["real"], r["sim"]]))


def plot(path: Path, baseline: list[dict], selected: list[dict], title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = sorted({r["run"] for r in baseline})
    figure, axes = plt.subplots(ARM_JOINTS, len(runs), figsize=(6.5 * len(runs), 2.3 * ARM_JOINTS),
                                squeeze=False)
    for base, new in zip(baseline, selected):
        axis = axes[base["joint"] - 1, runs.index(base["run"])]
        j = base["joint"] - 1
        scored = base["mask"]
        t = base["t"][scored]
        axis.plot(t, base["command"][scored, j], color="gray", alpha=0.5, label="command")
        axis.plot(t, base["real"][scored, j], color="black", linewidth=0.9, label="real")
        axis.plot(t, base["sim"][scored, j], color="#d68720", linewidth=0.8, label="baseline")
        axis.plot(t, new["sim"][scored, j], color="#1978c8", linewidth=0.8, label="identified")
        axis.set_title(f"joint {base['joint']} | {base['run']} | {title}", fontsize=9)
        axis.set_ylabel("rad")
    axes[0, 0].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def identify(cfg: dict, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an experiment: {output}")
    versions = runtime_versions(cfg["runtime"])
    output.mkdir(parents=True)
    (output / "config.resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (output / "versions.yaml").write_text(yaml.safe_dump(versions, sort_keys=False))
    print("Runtime:", versions, flush=True)
    # Fitting and replays run on CPU; hide GPUs from the workers so concurrent
    # training keeps its device memory.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    runs = cfg["data"]["runs"]
    variants = cfg["fit"]["variants"]
    # Fit every structure on both runs, and on each run alone to show parameter spread.
    datasets = {"all": runs} | {Path(r).parent.name: [r] for r in runs}
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=len(variants) * len(datasets), mp_context=context) as pool:
        futures = {(name, key): pool.submit(fit_variant, cfg, name, list(trainable), dataset)
                   for name, trainable in variants.items() for key, dataset in datasets.items()}
        fits = {key: future.result() for key, future in futures.items()}
    for (name, key), result in fits.items():
        print(f"fit {name}/{key}: " + ", ".join(
            f"{k}={np.round(v, 4).tolist()}" for k, v in result["params"].items()), flush=True)
        with (output / f"fit_{name}_{key}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(result["log"][0]))
            writer.writeheader()
            writer.writerows(result["log"])

    candidates = {"baseline": baseline_params(cfg)} | {name: fits[(name, "all")]["params"] for name in variants}
    chirp_jobs = [(label, run, joint) for label in candidates for run in runs for joint in range(1, ARM_JOINTS + 1)]
    with ProcessPoolExecutor(max_workers=12, mp_context=context) as pool:
        futures = [pool.submit(replay_job, cfg, candidates[label], run, joint, "chirp")
                   for label, run, joint in chirp_jobs]
        chirp = {}
        for (label, _, _), future in zip(chirp_jobs, futures):
            chirp.setdefault(label, []).append(future.result())
    selection = {label: pooled_rmse(records) for label, records in chirp.items()}
    selected = select(variants, selection, float(cfg["validation"]["selection_tolerance"]))
    print("Chirp validation (official replay):", selection, "selected:", selected, flush=True)

    with ProcessPoolExecutor(max_workers=12, mp_context=context) as pool:
        futures = {label: [pool.submit(replay_job, cfg, candidates[label], run, joint, "step")
                           for run in runs for joint in range(1, ARM_JOINTS + 1)]
                   for label in ("baseline", selected)}
        step = {label: [f.result() for f in jobs] for label, jobs in futures.items()}
    ratio = float(cfg["validation"]["allowed_rmse_ratio"])
    trajectories = []
    for base, new in zip(step["baseline"], step[selected]):
        b, n = base["rmse_rad"][base["joint"] - 1], new["rmse_rad"][new["joint"] - 1]
        trajectories.append({"run": base["run"], "joint": base["joint"], "baseline_active_rmse_rad": b,
                             "identified_active_rmse_rad": n, "passed": bool(not new["diverged"] and n <= b * ratio)})
    accepted = all(t["passed"] for t in trajectories)

    params = candidates[selected]
    record = {
        "schema_version": 1, "model": f"coupled_greybox_{selected}", "command_input": "q_cmd_smooth",
        "physics_dt": cfg["plant"]["physics_dt"], "delay_unit": "physics ticks of physics_dt",
        "plant": "fixed-base R5a, source inertia (identical to the B2arx USD), zero passive damping, "
                 "gravity neutral, Newton + MuJoCo implicitfast",
        "friction_semantics": "MuJoCo/MJWarp dof frictionloss with default solref/solimp",
        "label_is_measured": False, "accepted_on_step_test": accepted,
        "joints": isaaclab_record(params, cfg),
        "action_contract": {"kp": params["kp"], "kd": params["kd"],
                            "delay_seconds": [t * cfg["plant"]["physics_dt"] for t in params["delay_ticks"]],
                            "armature": params["armature"], "friction": params["friction"]},
    }
    (output / "params.yaml").write_text(yaml.safe_dump(record, sort_keys=False))
    report = {
        "versions": versions, "selected": selected, "accepted_on_step_test": accepted,
        "selection_metric": "pooled active-joint position RMSE on chirp validation blocks, official replay; "
                            "simplest structure within selection_tolerance of the best",
        "per_run_relative_spread": {name: run_spread(fits, name, runs) for name in variants},
        "chirp_validation": selection,
        "step_test": {label: pooled_rmse(records) for label, records in step.items()},
        "step_trajectories": trajectories,
        "candidates": candidates,
        "per_run_fits": {f"{name}/{key}": fits[(name, key)]["params"] for name, key in fits if key != "all"},
        "chirp_records": {label: summary(records) for label, records in chirp.items()},
        "limitation": "Small motions around one pose; position-only labels; MJWarp frictionloss creeps, "
                      "so real stiction offsets that persist for seconds are not reproducible in simulation",
    }
    (output / "report.yaml").write_text(yaml.safe_dump(report, sort_keys=False))
    write_traces(output / "traces", "baseline", step["baseline"] + chirp["baseline"])
    write_traces(output / "traces", selected, step[selected] + chirp[selected])
    plot(output / "step_test.png", step["baseline"], step[selected], "step (test)")
    plot(output / "chirp_validation.png", chirp["baseline"], chirp[selected], "chirp validation")
    print(f"Selected {selected}; step test {report['step_test']}; accepted={accepted}", flush=True)
    print(f"Parameters: {output / 'params.yaml'}", flush=True)
    return report
