from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import newton
from newton.solvers import SolverMuJoCo
import numpy as np
import torch
import warp as wp
import yaml

from ..config import physics_substeps, runtime_versions
from ..data import Trace, history_features, load_trace
from .integrations import make_newton_actuator
from ..plant import make_builder


def verify_native_export(record: dict) -> dict:
    """Exercise the real official drive, history, clamp and state swapping."""
    actuator = make_newton_actuator(record, with_delay=False)
    current, following = actuator.state(), actuator.state()
    state = SimpleNamespace(joint_q=wp.zeros(1, device="cpu"), joint_qd=wp.zeros(1, device="cpu"))
    control = SimpleNamespace(joint_target_q=wp.zeros(1, device="cpu"),
                              joint_target_qd=wp.zeros(1, device="cpu"), joint_f=wp.zeros(1, device="cpu"))
    network = torch.jit.load(record["exports"]["torchscript"]).eval()
    meta = record["metadata"]
    error = np.sin(np.arange(32) * 0.23) * 0.03
    velocity = np.cos(np.arange(32) * 0.37) * 0.2
    features = history_features(error, velocity, meta["input_idx"])
    k = len(meta["input_idx"])
    features[:, :k] *= meta["pos_scale"]
    features[:, k:] *= meta["vel_scale"]
    with torch.no_grad():
        expected = network(torch.tensor(features, dtype=torch.float32)).ravel().numpy()
    expected = np.clip(expected, -meta["effort_limit"], meta["effort_limit"])
    actual = []
    for e, v in zip(error, velocity):
        state.joint_qd.assign(np.array([v], dtype=np.float32))
        control.joint_target_q.assign(np.array([e], dtype=np.float32))
        control.joint_f.zero_()
        actuator.step(state, control, current, following, dt=meta["sample_dt"])
        current, following = following, current
        actual.append(float(control.joint_f.numpy()[0]))
    difference = float(np.max(np.abs(expected - actual)))
    if difference > 5e-5:
        raise RuntimeError(f"Newton export/history mismatch for {record['name']}: {difference}")
    return {"passed": True, "samples": len(actual), "max_effort_difference": difference,
            "api": "Newton 1.6 Actuator + DriveNeuralMLP(.pt2) + ClampingMaxEffort"}


def replay(model: newton.Model, trace: Trace, record: dict, cfg: dict, neural: bool) -> dict:
    solver = SolverMuJoCo(model, use_mujoco_cpu=True, integrator="implicitfast",
                          disable_contacts=True, disable_sensors=True)
    state, following = model.state(), model.state()
    control = model.control()
    state.joint_q.assign(np.array([trace.q[0]], dtype=np.float32))
    state.joint_qd.assign(np.array([trace.velocity[0]], dtype=np.float32))
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    actuator = make_newton_actuator(record, neural=neural)
    act_state, act_next = actuator.state(), actuator.state()
    q = np.full(len(trace.t), np.nan)
    velocity = q.copy()
    torque = q.copy()
    substeps = physics_substeps(cfg)
    diverged = False
    for i, command in enumerate(trace.command):
        q[i] = float(state.joint_q.numpy()[0])
        velocity[i] = float(state.joint_qd.numpy()[0])
        if not np.isfinite(q[i]) or not np.isfinite(velocity[i]) or abs(q[i] - command) > 0.5 or abs(velocity[i]) > 50:
            diverged = True
            break
        control.joint_target_q.assign(np.array([command], dtype=np.float32))
        control.joint_f.zero_()
        actuator.step(state, control, act_state, act_next, dt=cfg["data"]["sample_dt"])
        act_state, act_next = act_next, act_state
        torque[i] = float(control.joint_f.numpy()[0])
        for _ in range(substeps):
            state.clear_forces()
            solver.step(state, following, control, None, cfg["plant"]["physics_dt"])
            state, following = following, state
    return {"q": q, "velocity": velocity, "torque": torque, "diverged": diverged}


def metrics(trace: Trace, result: dict, mask: np.ndarray) -> dict:
    valid = not result["diverged"] and np.all(np.isfinite(result["q"][mask]))
    if not valid:
        return {"q_rmse_rad": float("inf"), "velocity_rmse_rad_s": float("inf"),
                "peak_position_error_rad": float("inf"), "diverged": True}
    return {"q_rmse_rad": float(np.sqrt(np.mean((result["q"][mask] - trace.q[mask]) ** 2))),
            "velocity_rmse_rad_s": float(np.sqrt(np.mean((result["velocity"][mask] - trace.velocity[mask]) ** 2))),
            "peak_position_error_rad": float(np.max(np.abs(result["q"][mask] - trace.q[mask]))),
            "diverged": False}


def validate(run_dir: Path, stage: str = "all") -> dict:
    cfg = yaml.safe_load((run_dir / "config.resolved.yaml").read_text())
    versions = runtime_versions(cfg["runtime"])
    manifest = yaml.safe_load((run_dir / "manifest.yaml").read_text())
    torch.set_num_threads(2)
    output = run_dir / f"validation_{stage}"
    output.mkdir(exist_ok=False)
    report = {"stage": stage, "versions": versions, "label_is_measured": False,
              "plant": "Fixed-base single active joint, original inertias/passive damping, zero gravity, no contacts",
              "actuator_tick_s": manifest["sample_dt"], "physics_dt_s": manifest["physics_dt"],
              "native_exports": [], "records": [], "rollout_acceptance": True,
              "valid_for_policy_training": False,
              "limitation": "Small motions around one pose; estimated torque labels; no full robot/contact or real motor validation"}
    profiles = {"validation": ["chirp"], "test": ["step"], "all": ["chirp", "step"]}[stage]
    figure, axes = plt.subplots(len(manifest["joints"]), len(cfg["data"]["runs"]),
                                figsize=(13, 2.4 * len(manifest["joints"])), squeeze=False)
    for index, record in enumerate(manifest["joints"]):
        native = verify_native_export(record)
        report["native_exports"].append({"joint": record["joint"], **native})
        model = make_builder(Path(record["plant_xml"]).read_text()).finalize(device="cpu")
        for run_index, run in enumerate(cfg["data"]["runs"]):
            for profile in profiles:
                trace = load_trace(run, record["joint"], profile, cfg["data"])
                mask = trace.val_mask if profile == "chirp" else trace.score_mask
                print(f"Replay {record['name']} {Path(run).parent.name}/{profile}", flush=True)
                pd = replay(model, trace, record, cfg, neural=False)
                mlp = replay(model, trace, record, cfg, neural=True)
                pd_metrics, mlp_metrics = metrics(trace, pd, mask), metrics(trace, mlp, mask)
                passed = (not mlp_metrics["diverged"] and
                          mlp_metrics["q_rmse_rad"] <= pd_metrics["q_rmse_rad"] * cfg["validation"]["allowed_rmse_ratio"] and
                          mlp_metrics["peak_position_error_rad"] < cfg["validation"]["max_tracking_error_rad"])
                report["rollout_acceptance"] &= bool(passed)
                item = {"joint": record["joint"], "run": Path(run).parent.name, "profile": profile,
                        "csv": str(trace.path), "scored_samples": int(mask.sum()),
                        "pd": pd_metrics, "mlp": mlp_metrics, "passed": bool(passed)}
                report["records"].append(item)
                print(f"  position RMSE: PD={pd_metrics['q_rmse_rad']:.6f}, MLP={mlp_metrics['q_rmse_rad']:.6f} rad; pass={passed}", flush=True)
                stem = f"{Path(run).parent.name}_joint{record['joint']}_{profile}"
                with (output / f"{stem}.csv").open("w") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["t", "command_smooth", "q_real", "q_pd", "q_mlp", "tau_pd", "tau_mlp", "scored"])
                    writer.writerows(zip(trace.t, trace.command, trace.q, pd["q"], mlp["q"],
                                         pd["torque"], mlp["torque"], mask.astype(int)))
                axis = axes[index, run_index]
                if profile == profiles[-1]:
                    axis.plot(trace.t, trace.command, color="gray", alpha=0.5, label="command")
                    axis.plot(trace.t, trace.q, color="black", linewidth=0.9, label="real")
                    axis.plot(trace.t, pd["q"], color="#d68720", linewidth=0.8, label="PD")
                    axis.plot(trace.t, mlp["q"], color="#1978c8", linewidth=0.8, label="MLP")
                    axis.set_title(f"joint {record['joint']} | {Path(run).parent.name} | {profile}", fontsize=9)
                    axis.set_ylabel("rad")
                    if index == 0:
                        axis.legend(fontsize=8)
                (output / "report.yaml").write_text(yaml.safe_dump(report, sort_keys=False))
    figure.tight_layout()
    figure.savefig(output / "responses.png", dpi=140)
    plt.close(figure)
    (output / "report.yaml").write_text(yaml.safe_dump(report, sort_keys=False))
    print(f"Report: {output / 'report.yaml'}; rollout_acceptance={report['rollout_acceptance']}", flush=True)
    return report
