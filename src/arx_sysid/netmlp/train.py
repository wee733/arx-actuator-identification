from __future__ import annotations

import copy
import csv
from pathlib import Path

import numpy as np
import torch
import yaml

from ..config import runtime_versions
from ..data import delay_command, history_features, load_trace, trace_path
from .network import ActuatorMLP, export_network
from ..plant import InverseDynamics, single_joint_xml
from .trajectory import fit_trajectory


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train(cfg: dict, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an experiment: {output}")
    versions = runtime_versions(cfg["runtime"])
    output.mkdir(parents=True)
    (output / "config.resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (output / "versions.yaml").write_text(yaml.safe_dump(versions, sort_keys=False))
    print("Runtime:", versions, flush=True)
    device = str(cfg["training"]["device"])
    torch.manual_seed(int(cfg["training"]["seed"]))
    torch.set_num_threads(2)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Configured training GPU is unavailable")
    baseline = yaml.safe_load(Path(cfg["plant"]["baseline"]).read_text())
    manifest = {"schema_version": 1, "versions": versions, "sample_dt": cfg["data"]["sample_dt"],
                "physics_dt": cfg["plant"]["physics_dt"], "command_source": "q_cmd_smooth",
                "label_source": "Newton inverse dynamics M*qdd+C (zero gravity) + source passive damping*qd",
                "label_is_measured": False, "gravity_neutral": True,
                "baseline_source": cfg["plant"]["baseline"], "joints": []}
    for joint in cfg["network"]["joints"]:
        name = f"R5a_joint{joint}"
        prior = baseline["joints"][name]
        if prior["filter_alpha"] != 1 or prior.get("tau_lag_alpha", 1) != 1:
            raise ValueError("Only an unfiltered post-EMA PD baseline is supported")
        xml = single_joint_xml(cfg["plant"]["source_xml"], joint, cfg["plant"])
        directory = output / name
        directory.mkdir()
        (directory / "plant.xml").write_text(xml)
        inverse = InverseDynamics(xml, int(cfg["training"]["inverse_dynamics_batch"]), device)
        xs, ys, train_masks, val_masks, provenance, traces, commands = [], [], [], [], [], [], []
        for run in cfg["data"]["runs"]:
            trace = load_trace(run, joint, "chirp", cfg["data"])
            traces.append(trace)
            command = delay_command(trace.command, int(prior["delay_steps"]))
            commands.append(command)
            features = history_features(command - trace.q, trace.velocity, cfg["network"]["input_idx"])
            labels = inverse.labels(trace.q, trace.velocity, trace.acceleration)
            xs.append(features)
            ys.append(labels)
            train_masks.append(trace.train_mask)
            val_masks.append(trace.val_mask)
            provenance.append({"csv": str(trace.path), "train_samples": int(trace.train_mask.sum()),
                               "val_samples": int(trace.val_mask.sum()), "split_time_s": float(trace.t[-1] * cfg["data"]["train_fraction"]),
                               "heldout_step_csv": str(trace_path(run, joint, "step"))})
        x, y = np.concatenate(xs), np.concatenate(ys)
        train_mask, val_mask = np.concatenate(train_masks), np.concatenate(val_masks)
        k = len(cfg["network"]["input_idx"])
        pos_scale = 1 / max(float(np.std(x[train_mask, 0])), 0.01)
        vel_scale = 1 / max(float(np.std(x[train_mask, k])), 0.05)
        kp, kd = float(prior["kp"]), float(prior["kd"])
        pd_torque = kp * x[:, 0] - kd * x[:, k]
        x[:, :k] *= pos_scale
        x[:, k:] *= vel_scale
        residual_scale = max(0.05, float(np.quantile(np.abs(y[train_mask] - pd_torque[train_mask]), 0.99)) * 1.5)
        label_scale = max(float(np.std(y[train_mask])), 0.05)
        hidden = int(cfg["network"]["hidden_size"])
        model = ActuatorMLP(k, hidden, kp, kd, pos_scale, vel_scale, residual_scale).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["learning_rate"]),
                                     weight_decay=float(cfg["training"]["weight_decay"]))
        xt = torch.tensor(x[train_mask], dtype=torch.float32, device=device)
        yt = torch.tensor(y[train_mask, None], dtype=torch.float32, device=device)
        xv = torch.tensor(x[val_mask], dtype=torch.float32, device=device)
        yv = torch.tensor(y[val_mask, None], dtype=torch.float32, device=device)
        pd_prior = torch.tensor(pd_torque[train_mask, None], dtype=torch.float32, device=device)
        best_loss, best_epoch, best_state = float("inf"), 0, None
        rows = []
        for epoch in range(int(cfg["training"]["epochs"]) + 1):
            if epoch:
                model.train()
                prediction = model(xt)
                fit_loss = torch.nn.functional.smooth_l1_loss(prediction / label_scale, yt / label_scale, beta=0.5)
                regularization = ((prediction - pd_prior) / label_scale).square().mean()
                loss = fit_loss + float(cfg["training"]["pd_prior_weight"]) * regularization
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            model.eval()
            with torch.no_grad():
                train_rmse = float((model(xt) - yt).square().mean().sqrt())
                val_rmse = float((model(xv) - yv).square().mean().sqrt())
            if val_rmse < best_loss:
                best_loss, best_epoch = val_rmse, epoch
                best_state = copy.deepcopy(model.state_dict())
            rows.append({"epoch": epoch, "train_torque_estimate_rmse": train_rmse,
                         "val_torque_estimate_rmse": val_rmse})
            if epoch % 100 == 0:
                print(f"{name}: epoch={epoch} train={train_rmse:.5f} val={val_rmse:.5f} N*m (estimated labels)", flush=True)
        model.load_state_dict(best_state)
        trajectory = None
        if cfg["training"].get("trajectory_epochs", 0):
            trajectory = fit_trajectory(
                model, traces, commands, prior, cfg, pos_scale, vel_scale, label_scale,
                float(inverse.mass.numpy()[0, 0, 0]), float(inverse.damping[0]),
            )
            write_rows(directory / "trajectory_training.csv", trajectory.pop("rows"))
            with torch.no_grad():
                best_loss = float((model(xv) - yv).square().mean().sqrt())
        metadata = {"model_type": "mlp", "input_idx": cfg["network"]["input_idx"], "input_order": "pos_vel",
                    "pos_scale": pos_scale, "vel_scale": vel_scale, "effort_scale": 1.0, "torque_scale": 1.0,
                    "sample_dt": cfg["data"]["sample_dt"], "joint_name": name,
                    "command_delay_steps": int(prior["delay_steps"]), "effort_limit": float(prior["tau_limit"]),
                    "gravity_neutral": True, "label_is_measured": False,
                    "architecture": f"fixed PD prior + bounded {hidden}-{hidden} tanh MLP residual", "best_epoch": best_epoch}
        if trajectory:
            metadata["trajectory_refinement"] = trajectory
        exports = export_network(model, directory, metadata)
        # The model holds the selected weights after either training stage.
        torch.save({"state_dict": model.state_dict(), "metadata": metadata, "config": cfg},
                   directory / "training_state.pth")
        write_rows(directory / "training.csv", rows)
        record = {"name": name, "joint": joint, "metadata": metadata, "exports": exports,
                  "best_epoch": best_epoch, "val_torque_estimate_rmse": best_loss,
                  "pd_val_torque_estimate_rmse": float(np.sqrt(np.mean((pd_torque[val_mask] - y[val_mask]) ** 2))),
                  "baseline": prior, "plant_xml": str(directory / "plant.xml"), "provenance": provenance,
                  "residual_scale": residual_scale, "train_feature_min": x[train_mask].min(axis=0).tolist(),
                  "train_feature_max": x[train_mask].max(axis=0).tolist()}
        if trajectory:
            record["trajectory_refinement"] = trajectory
        manifest["joints"].append(record)
        (output / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
        print(f"{name}: exported epoch {best_epoch}, val estimate RMSE={best_loss:.5f} N*m", flush=True)
    return manifest
