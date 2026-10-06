from __future__ import annotations

import copy

import numpy as np
import torch

from ..config import physics_substeps
from ..data import Trace


def discrete_coefficients(mass: float, damping: float, physics_dt: float, substeps: int) -> tuple:
    """Exact reduction of the plant's implicit damping + semi-implicit position steps.

    A rigid chain with a single unlocked revolute joint has constant scalar
    inertia. In this gravity-neutral, contact-free plant, C=G=0. The coefficients
    below reproduce SolverMuJoCo(implicitfast) for a held motor effort. They are
    only used to differentiate the small identification plant; official Newton
    still performs the final trajectory replay.
    """
    if mass <= 0 or damping < 0 or substeps < 1:
        raise ValueError("Invalid reduced dynamics")
    duration = physics_dt * substeps
    if damping == 0:
        return (1.0, duration / mass, duration,
                physics_dt ** 2 * substeps * (substeps + 1) / (2 * mass))
    r = 1 / (1 + physics_dt * damping / mass)
    position_v = physics_dt * r * (1 - r ** substeps) / (1 - r)
    return r ** substeps, (1 - r ** substeps) / damping, position_v, (duration - position_v) / damping


def pos_vel_features(error_history: torch.Tensor, velocity_history: torch.Tensor,
                     lags: torch.Tensor, pos_scale: float, vel_scale: float) -> torch.Tensor:
    """Torch twin of data.history_features; history column j holds sample t-j."""
    return torch.cat([error_history[:, lags] * pos_scale, velocity_history[:, lags] * vel_scale], 1)


def fit_trajectory(model, traces: list[Trace], commands: list[np.ndarray], prior: dict, cfg: dict,
                   pos_scale: float, vel_scale: float, label_scale: float, mass: float,
                   damping: float) -> dict:
    """Refine with encoder positions from chirp training blocks; never load step.

    `commands` are the traces' smooth commands after the baseline delay.
    """
    device = next(model.parameters()).device
    steps = int(cfg["training"]["trajectory_steps"])
    lags = cfg["network"]["input_idx"]
    max_lag = max(lags)
    train_starts, val_starts = [], []
    positions, velocities = [], []
    offset = 0
    for trace in traces:
        positions.append(trace.q)
        velocities.append(trace.velocity)
        indices = np.arange(max_lag, len(trace.t) - steps)
        for mask, target, stride in [(trace.train_mask, train_starts, 1), (trace.val_mask, val_starts, steps)]:
            starts = indices[mask[indices] & mask[indices + steps]]
            target.extend((starts[::stride] + offset).tolist())
        offset += len(trace.t)
    command = torch.tensor(np.concatenate(commands), dtype=torch.float32, device=device)
    measured_q = torch.tensor(np.concatenate(positions), dtype=torch.float32, device=device)
    measured_v = torch.tensor(np.concatenate(velocities), dtype=torch.float32, device=device)
    training_indices = torch.tensor(train_starts, dtype=torch.long, device=device)
    validation_indices = torch.tensor(val_starts, dtype=torch.long, device=device)
    if not len(training_indices) or not len(validation_indices):
        raise ValueError("Chirp blocks are too short for trajectory refinement")
    a, b, c, d = discrete_coefficients(mass, damping, cfg["plant"]["physics_dt"], physics_substeps(cfg))
    k = len(lags)
    effort_limit = float(prior["tau_limit"])
    history_offsets = torch.arange(max_lag + 1, device=device)
    lag_index = torch.tensor(lags, device=device)
    zero_input = torch.zeros(1, 2 * k, device=device)

    def measured_history(index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        previous = index[:, None] - history_offsets[None, :]
        return command[previous] - measured_q[previous], measured_v[previous]

    def rollout_loss(starts: torch.Tensor) -> torch.Tensor:
        # Start from measured history, then shift simulated samples into column 0.
        eh, vh = measured_history(starts)
        q, v = measured_q[starts], measured_v[starts]
        losses = []
        for tick in range(steps):
            if tick:
                eh = torch.cat([(command[starts + tick] - q)[:, None], eh[:, :-1]], 1)
                vh = torch.cat([v[:, None], vh[:, :-1]], 1)
            features = pos_vel_features(eh, vh, lag_index, pos_scale, vel_scale)
            tau = model(features).squeeze(1).clamp(-effort_limit, effort_limit)
            next_q, next_v = q + c * v + d * tau, a * v + b * tau
            error_q = (next_q - measured_q[starts + tick + 1]) / 0.02
            error_v = (next_v - measured_v[starts + tick + 1]) / 0.2
            losses.append(error_q.square().mean() + 0.02 * error_v.square().mean())
            q, v = next_q, next_v
        return torch.stack(losses).mean()

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["trajectory_learning_rate"]),
                                 weight_decay=float(cfg["training"]["weight_decay"]))
    best_loss, best_epoch, best_state = float("inf"), 0, None
    rows = []
    epochs = int(cfg["training"]["trajectory_epochs"])
    for epoch in range(epochs + 1):
        if epoch:
            selected = training_indices[torch.randint(len(training_indices),
                (int(cfg["training"]["trajectory_batch"]),), device=device)]
            model.train()
            loss = rollout_loss(selected)
            # Preserve a restoring low-frequency response and avoid learning an
            # undamped velocity self-excitation from inverse-dynamics targets.
            samples = pos_vel_features(*measured_history(selected[:32]), lag_index, pos_scale,
                                       vel_scale).detach().requires_grad_(True)
            gradients = torch.autograd.grad(model(samples).sum(), samples, create_graph=True)[0]
            ke = gradients[:, :k].sum(1) * pos_scale
            kv = gradients[:, k:].sum(1) * vel_scale
            stability = torch.relu(0.1 - ke / max(float(prior["kp"]), 0.1)).square().mean()
            stability += torch.relu(kv / max(damping, 0.1)).square().mean()
            equilibrium = (model(zero_input) / label_scale).square().mean()
            loss = loss + float(cfg["training"]["stability_weight"]) * stability
            loss = loss + float(cfg["training"]["equilibrium_weight"]) * equilibrium
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
        if epoch % 10 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                value = float(rollout_loss(validation_indices))
            rows.append({"epoch": epoch, "val_trajectory_loss": value})
            if value < best_loss:
                best_loss, best_epoch = value, epoch
                best_state = copy.deepcopy(model.state_dict())
            if epoch % 50 == 0:
                print(f"trajectory: epoch={epoch} heldout-chirp-window-loss={value:.6f}", flush=True)
    model.load_state_dict(best_state)
    return {"best_epoch": best_epoch, "val_trajectory_loss": best_loss,
            "rows": rows, "mass": mass, "passive_damping": damping,
            "training_windows": len(train_starts), "validation_windows": len(val_starts),
            "selection": "position/velocity rollout on chirp validation windows; no step data"}
