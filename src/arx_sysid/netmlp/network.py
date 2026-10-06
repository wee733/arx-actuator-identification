from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn


class ActuatorMLP(nn.Module):
    """Official MLP input/output shape with a fixed PD initialization prior.

    The residual is learned from inverse-dynamics estimates, not PD-generated
    targets. Its bounded output limits extrapolation of the small-motion data.
    """

    def __init__(self, history_size: int, hidden: int, kp: float, kd: float,
                 pos_scale: float, vel_scale: float, residual_scale: float):
        super().__init__()
        self.history_size = history_size
        self.register_buffer("kp_scaled", torch.tensor(kp / pos_scale))
        self.register_buffer("kd_scaled", torch.tensor(kd / vel_scale))
        self.register_buffer("residual_scale", torch.tensor(residual_scale))
        self.residual = nn.Sequential(
            nn.Linear(2 * history_size, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1), nn.Tanh(),
        )
        nn.init.zeros_(self.residual[-2].weight)
        nn.init.zeros_(self.residual[-2].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prior = self.kp_scaled * x[:, 0:1] - self.kd_scaled * x[:, self.history_size:self.history_size + 1]
        return prior + self.residual_scale * self.residual(torch.clamp(x, -4.0, 4.0))


def export_network(model: ActuatorMLP, directory: Path, metadata: dict) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    model = model.cpu().eval()
    metadata_json = json.dumps(metadata, ensure_ascii=False)
    script = torch.jit.script(model)
    script_path = directory / "actuator.pt"
    torch.jit.save(script, str(script_path), _extra_files={"metadata.json": metadata_json})
    example = torch.zeros(2, 2 * model.history_size)
    exported = torch.export.export(
        model, (example,), dynamic_shapes={"x": {0: torch.export.Dim("batch", min=1, max=16384)}}
    )
    export_path = directory / "actuator.pt2"
    torch.export.save(exported, export_path, extra_files={"metadata.json": metadata_json})
    (directory / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    # Check nonzero inputs and more than one batch size at the export boundary.
    max_error = 0.0
    exported_module = exported.module()
    for count in (1, 7):
        sample = torch.linspace(-1.5, 1.5, count * 2 * model.history_size).reshape(count, -1)
        with torch.no_grad():
            expected = model(sample)
            max_error = max(max_error, float((script(sample) - expected).abs().max()),
                            float((exported_module(sample) - expected).abs().max()))
    if max_error > 1e-5:
        raise RuntimeError(f"Export changed the network output: {max_error}")
    return {"torchscript": str(script_path), "pt2": str(export_path), "max_export_error": max_error}
