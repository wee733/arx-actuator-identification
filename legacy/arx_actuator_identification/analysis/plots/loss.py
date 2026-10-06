from __future__ import annotations

from pathlib import Path

import numpy as np


def _polyline(x: np.ndarray, y: np.ndarray, x_min: float, x_scale: float, y_min: float, y_scale: float) -> str:
    points = []
    for x_value, y_value in zip(x, y):
        if np.isfinite(x_value) and np.isfinite(y_value):
            x_px = 60.0 + (float(x_value) - x_min) * x_scale
            y_px = 330.0 - (float(y_value) - y_min) * y_scale
            points.append(f"{x_px:.2f},{y_px:.2f}")
    return " ".join(points)


def plot_loss_trace_svg(path: str | Path, candidate_index: np.ndarray, loss: np.ndarray, best_loss: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(candidate_index, dtype=float)
    loss = np.asarray(loss, dtype=float)
    best_loss = np.asarray(best_loss, dtype=float)
    finite_x = x[np.isfinite(x)]
    finite_y = np.concatenate([loss[np.isfinite(loss)], best_loss[np.isfinite(best_loss)]])
    if finite_x.size == 0 or finite_y.size == 0:
        raise ValueError("loss trace must contain finite values")

    x_min = float(np.min(finite_x))
    x_max = float(np.max(finite_x))
    y_min = float(np.min(finite_y))
    y_max = float(np.max(finite_y))
    if x_max <= x_min:
        x_max = x_min + 1.0
    if y_max <= y_min:
        y_max = y_min + 1.0
    y_margin = 0.08 * (y_max - y_min)
    y_min -= y_margin
    y_max += y_margin
    x_scale = 700.0 / (x_max - x_min)
    y_scale = 280.0 / (y_max - y_min)

    loss_points = _polyline(x, loss, x_min, x_scale, y_min, y_scale)
    best_points = _polyline(x, best_loss, x_min, x_scale, y_min, y_scale)
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="820" height="390" viewBox="0 0 820 390">
  <rect width="100%" height="100%" fill="white" />
  <line x1="60" y1="330" x2="760" y2="330" stroke="#222" stroke-width="1" />
  <line x1="60" y1="50" x2="60" y2="330" stroke="#222" stroke-width="1" />
  <text x="340" y="372" font-size="14">candidate index</text>
  <text x="18" y="210" font-size="14" transform="rotate(-90 18 210)">loss</text>
  <text x="60" y="28" font-size="14">loss range: {y_min:.6f} to {y_max:.6f}</text>
  <text x="60" y="350" font-size="12">{x_min:.0f}</text>
  <text x="722" y="350" font-size="12">{x_max:.0f}</text>
  <text x="8" y="55" font-size="12">{y_max:.4f}</text>
  <text x="8" y="334" font-size="12">{y_min:.4f}</text>
  <polyline fill="none" stroke="#9ecae1" stroke-width="1.0" points="{loss_points}" />
  <polyline fill="none" stroke="#d62728" stroke-width="2.0" points="{best_points}" />
  <line x1="610" y1="32" x2="650" y2="32" stroke="#9ecae1" stroke-width="2" />
  <text x="660" y="36" font-size="13">candidate loss</text>
  <line x1="610" y1="54" x2="650" y2="54" stroke="#d62728" stroke-width="2" />
  <text x="660" y="58" font-size="13">best loss</text>
</svg>
'''
    path.write_text(svg, encoding="utf-8")
