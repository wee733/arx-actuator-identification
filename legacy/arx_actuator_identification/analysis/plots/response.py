from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np


def _polyline_points(
    t: np.ndarray,
    y: np.ndarray,
    x_min: float,
    x_scale: float,
    y_min: float,
    y_scale: float,
    y_base: float,
) -> str:
    points = []
    for x_value, y_value in zip(t, y):
        if not np.isfinite(x_value) or not np.isfinite(y_value):
            continue
        x_px = 60.0 + (float(x_value) - x_min) * x_scale
        y_px = y_base - (float(y_value) - y_min) * y_scale
        points.append(f"{x_px:.2f},{y_px:.2f}")
    return " ".join(points)


def _range(values: list[np.ndarray]) -> tuple[float, float]:
    finite_y = np.concatenate(values)
    finite_y = finite_y[np.isfinite(finite_y)]
    if finite_y.size == 0:
        raise ValueError("plot data must contain finite values")
    y_min = float(np.min(finite_y))
    y_max = float(np.max(finite_y))
    if y_max <= y_min:
        y_max = y_min + 1.0
    y_margin = 0.08 * (y_max - y_min)
    return y_min - y_margin, y_max + y_margin


def _panel_svg(
    t: np.ndarray,
    series: dict[str, np.ndarray],
    colors: dict[str, str],
    x_min: float,
    x_max: float,
    top: float,
    bottom: float,
    ylabel: str,
) -> str:
    y_min, y_max = _range(list(series.values()))
    x_scale = 700.0 / (x_max - x_min)
    y_scale = (bottom - top) / (y_max - y_min)

    polylines = []
    legend = []
    for idx, (name, values) in enumerate(series.items()):
        points = _polyline_points(t, values, x_min, x_scale, y_min, y_scale, bottom)
        color = colors[name]
        polylines.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="1.8" points="{points}" />'
        )
        legend_y = top - 18 + idx * 18
        legend.append(
            f'<line x1="610" y1="{legend_y}" x2="650" y2="{legend_y}" stroke="{color}" stroke-width="2" />'
            f'<text x="660" y="{legend_y + 4}" font-size="12">{escape(name)}</text>'
        )
    return f'''
  <line x1="60" y1="{bottom}" x2="760" y2="{bottom}" stroke="#222" stroke-width="1" />
  <line x1="60" y1="{top}" x2="60" y2="{bottom}" stroke="#222" stroke-width="1" />
  <text x="18" y="{(top + bottom) / 2:.0f}" font-size="13" transform="rotate(-90 18 {(top + bottom) / 2:.0f})">{escape(ylabel)}</text>
  <text x="60" y="{top - 22:.0f}" font-size="13">range: {y_min:.4f} to {y_max:.4f}</text>
  <text x="8" y="{top + 5:.0f}" font-size="11">{y_max:.3f}</text>
  <text x="8" y="{bottom + 4:.0f}" font-size="11">{y_min:.3f}</text>
  {''.join(polylines)}
  {''.join(legend)}
'''


def plot_joint_response_svg(
    path: str | Path,
    t: np.ndarray,
    q_cmd: np.ndarray,
    q_real: np.ndarray,
    q_sim: np.ndarray,
    qd_real: np.ndarray | None = None,
    qd_sim: np.ndarray | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.asarray(t, dtype=float)
    finite_t = t[np.isfinite(t)]
    if finite_t.size == 0:
        raise ValueError("plot data must contain finite time values")
    x_min = float(np.min(finite_t))
    x_max = float(np.max(finite_t))
    if x_max <= x_min:
        x_max = x_min + 1.0

    q_series = {
        "q_cmd": np.asarray(q_cmd, dtype=float),
        "q_real": np.asarray(q_real, dtype=float),
        "q_sim": np.asarray(q_sim, dtype=float),
    }
    q_colors = {"q_cmd": "#555555", "q_real": "#1f77b4", "q_sim": "#d62728"}
    panels = [
        _panel_svg(t, q_series, q_colors, x_min, x_max, top=70.0, bottom=310.0, ylabel="q [rad]")
    ]
    height = 380
    x_axis_y = 310.0
    if qd_real is not None and qd_sim is not None:
        qd_series = {
            "qd_real": np.asarray(qd_real, dtype=float),
            "qd_sim": np.asarray(qd_sim, dtype=float),
        }
        qd_colors = {"qd_real": "#2ca02c", "qd_sim": "#ff7f0e"}
        panels.append(
            _panel_svg(t, qd_series, qd_colors, x_min, x_max, top=390.0, bottom=630.0, ylabel="qd [rad/s]")
        )
        height = 700
        x_axis_y = 630.0

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="820" height="{height}" viewBox="0 0 820 {height}">
  <rect width="100%" height="100%" fill="white" />
  {''.join(panels)}
  <text x="365" y="{height - 18}" font-size="14">time [s]</text>
  <text x="60" y="{x_axis_y + 20:.0f}" font-size="12">{x_min:.2f}</text>
  <text x="722" y="{x_axis_y + 20:.0f}" font-size="12">{x_max:.2f}</text>
</svg>
'''
    path.write_text(svg, encoding="utf-8")


def plot_joint_response(path: str | Path, t: np.ndarray, q_cmd: np.ndarray, q_real: np.ndarray, q_sim: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, q_cmd, label="q_cmd", linewidth=1.0)
    ax.plot(t, q_real, label="q_real", linewidth=1.0)
    ax.plot(t, q_sim, label="q_sim", linewidth=1.0)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("joint position [rad]")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


__all__ = ["plot_joint_response", "plot_joint_response_svg"]
