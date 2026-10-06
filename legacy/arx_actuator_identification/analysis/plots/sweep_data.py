from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np


def _range(values: list[np.ndarray]) -> tuple[float, float]:
    finite = np.concatenate([np.asarray(value, dtype=float).ravel() for value in values])
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("plot data must contain finite values")
    y_min = float(np.min(finite))
    y_max = float(np.max(finite))
    if y_max <= y_min:
        y_max = y_min + 1.0
    margin = 0.08 * (y_max - y_min)
    return y_min - margin, y_max + margin


def _polyline(t: np.ndarray, y: np.ndarray, x_min: float, x_scale: float, y_min: float, y_scale: float, y_base: float) -> str:
    points: list[str] = []
    for x_value, y_value in zip(t, y):
        if not np.isfinite(x_value) or not np.isfinite(y_value):
            continue
        x_px = 60.0 + (float(x_value) - x_min) * x_scale
        y_px = y_base - (float(y_value) - y_min) * y_scale
        points.append(f"{x_px:.2f},{y_px:.2f}")
    return " ".join(points)


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
    polylines: list[str] = []
    legend: list[str] = []
    for idx, (name, values) in enumerate(series.items()):
        color = colors[name]
        points = _polyline(t, values, x_min, x_scale, y_min, y_scale, bottom)
        polylines.append(f'<polyline fill="none" stroke="{color}" stroke-width="1.7" points="{points}" />')
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


def plot_sweep_data_svg(
    path: str | Path,
    t: np.ndarray,
    q_cmd: np.ndarray,
    q_real: np.ndarray,
    qd_real: np.ndarray,
    tracking_error: np.ndarray,
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

    panels = [
        _panel_svg(
            t,
            {"q_cmd": np.asarray(q_cmd, dtype=float), "q_real": np.asarray(q_real, dtype=float)},
            {"q_cmd": "#555555", "q_real": "#1f77b4"},
            x_min,
            x_max,
            top=70.0,
            bottom=260.0,
            ylabel="q [rad]",
        ),
        _panel_svg(
            t,
            {"qd_real": np.asarray(qd_real, dtype=float)},
            {"qd_real": "#2ca02c"},
            x_min,
            x_max,
            top=350.0,
            bottom=540.0,
            ylabel="qd [rad/s]",
        ),
        _panel_svg(
            t,
            {"q_real - q_cmd": np.asarray(tracking_error, dtype=float)},
            {"q_real - q_cmd": "#d62728"},
            x_min,
            x_max,
            top=630.0,
            bottom=820.0,
            ylabel="error [rad]",
        ),
    ]
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="820" height="890" viewBox="0 0 820 890">
  <rect width="100%" height="100%" fill="white" />
  {''.join(panels)}
  <text x="365" y="870" font-size="14">time [s]</text>
  <text x="60" y="840" font-size="12">{x_min:.2f}</text>
  <text x="722" y="840" font-size="12">{x_max:.2f}</text>
</svg>
'''
    path.write_text(svg, encoding="utf-8")


__all__ = ["plot_sweep_data_svg"]
