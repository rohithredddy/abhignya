"""Template text explanations (no LLM). Implements 02_METHODOLOGY §6."""

from __future__ import annotations

import numpy as np

from certivoi.triage import HIGH, LOW, POTENTIAL

TIER_TEXT = {
    LOW: "Low value (certified at α = {alpha:g})",
    POTENTIAL: "Potentially useful",
    HIGH: "High value",
}
LAB_NAMES = {"HGB": "Hgb", "K": "K", "CREAT": "Creatinine", "PLT": "Platelets"}


def _fmt(x: float) -> str:
    if not np.isfinite(x):
        return "n/a"
    return f"{x:.0f}" if abs(x) >= 100 else f"{x:.1f}" if abs(x) >= 1 else f"{x:.2f}"


VERBS = {"value": ("raises the forecast", "lowers the forecast"),
         "uncertainty": ("widens the range", "narrows the range")}


def reasons(phi_row: np.ndarray, values: np.ndarray, names: list[str],
            descriptions: dict[str, str], k: int = 3, kind: str = "value") -> list[str]:
    """Top-k features by |φ| as short phrases with the feature value and direction.

    ``kind='value'`` for φ^med (moves the median), ``'uncertainty'`` for φ^unc
    (changes the 80% interval width).
    """
    up, down = VERBS[kind]
    order = np.argsort(-np.abs(phi_row))[:k]
    out = []
    for i in order:
        desc = descriptions.get(names[i], names[i])
        out.append(f"{desc} = {_fmt(values[i])} ({up if phi_row[i] > 0 else down})")
    return out


def render(lab: str, unit: str, tier: str, alpha: float, y_hat: float, q_lo: float,
           q_hi: float, margin: float, boundary: float, p_change: float,
           reasons_med: list[str], reasons_unc: list[str]) -> str:
    """One explanation paragraph for an event."""
    name = LAB_NAMES.get(lab, lab)
    head = TIER_TEXT[tier].format(alpha=alpha)
    if np.isfinite(boundary):
        side = "inside" if margin >= 0 else "outside"
        pos = (f"{_fmt(abs(margin))} {unit} {side} the current action range "
               f"(nearest threshold {_fmt(boundary)} {unit})")
    else:
        pos = "no action threshold nearby"
    return (f"{head}. Predicted {name} {_fmt(y_hat)} {unit} (80% range {_fmt(q_lo)}–"
            f"{_fmt(q_hi)}), {pos}; calibrated P(action change) ≈ {p_change:.2f}. "
            f"Why the value: {'; '.join(reasons_med)}. "
            f"Why the uncertainty: {'; '.join(reasons_unc)}. "
            "Decision support only — not a diagnosis.")
