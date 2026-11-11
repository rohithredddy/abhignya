"""Action maps: value (+ creatinine history) -> discrete action category.

Implements 02_METHODOLOGY §2 (action maps, KDIGO creatinine criteria) and §4
(category value intervals used by EDCP). Pure, vectorised functions.

Every lab is represented by per-event category *edges* ``E`` of shape (n, K+1)
with ``E[:,0] = -inf`` and ``E[:,K] = +inf``; category ``c`` covers
``[E[:,c], E[:,c+1])``. For interval labs the edges are constant; for
creatinine they depend on the 48 h / 7 d baselines of each event.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# Floating tolerance for "value equals threshold" (e.g. 1.0 + 0.3 vs 1.3). Lab values
# are reported to 0.1 or coarser, so 1e-9 never merges distinct reported values.
EPS = 1e-9


def interval_category(y: np.ndarray, boundaries: list[float],
                      at_boundary_upper: list[bool] | None = None) -> np.ndarray:
    """Category index for fixed boundaries.

    ``y`` exactly at boundary k goes to the upper category iff
    ``at_boundary_upper[k]`` (default True, i.e. left-closed intervals).
    NaN inputs give category -1.
    """
    y = np.asarray(y, dtype=float)
    upper = at_boundary_upper or [True] * len(boundaries)
    cat = np.zeros(y.shape, dtype=int)
    for b, up in zip(boundaries, upper, strict=True):
        at = np.abs(y - b) <= EPS
        cat += ((y > b) & ~at) | (at & up)
    return np.where(np.isnan(y), -1, cat)


def interval_edges(boundaries: list[float], n: int) -> np.ndarray:
    """Constant edges (n, K+1) for fixed boundaries."""
    row = np.array([-np.inf, *boundaries, np.inf], dtype=float)
    return np.tile(row, (n, 1))


@dataclass(frozen=True)
class KdigoParams:
    """KDIGO creatinine criteria (config ``actions.CREAT``)."""

    abs_rise_48h: float = 0.3
    ratio_1: float = 1.5
    ratio_2: float = 2.0
    ratio_3: float = 3.0
    stage3_abs: float = 4.0

    @classmethod
    def from_config(cls, d: dict[str, Any]) -> KdigoParams:
        r = d["ratio_7d"]
        return cls(float(d["abs_rise_48h"]), float(r[0]), float(r[1]), float(r[2]),
                   float(d["stage3_abs"]))


def kdigo_stage(y: np.ndarray, b48: np.ndarray, b7: np.ndarray, p: KdigoParams) -> np.ndarray:
    """KDIGO creatinine stage 0-3 (direct rule, mirrors MIMIC-code ``kdigo_creatinine``).

    Inputs: new value ``y``; ``b48`` / ``b7`` = minimum prior creatinine in the
    previous 48 h / 7 d (NaN if none). Stage 3 if y >= 3x b7, or y >= 4.0 with
    an acute rise (b48 <= 4.0 - 0.3, or y >= 1.5x b7); stage 2 if y >= 2x b7;
    stage 1 if y >= b48 + 0.3 or y >= 1.5x b7; else 0.
    """
    y, b48, b7 = (np.asarray(a, dtype=float) for a in (y, b48, b7))
    with np.errstate(invalid="ignore"):
        ge = lambda a, b: np.nan_to_num(a - b, nan=-np.inf) >= -EPS  # noqa: E731
        s3 = ge(y, p.ratio_3 * b7) | (
            ge(y, p.stage3_abs)
            & (ge(p.stage3_abs - p.abs_rise_48h, b48) | ge(y, p.ratio_1 * b7)))
        s2 = ge(y, p.ratio_2 * b7)
        s1 = ge(y, b48 + p.abs_rise_48h) | ge(y, p.ratio_1 * b7)
    return np.select([s3, s2, s1], [3, 2, 1], default=0)


def kdigo_edges(b48: np.ndarray, b7: np.ndarray, p: KdigoParams) -> np.ndarray:
    """Per-event creatinine edges (n, 5): stage(y) = #{k : y >= thr_k}.

    The stage is a non-decreasing step function of y for fixed baselines; its
    jump points are thr3 = min(3 b7, 4.0 if b48 <= 3.7 else max(4.0, 1.5 b7)),
    thr2 = min(2 b7, thr3), thr1 = min(b48 + 0.3, 1.5 b7, thr2). Missing
    baselines make the corresponding criterion unreachable (+inf).
    """
    b48, b7 = (np.asarray(a, dtype=float) for a in (b48, b7))
    inf = np.inf
    r3 = np.where(np.isnan(b7), inf, p.ratio_3 * b7)
    r2 = np.where(np.isnan(b7), inf, p.ratio_2 * b7)
    r1 = np.where(np.isnan(b7), inf, p.ratio_1 * b7)
    acute = np.where(np.isnan(b48), False, b48 <= p.stage3_abs - p.abs_rise_48h + EPS)
    abs3 = np.where(acute, p.stage3_abs, np.maximum(p.stage3_abs, r1))
    thr3 = np.minimum(r3, abs3)
    thr2 = np.minimum(r2, thr3)
    rise = np.where(np.isnan(b48), inf, b48 + p.abs_rise_48h)
    thr1 = np.minimum(np.minimum(rise, r1), thr2)
    n = len(b48)
    return np.column_stack([np.full(n, -inf), thr1, thr2, thr3, np.full(n, inf)])


def category_from_edges(y: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Category = number of finite inner edges with y >= edge (tolerance EPS)."""
    y = np.asarray(y, dtype=float)[:, None]
    inner = edges[:, 1:-1]
    return (y - inner >= -EPS).sum(axis=1)


@dataclass(frozen=True)
class ActionMap:
    """Action map for one lab built from config ``actions.<LAB>``."""

    lab: str
    names: list[str]
    boundaries: list[float] | None = None
    at_boundary_upper: list[bool] | None = None
    kdigo: KdigoParams | None = None

    @property
    def n_categories(self) -> int:
        return len(self.names)

    @property
    def no_action_index(self) -> int:
        """Index of the category whose name starts with ``no_`` (no action)."""
        for i, n in enumerate(self.names):
            if n.startswith("no_"):
                return i
        raise ValueError(f"{self.lab}: no 'no_*' category in {self.names}")

    def category(self, y: np.ndarray, b48: np.ndarray | None = None,
                 b7: np.ndarray | None = None) -> np.ndarray:
        if self.kdigo is not None:
            return kdigo_stage(y, b48, b7, self.kdigo)
        return interval_category(y, self.boundaries, self.at_boundary_upper)

    def edges(self, n: int, b48: np.ndarray | None = None,
              b7: np.ndarray | None = None) -> np.ndarray:
        if self.kdigo is not None:
            return kdigo_edges(b48, b7, self.kdigo)
        return interval_edges(self.boundaries, n)


def action_map(cfg_actions: dict[str, Any], lab: str,
               override: dict[str, Any] | None = None) -> ActionMap:
    """Build the :class:`ActionMap` for ``lab`` from the ``actions`` config section.

    ``override`` (e.g. ``actions.sensitivity_variants.HGB``) replaces boundaries.
    Names are regenerated generically if the override changes the category count.
    """
    spec = dict(cfg_actions[lab])
    if spec.get("rule") == "kdigo":
        return ActionMap(lab, list(spec["names"]), kdigo=KdigoParams.from_config(spec))
    names = list(spec["names"])
    if override:
        spec.update(override)
        if len(spec["boundaries"]) + 1 != len(names):
            names = [f"cat{i}" for i in range(len(spec["boundaries"]))] + ["no_action"]
            if lab == "K":  # middle category is "no action" for potassium
                names = ["low", "no_action", "high"][: len(spec["boundaries"]) + 1]
        if "at_boundary_upper" not in override:
            spec.pop("at_boundary_upper", None)
    return ActionMap(lab, names, boundaries=[float(b) for b in spec["boundaries"]],
                     at_boundary_upper=spec.get("at_boundary_upper"))
