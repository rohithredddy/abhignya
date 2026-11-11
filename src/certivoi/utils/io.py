"""Result writing with ``meta.json`` and privacy guards; interim cache paths.

Implements CONTRIBUTING.md §4 (aggregate-only ``results/``; small-cell suppression),
§6 (interim cache keyed by config hash) and §7 (``meta.json`` per experiment).
"""

from __future__ import annotations

import json
import platform
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from certivoi.config import Config

# Columns that identify individual patients/events; never allowed in results/.
ID_COLUMNS = frozenset({"subject_id", "hadm_id", "stay_id", "labevent_id", "specimen_id"})


class PrivacyError(RuntimeError):
    """Raised when an output destined for results/ looks row-level."""


def git_commit() -> str:
    """Current git commit hash, or ``"unknown"`` outside a git repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=10
        )
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def suppress_small_counts(df: pd.DataFrame, count_cols: list[str], min_n: int) -> pd.DataFrame:
    """Mask count cells below ``min_n`` with NA (PhysioNet small-cell rule).

    Inputs: aggregate table, names of count columns, threshold (config
    ``eval.min_cell_count``). Output: copy with small counts set to NA.
    Zero counts are kept (they reveal no individual).
    """
    out = df.copy()
    for col in count_cols:
        vals = out[col]
        mask = (vals > 0) & (vals < min_n)
        out[col] = vals.astype("Int64").mask(mask)
    return out


def check_aggregate(df: pd.DataFrame, name: str) -> None:
    """Raise :class:`PrivacyError` if a table has patient/event identifier columns."""
    bad = ID_COLUMNS.intersection(c.lower() for c in df.columns)
    if bad:
        raise PrivacyError(
            f"table '{name}' has identifier columns {sorted(bad)}; per-event outputs "
            "belong under data/, never results/ (CONTRIBUTING.md §4.6)"
        )


def build_meta(cfg: Config, exp_id: str, runtime_sec: float, **extra: Any) -> dict[str, Any]:
    """Metadata dict stored as ``meta.json`` next to every experiment output."""
    meta: dict[str, Any] = {
        "exp_id": exp_id,
        "config_hash": cfg.hash,
        "config_source": cfg.source,
        "git_commit": git_commit(),
        "data_mode": cfg.data_mode,
        "seed": cfg.seed,
        "sample_frac": cfg.sample_frac,
        "runtime_sec": round(float(runtime_sec), 3),
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "python": platform.python_version(),
    }
    meta.update(extra)
    return meta


def write_results(
    cfg: Config,
    exp_id: str,
    tables: dict[str, pd.DataFrame] | None = None,
    figures: dict[str, Any] | None = None,
    runtime_sec: float = 0.0,
    **extra_meta: Any,
) -> Path:
    """Write aggregate tables (CSV), figures (PNG) and ``meta.json`` to results/<exp_id>/.

    Inputs: config, experiment id, ``{name: DataFrame}``, ``{name: matplotlib Figure}``,
    runtime in seconds, extra metadata. Output: the experiment directory.
    Refuses tables with identifier columns or names starting with ``rows``.
    """
    out_dir = cfg.results_dir / exp_id
    tables = tables or {}
    figures = figures or {}
    for name, df in tables.items():
        if name.lower().startswith("rows"):
            raise PrivacyError(f"table name '{name}' is reserved for row-level data")
        check_aggregate(df, name)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)
    for name, fig in figures.items():
        fig.savefig(out_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    meta = build_meta(cfg, exp_id, runtime_sec, **extra_meta)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out_dir


_BASE_KEYS = ["data.mode", "data.root", "data.sample_frac", "project.seed", "labs", "feature_labs"]
_COHORT_KEYS = [*_BASE_KEYS, "cohort", "data.max_events_per_lab"]
_LABEL_KEYS = [*_COHORT_KEYS, "actions"]
_SPLIT_KEYS = [*_COHORT_KEYS, "splits"]
# Config sections each pipeline stage depends on (its cache key).
STAGE_KEYS: dict[str, list[str]] = {
    "extract": _BASE_KEYS,
    "cohort": _COHORT_KEYS,
    "split": _SPLIT_KEYS,
    "labels": _LABEL_KEYS,
    "features": _LABEL_KEYS,
    "models": [*_LABEL_KEYS, "splits", "model", "baselines", "triage"],
}


def stage_hash(cfg: Config, stage: str) -> str:
    """Hash of only the config keys ``stage`` depends on (see ``STAGE_KEYS``)."""
    from certivoi.config import hash_dict

    return hash_dict({k: cfg.get(k, None) for k in STAGE_KEYS[stage]})


def stage_dir(cfg: Config, stage: str) -> Path:
    """Interim directory ``<interim_dir>/<stage>/<stage_hash>`` (created)."""
    p = cfg.interim_dir / stage / stage_hash(cfg, stage)
    p.mkdir(parents=True, exist_ok=True)
    return p


def interim_path(cfg: Config, stage: str, name: str) -> Path:
    """Cache location ``<interim_dir>/<config_hash>/<stage>/<name>`` (parents created)."""
    p = cfg.interim_dir / cfg.hash / stage / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


class Timer:
    """Context manager measuring wall-clock seconds in ``.elapsed``."""

    def __enter__(self) -> Timer:
        self._t0 = time.perf_counter()
        self.elapsed = 0.0
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed = time.perf_counter() - self._t0


def privacy_suppress(cfg: Config, df: pd.DataFrame, count_cols: list[str]) -> pd.DataFrame:
    """Apply small-cell suppression when ``data.mode`` is ``full`` (no-op otherwise)."""
    if not cfg.is_full:
        return df
    return suppress_small_counts(df, count_cols, int(cfg.get("eval.min_cell_count")))
