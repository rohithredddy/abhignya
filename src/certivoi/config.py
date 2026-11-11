"""Configuration loading with typed access and a stable config hash.

Implements 05_IMPLEMENTATION_ROADMAP Phase 0 (config) and CONTRIBUTING.md §7
(config-driven, no magic numbers). All thresholds, alphas and grids live in
``configs/default.yaml``; code reads them through :class:`Config`.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DATA_MODES = ("demo", "full", "synthetic")
TARGET_LABS = ("HGB", "K", "CREAT", "PLT")
SPLIT_NAMES = ("train", "val", "calib", "test")
_SPLIT_SUM_TOL = 1e-9


class ConfigError(ValueError):
    """Raised when a configuration is missing keys or has invalid values."""


@dataclass(frozen=True)
class LabSpec:
    """One target lab: MIMIC itemid, expected unit, plausibility bounds, stability delta."""

    code: str
    itemid: int
    unit: str
    plausible: tuple[float, float]
    stability_delta: float


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return a new dict: ``base`` recursively updated by ``override``."""
    out = copy.deepcopy(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def hash_dict(d: dict[str, Any], length: int = 12) -> str:
    """Stable short SHA-256 hash of a JSON-serialisable dict (key order independent)."""
    canon = json.dumps(d, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:length]


class Config:
    """Immutable view over the resolved configuration dict.

    Inputs: a resolved config dict (use :func:`load_config` to build one).
    Exposes typed properties for frequently used values and :meth:`get`
    for dotted-path access to anything else.
    """

    def __init__(self, raw: dict[str, Any], source: str | None = None) -> None:
        self._raw = copy.deepcopy(raw)
        self.source = source
        _validate(self._raw)
        self.hash = hash_dict(self._raw)

    # ---- generic access -------------------------------------------------
    @property
    def raw(self) -> dict[str, Any]:
        """Deep copy of the resolved config dict."""
        return copy.deepcopy(self._raw)

    def get(self, dotted: str, default: Any = ...) -> Any:
        """Look up ``a.b.c``; raise ConfigError if missing and no default given."""
        node: Any = self._raw
        for part in dotted.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            elif default is ...:
                raise ConfigError(f"missing config key: {dotted}")
            else:
                return default
        return copy.deepcopy(node)

    # ---- typed shortcuts ------------------------------------------------
    @property
    def seed(self) -> int:
        return int(self._raw["project"]["seed"])

    @property
    def data_mode(self) -> str:
        return str(self._raw["data"]["mode"])

    @property
    def is_full(self) -> bool:
        """True when running on credentialed MIMIC-IV (privacy rules apply)."""
        return self.data_mode == "full"

    @property
    def sample_frac(self) -> float:
        return float(self._raw["data"]["sample_frac"])

    @property
    def threads(self) -> int:
        return int(self._raw["compute"]["threads"])

    @property
    def data_root(self) -> Path:
        return Path(self._raw["data"]["root"])

    @property
    def interim_dir(self) -> Path:
        return Path(self._raw["data"]["interim_dir"])

    @property
    def processed_dir(self) -> Path:
        return Path(self._raw["data"]["processed_dir"])

    @property
    def results_dir(self) -> Path:
        return Path(self._raw["data"]["results_dir"])

    @property
    def alphas(self) -> list[float]:
        return [float(a) for a in self._raw["conformal"]["alphas"]]

    @property
    def labs(self) -> dict[str, LabSpec]:
        out: dict[str, LabSpec] = {}
        for code, spec in self._raw["labs"].items():
            lo, hi = spec["plausible"]
            out[code] = LabSpec(
                code=code,
                itemid=int(spec["itemid"]),
                unit=str(spec["unit"]),
                plausible=(float(lo), float(hi)),
                stability_delta=float(spec["stability_delta"]),
            )
        return out

    @property
    def feature_labs(self) -> dict[str, int]:
        return {k: int(v) for k, v in self._raw["feature_labs"].items()}

    def with_overrides(self, override: dict[str, Any]) -> Config:
        """Return a new Config with ``override`` deep-merged in (hash recomputed)."""
        return Config(_deep_merge(self._raw, override), source=self.source)

    def __repr__(self) -> str:
        return f"Config(source={self.source!r}, mode={self.data_mode}, hash={self.hash})"


def _validate(raw: dict[str, Any]) -> None:
    """Fail loudly on missing sections or out-of-range values."""
    for section in ("project", "data", "compute", "labs", "actions", "splits", "conformal"):
        if section not in raw:
            raise ConfigError(f"missing config section: {section}")

    mode = raw["data"].get("mode")
    if mode not in DATA_MODES:
        raise ConfigError(f"data.mode must be one of {DATA_MODES}, got {mode!r}")

    frac = raw["data"].get("sample_frac", 1.0)
    if not (0.0 < float(frac) <= 1.0):
        raise ConfigError(f"data.sample_frac must be in (0, 1], got {frac}")

    threads = raw["compute"].get("threads")
    if not isinstance(threads, int) or threads < 1:
        raise ConfigError(f"compute.threads must be a positive int, got {threads!r}")

    missing = [lab for lab in TARGET_LABS if lab not in raw["labs"]]
    if missing:
        raise ConfigError(f"labs section missing target labs: {missing}")
    for code, spec in raw["labs"].items():
        lo, hi = spec["plausible"]
        if not lo < hi:
            raise ConfigError(f"labs.{code}.plausible must be [low, high] with low < high")

    fr = raw["splits"]["fractions"]
    if set(fr) != set(SPLIT_NAMES):
        raise ConfigError(f"splits.fractions must have keys {SPLIT_NAMES}")
    if abs(sum(float(v) for v in fr.values()) - 1.0) > _SPLIT_SUM_TOL:
        raise ConfigError("splits.fractions must sum to 1")

    for a in raw["conformal"]["alphas"]:
        if not (0.0 < float(a) < 1.0):
            raise ConfigError(f"conformal.alphas must lie in (0, 1), got {a}")


def load_config(
    path: str | Path,
    override_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> Config:
    """Load YAML config, optionally deep-merge an override YAML file and a dict.

    Inputs: base YAML path; optional override YAML path; optional dict overrides
    (applied last, e.g. CLI ``--sample-frac``). Output: validated :class:`Config`.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if override_path is not None:
        with Path(override_path).open("r", encoding="utf-8") as fh:
            raw = _deep_merge(raw, yaml.safe_load(fh) or {})
    if overrides:
        raw = _deep_merge(raw, overrides)
    return Config(raw, source=str(path))
