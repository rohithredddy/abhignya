"""Shared pytest fixtures (synthetic data only, CONTRIBUTING.md §4.5)."""

from __future__ import annotations

from pathlib import Path

import pytest

from certivoi.config import Config, load_config
from synthetic import DEFAULT_CONFIG, make_synthetic_mimic

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def cfg() -> Config:
    return load_config(DEFAULT_CONFIG)


@pytest.fixture
def tmp_cfg(cfg: Config, tmp_path: Path) -> Config:
    """Default config with all output dirs redirected into a temp folder."""
    return cfg.with_overrides({"data": {
        "mode": "synthetic",
        "root": str(tmp_path / "raw"),
        "interim_dir": str(tmp_path / "interim"),
        "processed_dir": str(tmp_path / "processed"),
        "results_dir": str(tmp_path / "results"),
    }})


@pytest.fixture(scope="session")
def synth(cfg: Config) -> dict:
    return make_synthetic_mimic(n_patients=60, seed=123, cfg=cfg)
