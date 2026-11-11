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


@pytest.fixture(scope="session")
def pipe_cfg(cfg: Config, tmp_path_factory: pytest.TempPathFactory) -> Config:
    """Synthetic-mode config with raw MIMIC-layout CSVs written to a temp dir.

    Model settings are shrunk so the whole pipeline runs in seconds.
    """
    from synthetic import write_mimic_layout

    root = tmp_path_factory.mktemp("pipe")
    c = cfg.with_overrides({
        "data": {"mode": "synthetic", "root": str(root / "raw"),
                 "interim_dir": str(root / "interim"), "processed_dir": str(root / "proc"),
                 "results_dir": str(root / "results")},
        "compute": {"threads": 2},
        "model": {"lightgbm": {"n_estimators": 60, "min_child_samples": 20,
                               "early_stopping_rounds": 10}},
        "eval": {"bootstrap_reps": 20},
        "splits": {"crc_resamples": 10},
        "explain": {"shap_background_rows": 30, "shap_explain_rows": 60},
    })
    write_mimic_layout(make_synthetic_mimic(n_patients=120, seed=11, cfg=c), c.data_root)
    return c
