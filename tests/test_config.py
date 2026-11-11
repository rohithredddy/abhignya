from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from certivoi.config import Config, ConfigError, LabSpec, hash_dict, load_config
from synthetic import DEFAULT_CONFIG


def test_default_config_loads(cfg: Config) -> None:
    assert cfg.data_mode in ("demo", "full", "synthetic")
    assert set(cfg.labs) == {"HGB", "K", "CREAT", "PLT"}
    assert cfg.labs["K"] == LabSpec("K", 50971, "mEq/L", (1.0, 10.0), 0.3, (3.3, 5.1))
    assert cfg.threads >= 1
    assert all(0 < a < 1 for a in cfg.alphas)
    assert len(cfg.hash) == 12


def test_hash_is_stable_and_key_order_independent(cfg: Config) -> None:
    assert load_config(DEFAULT_CONFIG).hash == cfg.hash
    assert hash_dict({"a": 1, "b": {"c": 2}}) == hash_dict({"b": {"c": 2}, "a": 1})


def test_override_changes_hash_and_value(cfg: Config) -> None:
    c2 = load_config(DEFAULT_CONFIG, overrides={"data": {"sample_frac": 0.05}})
    assert c2.sample_frac == 0.05
    assert c2.hash != cfg.hash
    assert c2.labs == cfg.labs  # deep merge keeps untouched sections


def test_override_file(tmp_path: Path) -> None:
    p = tmp_path / "o.yaml"
    p.write_text(yaml.safe_dump({"compute": {"threads": 2}}), encoding="utf-8")
    c = load_config(DEFAULT_CONFIG, override_path=p)
    assert c.threads == 2
    assert c.seed == load_config(DEFAULT_CONFIG).seed


def test_get_dotted(cfg: Config) -> None:
    assert cfg.get("actions.HGB.boundaries") == [7.0, 8.0]
    assert cfg.get("no.such.key", None) is None
    with pytest.raises(ConfigError):
        cfg.get("no.such.key")


def test_raw_is_a_copy(cfg: Config) -> None:
    raw = cfg.raw
    raw["project"]["seed"] = -1
    assert cfg.seed != -1


@pytest.mark.parametrize(
    "override",
    [
        {"data": {"mode": "production"}},
        {"data": {"sample_frac": 0.0}},
        {"data": {"sample_frac": 1.5}},
        {"compute": {"threads": 0}},
        {"splits": {"fractions": {"train": 0.7, "val": 0.1, "calib": 0.15, "test": 0.15}}},
        {"conformal": {"alphas": [0.0]}},
        {"labs": {"HGB": {"plausible": [25.0, 2.0]}}},
    ],
)
def test_invalid_values_rejected(cfg: Config, override: dict) -> None:
    with pytest.raises(ConfigError):
        cfg.with_overrides(override)
