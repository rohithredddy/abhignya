from __future__ import annotations

import json
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from certivoi.config import Config  # noqa: E402
from certivoi.utils.io import (  # noqa: E402
    PrivacyError,
    Timer,
    interim_path,
    suppress_small_counts,
    write_results,
)
from certivoi.utils.logging import get_logger  # noqa: E402
from certivoi.utils.seed import set_all  # noqa: E402


def test_set_all_is_reproducible() -> None:
    g1 = set_all(7)
    a = (random.random(), np.random.rand(), g1.random())  # noqa: NPY002
    g2 = set_all(7)
    b = (random.random(), np.random.rand(), g2.random())  # noqa: NPY002
    assert a == b


def test_write_results_writes_csv_png_and_meta(tmp_cfg: Config) -> None:
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    table = pd.DataFrame({"lab": ["HGB"], "auroc": [0.8]})
    with Timer() as t:
        table = table.copy()
    assert t.elapsed >= 0.0
    out = write_results(tmp_cfg, "E0", {"summary": table}, {"fig": fig}, t.elapsed, note="test")
    plt.close(fig)
    assert (out / "summary.csv").exists() and (out / "fig.png").exists()
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    for key in ("config_hash", "git_commit", "data_mode", "seed", "runtime_sec", "sample_frac"):
        assert key in meta
    assert meta["config_hash"] == tmp_cfg.hash
    assert meta["data_mode"] == "synthetic"
    assert meta["note"] == "test"


@pytest.mark.parametrize("col", ["subject_id", "HADM_ID", "stay_id"])
def test_write_results_refuses_identifier_columns(tmp_cfg: Config, col: str) -> None:
    with pytest.raises(PrivacyError):
        write_results(tmp_cfg, "E0", {"bad": pd.DataFrame({col: [1], "x": [2]})})
    assert not (tmp_cfg.results_dir / "E0").exists()


def test_write_results_refuses_rows_names(tmp_cfg: Config) -> None:
    with pytest.raises(PrivacyError):
        write_results(tmp_cfg, "E0", {"rows_test": pd.DataFrame({"x": [1]})})


def test_suppress_small_counts() -> None:
    df = pd.DataFrame({"group": list("abcd"), "n": [0, 3, 10, 250]})
    out = suppress_small_counts(df, ["n"], min_n=10)
    assert out["n"].isna().tolist() == [False, True, False, False]
    assert out.loc[3, "n"] == 250


def test_interim_path_keyed_by_hash(tmp_cfg: Config) -> None:
    p = interim_path(tmp_cfg, "extract", "labs.parquet")
    assert tmp_cfg.hash in p.parts
    assert p.parent.exists()
    other = tmp_cfg.with_overrides({"data": {"sample_frac": 0.5}})
    assert interim_path(other, "extract", "labs.parquet") != p


def test_get_logger_idempotent() -> None:
    a = get_logger("certivoi.test")
    b = get_logger("certivoi.test")
    assert a is b and len(a.handlers) == 1
