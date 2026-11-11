"""Phase 4: model training end-to-end on synthetic data, metrics, E2 outputs."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from certivoi.config import Config
from certivoi.eval.metrics import bootstrap_ci, ece, metric_fns
from certivoi.experiments import e2
from certivoi.models.baselines import b1_rule_score
from certivoi.models.train import train_lab


@pytest.fixture(scope="module")
def preds(pipe_cfg: Config) -> dict[str, pd.DataFrame]:
    return {lab: pd.read_parquet(train_lab(pipe_cfg, lab)) for lab in pipe_cfg.labs}


def test_predictions_cover_all_events_and_scores_valid(preds) -> None:
    for lab, df in preds.items():
        assert df["event_id"].is_unique
        assert set(df["split"]) == {"train", "val", "calib", "test"}
        for c in [c for c in df.columns if c.startswith("score_")] + ["edcp_raw"]:
            assert df[c].between(0, 1).all(), (lab, c)
        q = df[[c for c in df.columns if c.startswith("q_")]].to_numpy()
        assert (np.diff(q, axis=1) >= 0).all(), lab


def test_model_beats_chance_on_synthetic(preds) -> None:
    from sklearn.metrics import roc_auc_score
    for lab in ("HGB", "K", "CREAT"):
        te = preds[lab][preds[lab]["split"] == "test"]
        assert roc_auc_score(te["D"], te["score_ours"]) > 0.6, lab


def test_b1_rule() -> None:
    f = pd.DataFrame({"a0": [2, 2, 2, 1, 2], "delta_last2": [0.1, 0.9, 0.1, 0.1, np.nan],
                      "hours_since_last": [6, 6, 30, 6, 6]})
    assert list(b1_rule_score(f, 2, 0.5, 24)) == [0, 1, 1, 1, 1]


def test_ece_perfect_and_bad() -> None:
    y = np.array([0, 1] * 50)
    assert ece(y, np.full(100, 0.5)) == pytest.approx(0.0)
    assert ece(y, y.astype(float)) == pytest.approx(0.0)
    assert ece(np.zeros(10), np.ones(10)) == pytest.approx(1.0)


def test_bootstrap_ci_contains_estimate() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    s = np.clip(y * 0.3 + rng.random(400) * 0.7, 0, 1)
    out = bootstrap_ci(y, {"m": s}, np.repeat(np.arange(100), 4), metric_fns(15), 50, 0)
    est, lo, hi = out["m"]["auroc"]
    assert lo <= est <= hi


def test_e2_outputs(pipe_cfg: Config, preds) -> None:
    t3 = e2.run(pipe_cfg)
    assert set(t3["model"]) == {"B2", "B3", "B4", "B5", "ours"}
    out = pipe_cfg.results_dir / "E2"
    for f in ("table3_performance.csv", "forecaster.csv", "fig3_reliability.png",
              "hyperparams.json", "meta.json"):
        assert (out / f).exists(), f
    hp = json.loads((out / "hyperparams.json").read_text())
    assert "quantile_best_iterations" in hp["HGB"]
    fc = pd.read_csv(out / "forecaster.csv")
    assert {"coverage_80", "coverage_90"} <= set(fc["metric"])
