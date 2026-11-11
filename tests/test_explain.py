"""Phase 6: decomposition, SHAP additivity, templates, faithfulness, E5."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from certivoi.config import Config
from certivoi.experiments import e5
from certivoi.explain.decompose import decompose, nearest_boundary
from certivoi.explain.faithfulness import deletion_test
from certivoi.explain.shap_attr import tree_shap
from certivoi.explain.templates import reasons, render
from certivoi.models.train import load_lab_frame, load_models, train_lab
from certivoi.triage import HIGH, LOW, POTENTIAL


def test_nearest_boundary_signed() -> None:
    edges = np.array([[-np.inf, 7.0, 8.0, np.inf]] * 3)
    m, b = nearest_boundary(np.array([9.3, 7.2, 6.5]), edges, np.array([2, 1, 1]))
    np.testing.assert_allclose(m, [1.3, 0.2, -0.5])
    np.testing.assert_allclose(b, [8.0, 7.0, 7.0])


def test_decompose_rho() -> None:
    edges = np.array([[-np.inf, 7.0, 8.0, np.inf]])
    d = decompose(np.array([8.8]), np.array([9.3]), np.array([9.8]), edges, np.array([2]))
    assert d["margin"][0] == pytest.approx(1.3) and d["width"][0] == pytest.approx(1.0)
    assert d["rho"][0] == pytest.approx(1.3)


@pytest.fixture(scope="module")
def hgb(pipe_cfg: Config):
    train_lab(pipe_cfg, "HGB")
    bundle = load_models(pipe_cfg, "HGB")
    frame, _ = load_lab_frame(pipe_cfg, "HGB")
    return bundle["models"]["ours"], bundle["features"], frame


def test_shap_additivity(hgb) -> None:
    fq, feats, frame = hgb
    te, tr = frame[frame["split"] == "test"], frame[frame["split"] == "train"]
    attr = tree_shap(fq, te[feats].head(80), tr[feats].head(40))
    assert max(attr.additivity_error().values()) <= 1e-3
    assert attr.unc(0.1, 0.9).shape == attr.med.shape


def test_faithfulness_top_beats_random(hgb) -> None:
    fq, feats, frame = hgb
    te, tr = frame[frame["split"] == "test"].head(80), frame[frame["split"] == "train"].head(40)
    attr = tree_shap(fq, te[feats], tr[feats])
    from certivoi.models.train import edge_matrix
    imp = np.abs(attr.med) + np.abs(attr.unc(0.1, 0.9))
    out = deletion_test(fq, te[feats], te["y_last"].to_numpy(), edge_matrix(te),
                        te["a0"].to_numpy(), imp, tr[feats], [1, 3], 0)
    assert (out["mean_abs_delta_top"] >= out["mean_abs_delta_random"]).all()


@pytest.mark.parametrize("tier", [LOW, POTENTIAL, HIGH])
def test_template_renders(tier: str) -> None:
    names = ["last", "slope_48h", "WBC_last"]
    desc = {"last": "last prior value", "slope_48h": "slope over 48 h"}
    r_med = reasons(np.array([0.5, -0.2, 0.1]), np.array([9.4, -0.05, 8.0]), names, desc)
    r_unc = reasons(np.array([0.1, 0.3, -0.2]), np.array([9.4, -0.05, 8.0]), names, desc,
                    kind="uncertainty")
    text = render("HGB", "g/dL", tier, 0.01, 9.3, 8.8, 9.8, 1.3, 8.0, 0.02, r_med, r_unc)
    assert "Predicted Hgb 9.3 g/dL" in text and "8.0 g/dL" in text
    assert "raises the forecast" in r_med[0] and "widens the range" in r_unc[0]
    assert "not a diagnosis" in text
    if tier == LOW:
        assert "certified at α = 0.01" in text


def test_e5_outputs(pipe_cfg: Config) -> None:
    t = e5.run(pipe_cfg)
    assert t["shap_additivity"]["passes"].all()
    assert set(t["table6_faithfulness"]["k"]) == set(pipe_cfg.get("explain.faithfulness_k"))
    out = pipe_cfg.results_dir / "E5"
    assert (out / "fig7_global_importance.png").exists()
    assert (out / "examples.txt").read_text(encoding="utf-8").startswith("Example")
    for f in out.glob("*.csv"):
        assert "subject_id" not in pd.read_csv(f).columns


def test_make_shap_consistent_rewrite() -> None:
    """Thresholds become float32 floors; missing-type-None splits default to where 0 goes;
    predictions on float32-exact inputs (with unseen NaNs) are unchanged.

    The end-to-end regression guard is test_shap_additivity / test_e5_outputs, which
    failed on pipeline data before this rewrite existed.
    """
    import lightgbm as lgb

    from certivoi.models.quantile import float32_floor, make_shap_consistent

    rng = np.random.default_rng(0)
    X = (np.round(rng.uniform(-5, 50, size=(3000, 5)), 2)
         .astype(np.float32).astype(np.float64))
    y = X[:, 0] * 0.1 + np.sin(X[:, 1]) + rng.normal(0, 0.3, 3000)
    m = lgb.LGBMRegressor(objective="quantile", alpha=0.5, n_estimators=80,
                          min_child_samples=5, verbose=-1).fit(X, y)
    fixed = make_shap_consistent(m.booster_)
    tdf = fixed.trees_to_dataframe()
    splits = tdf[tdf["split_feature"].notna()]
    thr = splits["threshold"].to_numpy(dtype=float)
    assert np.array_equal(thr, thr.astype(np.float32).astype(np.float64))
    none_type = splits["missing_type"].isna() | (splits["missing_type"] == "None")
    goes_left = splits["missing_direction"] == "left"
    assert ((0.0 <= thr) == goes_left.to_numpy())[none_type.to_numpy()].all()
    Xq = X[:500].copy()
    Xq[rng.random(Xq.shape) < 0.1] = np.nan
    np.testing.assert_allclose(fixed.predict(Xq), m.booster_.predict(Xq), atol=1e-12)
    assert float32_floor(21.770000000000003) == float(np.float32(21.76999855))
    assert float32_floor(float(np.float32(7.65))) == float(np.float32(7.65))
