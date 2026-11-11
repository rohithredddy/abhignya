"""Phase 8 demo back end (certivoi.demo) on synthetic data."""

from __future__ import annotations

import numpy as np
import pytest

from certivoi.demo import FullDataRefused, admissions, check_mode, explain_event, prepare_lab
from certivoi.triage import HIGH, LOW, POTENTIAL


def test_refuses_full_mode(cfg) -> None:
    with pytest.raises(FullDataRefused):
        check_mode(cfg.with_overrides({"data": {"mode": "full"}}))
    with pytest.raises(FullDataRefused):
        prepare_lab(cfg.with_overrides({"data": {"mode": "full"}}), "HGB")


def test_prepare_and_explain(pipe_cfg) -> None:
    view = prepare_lab(pipe_cfg, "K", alpha=0.05)
    ev = view.events
    assert len(ev) > 0 and set(ev["tier"]) <= {LOW, POTENTIAL, HIGH}
    assert ev["score"].between(0, 1).all()
    assert (ev.loc[ev["tier"] == LOW, "score"] <= view.lam).all()
    assert len(admissions(view)) == ev["hadm_id"].nunique()
    info = explain_event(view, 0)
    assert "Decision support only" in info["text"]
    assert info["q_lo"] <= info["y_hat"] <= info["q_hi"]
    assert info["thresholds"] == [3.5, 5.5]
    assert np.all(np.diff(np.sort(info["quantiles"])) >= 0)


def test_interval_figure(pipe_cfg) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from certivoi.demo import interval_figure

    view = prepare_lab(pipe_cfg, "HGB")
    fig = interval_figure(view, explain_event(view, 0))
    assert fig.axes
