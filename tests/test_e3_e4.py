"""Phase 5 experiments E3 / E4 on synthetic data (aggregate outputs only)."""

from __future__ import annotations

import pandas as pd

from certivoi.config import Config
from certivoi.experiments import e3, e4


def test_e3_outputs(pipe_cfg: Config) -> None:
    t4 = e3.run(pipe_cfg)
    assert set(t4["alpha"]) == set(pipe_cfg.alphas)
    assert t4["reduction_test"].between(0, 1).all()
    # CRC: mean realised risk over resamples <= alpha (+ small finite-sample slack)
    assert (t4["resample_risk_mean"] <= t4["alpha"] * 1.5 + 0.002).all()
    out = pipe_cfg.results_dir / "E3"
    for f in ("table4_certified.csv", "ltt.csv", "natural_thresholds.csv",
              "crc_all_scores.csv", "tiers.csv", "fig4_risk_reduction.png",
              "fig5_risk_resamples.png"):
        assert (out / f).exists(), f
    tiers = pd.read_csv(out / "tiers.csv")
    s = tiers.groupby(["lab", "alpha"])["share"].sum()
    assert ((s - 1).abs() < 1e-9).all()
    for f in out.glob("*.csv"):
        assert "subject_id" not in pd.read_csv(f).columns


def test_e4_outputs(pipe_cfg: Config) -> None:
    r = e4.run(pipe_cfg)
    sub = r["subgroups"]
    assert set(sub["method"]) == {"global", "mondrian"}
    assert set(sub["group_var"]) == set(pipe_cfg.get("conformal.mondrian_groups"))
    temp = r["temporal"]
    assert set(temp["setting"]) == {"in_period", "shifted", "recalibrated"}
    sens = r["sensitivity"]
    assert set(sens["lab"]) == {"HGB", "K"}
    assert (pipe_cfg.results_dir / "E4" / "fig6_temporal_shift.png").exists()
