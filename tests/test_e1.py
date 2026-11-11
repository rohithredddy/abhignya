from __future__ import annotations

import pandas as pd

from certivoi.config import Config
from certivoi.data.cohort import build_cohort
from certivoi.data.extract import extract
from certivoi.experiments import e1


def test_e1_outputs(pipe_cfg: Config) -> None:
    extract(pipe_cfg)
    build_cohort(pipe_cfg)
    summary = e1.run(pipe_cfg)
    assert set(summary["lab"]) == {"HGB", "K", "CREAT", "PLT"}
    assert summary["prev_D"].between(0, 1).all()
    out = pipe_cfg.results_dir / "E1"
    assert (out / "table2_labels.csv").exists() and (out / "fig2_proxy_disagreement.png").exists()
    cont = pd.read_csv(out / "contingency.csv")
    assert "subject_id" not in cont.columns
