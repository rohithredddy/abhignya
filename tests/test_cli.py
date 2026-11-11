from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

from conftest import REPO_ROOT

RUN = REPO_ROOT / "scripts" / "run.py"


def _run(*args: str, timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(RUN), *args], capture_output=True, text=True,
                          cwd=REPO_ROOT, timeout=timeout)


def test_help() -> None:
    r = _run("--help")
    assert r.returncode == 0
    for cmd in ("extract", "verify", "cohort", "split", "labels", "features", "train", "exp",
                "all"):
        assert cmd in r.stdout


def test_info_shows_hash_and_sample_frac() -> None:
    r = _run("info", "--config", "configs/default.yaml", "--sample-frac", "0.05")
    assert r.returncode == 0, r.stderr
    assert "config_hash:" in r.stdout and "sample_frac: 0.05" in r.stdout


def test_bad_lab_rejected() -> None:
    assert _run("train", "--lab", "NA").returncode != 0
    assert _run("exp", "E9").returncode != 0


def test_all_end_to_end_synthetic(tmp_path: Path) -> None:
    """`run.py all` on a small synthetic dataset produces every experiment's outputs."""
    root = tmp_path.as_posix()
    override = {
        "data": {"mode": "synthetic", "root": f"{root}/raw", "interim_dir": f"{root}/interim",
                 "processed_dir": f"{root}/proc", "results_dir": f"{root}/results"},
        "synthetic": {"n_patients": 100, "seed": 3},
        "compute": {"threads": 2},
        "model": {"lightgbm": {"n_estimators": 40, "min_child_samples": 20,
                               "early_stopping_rounds": 10}},
        "eval": {"bootstrap_reps": 10},
        "splits": {"crc_resamples": 5},
        "explain": {"shap_background_rows": 20, "shap_explain_rows": 40},
    }
    ov = tmp_path / "override.yaml"
    ov.write_text(yaml.safe_dump(override), encoding="utf-8")
    r = _run("all", "--config", "configs/default.yaml", "--override", str(ov), timeout=900)
    assert r.returncode == 0, r.stderr[-3000:]
    res = tmp_path / "results"
    for exp in ("cohort", "verify_items", "features", "E1", "E2", "E3", "E4", "E5", "E6",
                "E7"):
        meta = json.loads((res / exp / "meta.json").read_text(encoding="utf-8"))
        assert meta["data_mode"] == "synthetic"
    t10 = pd.read_csv(res / "E7" / "table10_online.csv")
    assert {"static", "online_p0.2"} <= set(t10["method"].dropna())
    assert {"CBC", "BMP"} <= set(pd.read_csv(res / "E7" / "table9_panel.csv")["panel"])
    t7 = pd.read_csv(res / "E6" / "table7_ablations.csv")
    assert {"main", "no_isotonic", "target_absolute", "no_trend", "no_panel"} <= set(
        t7["variant"])
    mf = subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "make_figures.py"),
                         "--results", str(res), "--paper", str(tmp_path / "paper")],
                        capture_output=True, text=True, timeout=120)
    assert mf.returncode == 0, mf.stderr
    tex = (tmp_path / "paper" / "tables" / "table4_certified.tex").read_text(encoding="utf-8")
    assert r"\toprule" in tex and "source:" in tex
