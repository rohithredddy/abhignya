from __future__ import annotations

import subprocess
import sys

import pytest

from conftest import REPO_ROOT

RUN = REPO_ROOT / "scripts" / "run.py"


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(RUN), *args], capture_output=True, text=True,
                          cwd=REPO_ROOT, timeout=120)


def test_help() -> None:
    r = _run("--help")
    assert r.returncode == 0
    for cmd in ("extract", "cohort", "labels", "features", "train", "exp", "all"):
        assert cmd in r.stdout


def test_info_shows_hash_and_sample_frac() -> None:
    r = _run("info", "--config", "configs/default.yaml", "--sample-frac", "0.05")
    assert r.returncode == 0, r.stderr
    assert "config_hash:" in r.stdout and "sample_frac: 0.05" in r.stdout


@pytest.mark.parametrize("args", [("exp", "E3")])
def test_unimplemented_stages_exit_cleanly(args: tuple[str, ...]) -> None:
    r = _run(*args)
    assert r.returncode == 2
    assert "not implemented yet" in r.stderr


def test_bad_lab_rejected() -> None:
    assert _run("train", "--lab", "NA").returncode != 0
