"""Patient-level random and temporal splits.

Implements 03_DATA_AND_COHORT §6 and CONTRIBUTING.md §5 (splits by ``subject_id``).
Output: ``data/interim/split/<hash>/splits.parquet`` with columns
``subject_id, split, split_temporal`` (``split_temporal`` is null for patients
outside both temporal groups).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from certivoi.config import SPLIT_NAMES, Config
from certivoi.data.cohort import build_cohort
from certivoi.utils.io import stage_dir
from certivoi.utils.logging import get_logger

log = get_logger("certivoi.split")


def assign_random(subjects: np.ndarray, fractions: dict[str, float], seed: int) -> np.ndarray:
    """Assign each subject to a split by seeded permutation; returns labels aligned to input.

    Implements 03_DATA_AND_COHORT §6 (main split). Boundaries are floor(cumsum*n);
    the last split takes the remainder.
    """
    subjects = np.asarray(subjects)
    n = len(subjects)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    labels = np.empty(n, dtype=object)
    names = [s for s in SPLIT_NAMES if fractions.get(s, 0) > 0]
    cuts = np.floor(np.cumsum([fractions[s] for s in names]) * n).astype(int)
    cuts[-1] = n
    start = 0
    for name, end in zip(names, cuts, strict=True):
        labels[perm[start:end]] = name
        start = end
    return labels


def assign_temporal(groups: pd.Series, fractions: dict[str, float], train_groups: list[str],
                    test_groups: list[str], seed: int) -> np.ndarray:
    """Temporal split: test = later year groups; train/val/calib from earlier groups.

    Earlier-period patients are split train/val/calib with the main fractions
    renormalised (test share removed). Others get ``None``.
    """
    labels = np.full(len(groups), None, dtype=object)
    early = groups.isin(train_groups).to_numpy()
    late = groups.isin(test_groups).to_numpy()
    labels[late] = "test"
    fr = {k: v for k, v in fractions.items() if k != "test"}
    tot = sum(fr.values())
    fr = {k: v / tot for k, v in fr.items()}
    idx = np.flatnonzero(early)
    if len(idx):
        labels[idx] = assign_random(idx, fr, seed)
    return labels


def split_path(cfg: Config) -> Path:
    return stage_dir(cfg, "split") / "splits.parquet"


def make_splits(cfg: Config, force: bool = False) -> Path:
    """Build and save splits for every subject with at least one repeat event."""
    out = split_path(cfg)
    if out.exists() and not force:
        log.info("split: cached at %s", out)
        return out
    co = build_cohort(cfg)
    subj = set()
    for lab in cfg.labs:
        subj |= set(pd.read_parquet(co[f"events_{lab}"], columns=["subject_id"])["subject_id"])
    adm = pd.read_parquet(co["admissions"], columns=["subject_id", "anchor_year_group"])
    pats = (adm.drop_duplicates("subject_id").set_index("subject_id")
            .loc[sorted(subj)].reset_index())
    fr = {k: float(v) for k, v in cfg.get("splits.fractions").items()}
    pats["split"] = assign_random(pats["subject_id"].to_numpy(), fr, cfg.seed)
    pats["split_temporal"] = assign_temporal(
        pats["anchor_year_group"], fr, list(cfg.get("splits.temporal.train_groups")),
        list(cfg.get("splits.temporal.test_groups")), cfg.seed)
    pats[["subject_id", "split", "split_temporal"]].to_parquet(out, index=False)
    log.info("split: %s", pats["split"].value_counts().to_dict())
    return out
