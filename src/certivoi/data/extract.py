"""Raw MIMIC-IV CSV.gz -> filtered Parquet with DuckDB.

Implements 03_DATA_AND_COHORT §3, §4, §8 and CONTRIBUTING.md §6 (DuckDB for raw
labevents; never loaded into pandas). Outputs (under ``data/interim/extract/<hash>/``):
``patients``, ``admissions``, ``icustays``, ``d_labitems``, ``labs`` (.parquet).
Leakage-only columns (diagnoses, DRG, discharge location, expire flag) are not
extracted; ``dischtime`` is kept only for the cohort LOS filter and patient-day
denominators, never as a feature.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from certivoi.config import Config
from certivoi.utils.io import stage_dir
from certivoi.utils.logging import get_logger

log = get_logger("certivoi.extract")

TABLES = ("patients", "admissions", "icustays", "d_labitems", "labs")

_LAB_TYPES = {
    "labevent_id": "BIGINT", "subject_id": "BIGINT", "hadm_id": "BIGINT", "itemid": "BIGINT",
    "charttime": "TIMESTAMP", "valuenum": "DOUBLE", "valueuom": "VARCHAR",
    "ref_range_lower": "DOUBLE", "ref_range_upper": "DOUBLE", "flag": "VARCHAR",
}


def connect(cfg: Config) -> duckdb.DuckDBPyConnection:
    """DuckDB connection using ``compute.threads`` and the RAM budget."""
    con = duckdb.connect()
    con.execute(f"SET threads TO {cfg.threads}")
    con.execute(f"SET memory_limit = '{int(cfg.get('compute.max_ram_gb'))}GB'")
    return con


def _raw(root: Path, module: str, table: str) -> str:
    """Path to a raw table, accepting .csv.gz or .csv."""
    for ext in (".csv.gz", ".csv"):
        p = root / module / f"{table}{ext}"
        if p.exists():
            return p.as_posix()
    raise FileNotFoundError(f"missing raw table {module}/{table}(.csv.gz) under {root}")


def all_lab_itemids(cfg: Config) -> list[int]:
    """Target + cross-lab feature itemids."""
    return sorted({s.itemid for s in cfg.labs.values()} | set(cfg.feature_labs.values()))


def extract_outputs(cfg: Config) -> dict[str, Path]:
    d = stage_dir(cfg, "extract")
    return {t: d / f"{t}.parquet" for t in TABLES}


def extract(cfg: Config, force: bool = False) -> dict[str, Path]:
    """Run extraction (skip if cached). Output: table name -> Parquet path.

    ``data.sample_frac`` < 1 keeps a seeded random subset of patients.
    """
    out = extract_outputs(cfg)
    if not force and all(p.exists() for p in out.values()):
        log.info("extract: cached at %s", out["labs"].parent)
        return out
    if cfg.data_mode == "synthetic":
        ensure_synthetic_raw(cfg)
    root = cfg.data_root
    con = connect(cfg)

    pats = con.execute(
        f"SELECT subject_id, gender, anchor_age, anchor_year, anchor_year_group "
        f"FROM read_csv_auto('{_raw(root, 'hosp', 'patients')}')"
    ).df()
    if cfg.sample_frac < 1.0:
        rng = np.random.default_rng(cfg.seed)
        n = max(1, int(round(cfg.sample_frac * len(pats))))
        pats = pats.iloc[np.sort(rng.choice(len(pats), size=n, replace=False))]
    con.register("pats_df", pats)
    con.execute(f"COPY (SELECT * FROM pats_df) TO '{out['patients'].as_posix()}' (FORMAT PARQUET)")

    con.execute(f"""
        COPY (SELECT subject_id, hadm_id, admittime, dischtime, admission_type, race, insurance
              FROM read_csv_auto('{_raw(root, 'hosp', 'admissions')}')
              WHERE subject_id IN (SELECT subject_id FROM pats_df))
        TO '{out['admissions'].as_posix()}' (FORMAT PARQUET)""")
    con.execute(f"""
        COPY (SELECT subject_id, hadm_id, stay_id, intime, outtime
              FROM read_csv_auto('{_raw(root, 'icu', 'icustays')}')
              WHERE subject_id IN (SELECT subject_id FROM pats_df))
        TO '{out['icustays'].as_posix()}' (FORMAT PARQUET)""")
    con.execute(f"""
        COPY (SELECT itemid, label, fluid, category
              FROM read_csv_auto('{_raw(root, 'hosp', 'd_labitems')}'))
        TO '{out['d_labitems'].as_posix()}' (FORMAT PARQUET)""")

    ids = ", ".join(str(i) for i in all_lab_itemids(cfg))
    types = "{" + ", ".join(f"'{k}': '{v}'" for k, v in _LAB_TYPES.items()) + "}"
    con.execute(f"""
        COPY (SELECT labevent_id, subject_id, hadm_id, itemid, charttime, valuenum, valueuom,
                     ref_range_lower, ref_range_upper, flag
              FROM read_csv('{_raw(root, 'hosp', 'labevents')}', header=true,
                            auto_detect=true, types={types})
              WHERE itemid IN ({ids}) AND hadm_id IS NOT NULL
                AND subject_id IN (SELECT subject_id FROM pats_df))
        TO '{out['labs'].as_posix()}' (FORMAT PARQUET)""")
    n_labs = con.execute(f"SELECT count(*) FROM '{out['labs'].as_posix()}'").fetchone()[0]
    log.info("extract: %d patients, %d lab rows -> %s", len(pats), n_labs, out["labs"].parent)
    con.close()
    return out


def ensure_synthetic_raw(cfg: Config, force: bool = False) -> None:
    """In ``data.mode: synthetic``, generate MIMIC-layout CSVs into ``data.root`` if missing."""
    from certivoi.data.synthetic import make_synthetic_mimic, write_mimic_layout

    marker = cfg.data_root / "hosp" / "labevents.csv.gz"
    if marker.exists() and not force:
        return
    tables = make_synthetic_mimic(
        n_patients=int(cfg.get("synthetic.n_patients")), seed=int(cfg.get("synthetic.seed")),
        cfg=cfg,
    )
    write_mimic_layout(tables, cfg.data_root)
    log.info("synthetic raw data written to %s", cfg.data_root)


def read_parquet(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Small helper: read selected columns of a Parquet file into pandas."""
    return pd.read_parquet(path, columns=columns)
