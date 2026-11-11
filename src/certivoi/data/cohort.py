"""Cohort construction and flow table.

Implements 03_DATA_AND_COHORT §5 and 02_METHODOLOGY §1 (repeat events). All steps
run in DuckDB on the extracted Parquet. Outputs (``data/interim/cohort/<hash>/``):

- ``admissions.parquet``: cohort admissions joined with patient demographics.
- ``clean_labs.parquet``: all target + feature lab values that pass the
  filters (history source for labels/features).
- ``events_<LAB>.parquet``: repeat events per target lab (after optional cap).

Aggregate flow table -> ``results/cohort/flow.csv``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from certivoi.config import Config
from certivoi.data.extract import connect, extract
from certivoi.utils.io import stage_dir, suppress_small_counts, write_results
from certivoi.utils.logging import get_logger

log = get_logger("certivoi.cohort")


def cohort_outputs(cfg: Config) -> dict[str, Path]:
    d = stage_dir(cfg, "cohort")
    out = {"admissions": d / "admissions.parquet", "clean_labs": d / "clean_labs.parquet"}
    for lab in cfg.labs:
        out[f"events_{lab}"] = d / f"events_{lab}.parquet"
    return out


def _code_map_sql(cfg: Config) -> str:
    """SQL VALUES list mapping itemid -> lab code (target and feature labs)."""
    pairs = [(s.itemid, c) for c, s in cfg.labs.items()]
    pairs += [(i, c) for c, i in cfg.feature_labs.items()]
    return ", ".join(f"({i}, '{c}')" for i, c in pairs)


def cap_by_patient(counts: pd.Series, cap: int, seed: int) -> set[int]:
    """Patient-level random sampling so total events <= cap.

    Inputs: events per subject_id (index = subject_id), cap, seed.
    Output: set of kept subject_ids (all if total <= cap).
    """
    if counts.sum() <= cap:
        return set(counts.index.tolist())
    rng = np.random.default_rng(seed)
    order = counts.index.to_numpy()[rng.permutation(len(counts))]
    cum = counts.loc[order].cumsum().to_numpy()
    return set(order[cum <= cap].tolist())


def build_cohort(cfg: Config, force: bool = False) -> dict[str, Path]:
    """Run cohort steps; write per-lab repeat events and the flow table."""
    out = cohort_outputs(cfg)
    if not force and all(p.exists() for p in out.values()):
        log.info("cohort: cached at %s", out["admissions"].parent)
        return out
    ex = extract(cfg)
    con = connect(cfg)
    min_age = int(cfg.get("cohort.min_age"))
    min_hours = float(cfg.get("cohort.min_admission_hours"))
    lookback = float(cfg.get("cohort.repeat_lookback_hours"))
    gap = int(cfg.get("cohort.min_gap_minutes"))

    con.execute(f"""
        CREATE TABLE adm AS
        SELECT a.subject_id, a.hadm_id, a.admittime, a.dischtime, a.admission_type, a.race,
               a.insurance, p.gender, p.anchor_age, p.anchor_year, p.anchor_year_group,
               p.anchor_age >= {min_age} AS is_adult,
               date_diff('second', a.admittime, a.dischtime) / 3600.0 >= {min_hours} AS long_enough
        FROM '{ex['admissions'].as_posix()}' a
        JOIN '{ex['patients'].as_posix()}' p USING (subject_id)""")
    con.execute(f"CREATE TABLE codes(itemid BIGINT, code VARCHAR); "
                f"INSERT INTO codes VALUES {_code_map_sql(cfg)}")
    spec_rows = ", ".join(
        f"('{c}', '{s.unit}', {s.plausible[0]}, {s.plausible[1]}, "
        f"{(s.ref_range or (None, None))[0] or 'NULL'}, {(s.ref_range or (None, None))[1] or 'NULL'})"
        for c, s in cfg.labs.items())
    con.execute("CREATE TABLE spec(code VARCHAR, unit VARCHAR, lo DOUBLE, hi DOUBLE, "
                f"ref_lo DOUBLE, ref_hi DOUBLE); INSERT INTO spec VALUES {spec_rows}")

    # Step flags per lab row; feature labs have no unit/plausibility spec (only non-null).
    con.execute(f"""
        CREATE TABLE labs AS
        SELECT l.labevent_id, l.subject_id, l.hadm_id, c.code, l.itemid, l.charttime,
               l.valuenum, l.valueuom, l.flag,
               coalesce(l.ref_range_lower, s.ref_lo) AS ref_lo,
               coalesce(l.ref_range_upper, s.ref_hi) AS ref_hi,
               a.is_adult, a.long_enough,
               l.valuenum IS NOT NULL AS has_value,
               s.code IS NULL OR l.valueuom = s.unit AS unit_ok,
               s.code IS NULL OR (l.valuenum BETWEEN s.lo AND s.hi) AS plausible
        FROM '{ex['labs'].as_posix()}' l
        JOIN codes c USING (itemid)
        JOIN adm a USING (hadm_id)
        LEFT JOIN spec s ON s.code = c.code""")
    con.execute("""
        CREATE TABLE clean AS
        SELECT * EXCLUDE (rn) FROM (
            SELECT *, row_number() OVER (PARTITION BY hadm_id, itemid, charttime
                                         ORDER BY labevent_id) AS rn
            FROM labs
            WHERE is_adult AND long_enough AND has_value AND unit_ok AND plausible)
        WHERE rn = 1""")
    con.execute(f"""
        CREATE TABLE repeat_flag AS
        SELECT e.labevent_id, count(h.labevent_id) > 0 AS is_repeat
        FROM clean e LEFT JOIN clean h
          ON h.hadm_id = e.hadm_id AND h.itemid = e.itemid
         AND h.charttime < e.charttime - INTERVAL {gap} MINUTE
         AND h.charttime >= e.charttime - INTERVAL {int(lookback * 60)} MINUTE
        WHERE e.code IN ({", ".join(f"'{c}'" for c in cfg.labs)})
        GROUP BY e.labevent_id""")

    steps = [
        ("1_extracted_in_admission", "TRUE"),
        ("2_adult", "is_adult"),
        ("3_admission_ge_24h", "is_adult AND long_enough"),
        ("4_non_null_value", "is_adult AND long_enough AND has_value"),
        ("5_expected_unit", "is_adult AND long_enough AND has_value AND unit_ok"),
        ("6_plausible", "is_adult AND long_enough AND has_value AND unit_ok AND plausible"),
    ]
    flow: list[dict] = []
    for lab in cfg.labs:
        for name, cond in steps:
            n = con.execute(f"SELECT count(*), count(DISTINCT subject_id), "
                            f"count(DISTINCT hadm_id) FROM labs WHERE code='{lab}' AND {cond}"
                            ).fetchone()
            flow.append(dict(lab=lab, step=name, n_events=n[0], n_patients=n[1],
                             n_admissions=n[2]))
        n = con.execute(f"SELECT count(*), count(DISTINCT subject_id), count(DISTINCT hadm_id) "
                        f"FROM clean WHERE code='{lab}'").fetchone()
        flow.append(dict(lab=lab, step="7_deduplicated", n_events=n[0], n_patients=n[1],
                         n_admissions=n[2]))
        ev = con.execute(f"""
            SELECT c.labevent_id AS event_id, c.subject_id, c.hadm_id, c.charttime AS t,
                   c.valuenum AS y, c.ref_lo, c.ref_hi, c.flag
            FROM clean c JOIN repeat_flag r USING (labevent_id)
            WHERE c.code = '{lab}' AND r.is_repeat ORDER BY c.hadm_id, c.charttime""").df()
        flow.append(dict(lab=lab, step="8_repeat_events", n_events=len(ev),
                         n_patients=ev["subject_id"].nunique(),
                         n_admissions=ev["hadm_id"].nunique()))
        keep = cap_by_patient(ev.groupby("subject_id").size(),
                              int(cfg.get("data.max_events_per_lab")), cfg.seed)
        ev = ev[ev["subject_id"].isin(keep)].reset_index(drop=True)
        flow.append(dict(lab=lab, step="9_after_cap", n_events=len(ev),
                         n_patients=ev["subject_id"].nunique(),
                         n_admissions=ev["hadm_id"].nunique()))
        ev.to_parquet(out[f"events_{lab}"], index=False)
        log.info("cohort %s: %d repeat events", lab, len(ev))

    con.execute(f"COPY (SELECT * EXCLUDE (is_adult, long_enough) FROM adm "
                f"WHERE is_adult AND long_enough) TO '{out['admissions'].as_posix()}' "
                "(FORMAT PARQUET)")
    con.execute(f"COPY (SELECT labevent_id, subject_id, hadm_id, code, itemid, charttime, "
                f"valuenum, ref_lo, ref_hi, flag FROM clean ORDER BY hadm_id, charttime) "
                f"TO '{out['clean_labs'].as_posix()}' (FORMAT PARQUET)")
    con.close()

    flow_df = pd.DataFrame(flow)
    if cfg.is_full:
        flow_df = suppress_small_counts(flow_df, ["n_events", "n_patients", "n_admissions"],
                                        int(cfg.get("eval.min_cell_count")))
    write_results(cfg, "cohort", {"flow": flow_df})
    return out
