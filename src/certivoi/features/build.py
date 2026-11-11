"""Feature matrices from strictly-prior data.

Implements 03_DATA_AND_COHORT §7 and CONTRIBUTING.md §5 (features for an event at t use
only data with timestamps < t − min_gap). Race and insurance are never features;
they go to the per-event ``meta`` table used only for subgroup evaluation.

Outputs (``data/interim/features/<hash>/``, per event, git-ignored):
``features_<LAB>.parquet`` (event_id + features) and ``meta_<LAB>.parquet``
(event keys, split-independent subgroup columns, admission length for
patient-day denominators). Manifest (name, description, group) -> ``results/features/``.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from certivoi.config import Config
from certivoi.data.cohort import cohort_outputs
from certivoi.data.extract import extract_outputs
from certivoi.labels.actions import action_map
from certivoi.labels.build import build_labels, creat_baselines
from certivoi.utils.io import stage_dir, write_results
from certivoi.utils.logging import get_logger

log = get_logger("certivoi.features")

CONTEXT = {
    "age": "age at draw (years)",
    "sex_female": "1 if female",
    "adm_emergency": "admission type: emergency",
    "adm_urgent": "admission type: urgent",
    "adm_elective": "admission type: elective / same-day surgery",
    "adm_observation": "admission type: observation",
    "hours_since_admit": "hours from admission to draw",
    "in_icu": "1 if in an ICU stay at draw time",
    "hour_of_day": "hour of draw (0-23)",
    "weekday": "day of week of draw (0=Mon)",
    "n_prior_admissions": "number of earlier admissions of the patient",
}
HISTORY = {
    "last": "last prior value of target lab",
    "second_last": "second-last prior value (within 72 h)",
    "hours_since_last": "hours since last prior value",
    "count_24h": "number of prior values in 24 h",
    "count_72h": "number of prior values in 72 h",
    "a0": "default action category from last value",
    "dist_boundary": "signed distance of last value to nearest boundary of a0 (+ inside)",
    "n_cat_changes": "category changes so far in admission",
    "last_flag_abnormal": "last value flagged abnormal",
}
TREND = {
    "delta_last2": "last minus second-last value",
    "slope_48h": "OLS slope over last 48 h (units/day)",
    "min_48h": "minimum over last 48 h",
    "max_48h": "maximum over last 48 h",
    "std_48h": "standard deviation over last 48 h",
}
CREAT_EXTRA = {
    "b48": "minimum creatinine in prior 48 h",
    "b7": "minimum creatinine in prior 7 days",
}


def feature_paths(cfg: Config, lab: str) -> dict[str, Path]:
    d = stage_dir(cfg, "features")
    return {"features": d / f"features_{lab}.parquet", "meta": d / f"meta_{lab}.parquet"}


def manifest(cfg: Config, lab: str) -> pd.DataFrame:
    """Feature manifest (name, description, group) for ``lab``."""
    rows = [(k, v, "context") for k, v in CONTEXT.items()]
    rows += [(k, v, "history") for k, v in HISTORY.items()]
    if lab == "CREAT":
        rows += [(k, v, "history") for k, v in CREAT_EXTRA.items()]
    rows += [(k, v, "trend") for k, v in TREND.items()]
    for code in _panel_codes(cfg, lab):
        rows.append((f"{code}_last", f"last prior {code} value", "panel"))
        rows.append((f"{code}_hrs", f"hours since last prior {code}", "panel"))
    return pd.DataFrame(rows, columns=["name", "description", "group"])


# Feature values are rounded (removes float noise such as -0.6000000000000014)
# and then stored as float32-representable numbers. TreeSHAP's interventional
# algorithm casts inputs to float32 but compares against float64 split
# thresholds, while LightGBM predicts in float64; with float32-exact inputs both
# see identical values and SHAP additivity holds (02 §6). Hours keep 2 decimals.
FEATURE_DECIMALS = 4
HOURS_DECIMALS = 2


def round_features(f: pd.DataFrame) -> pd.DataFrame:
    """Round to FEATURE_DECIMALS (hours: HOURS_DECIMALS) and make float32-exact."""
    out = f.copy()
    for c in out.columns:
        hours = c.startswith("hours_") or c.endswith("_hrs")
        v = out[c].round(HOURS_DECIMALS if hours else FEATURE_DECIMALS).to_numpy()
        out[c] = v.astype(np.float32).astype(np.float64)
    return out


def _panel_codes(cfg: Config, lab: str) -> list[str]:
    return [c for c in [*cfg.labs, *cfg.feature_labs] if c != lab]


def race_group(race: pd.Series) -> pd.Series:
    """Coarse race group for subgroup evaluation only (never a feature)."""
    r = race.fillna("").str.upper()
    out = np.select(
        [r.str.startswith("WHITE"), r.str.startswith("BLACK"),
         r.str.contains("HISPANIC|LATINO"), r.str.startswith("ASIAN")],
        ["white", "black", "hispanic", "asian"], default="other_unknown")
    return pd.Series(out, index=race.index)


def build_features_lab(cfg: Config, lab: str, labels: pd.DataFrame, clean: pd.DataFrame,
                       adm: pd.DataFrame, icu: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute features + meta for one lab. All inputs are pandas frames.

    ``labels`` provides events (event_id, hadm_id, t) and history-derived a0,
    last_id and edges (computed from strictly-prior data in labels/build.py).
    """
    gap = int(cfg.get("cohort.min_gap_minutes"))
    look = int(float(cfg.get("cohort.repeat_lookback_hours")) * 60)
    con = duckdb.connect()
    con.execute(f"SET threads TO {cfg.threads}")
    ev = labels[["event_id", "subject_id", "hadm_id", "t"]]
    hist = clean[clean["code"] == lab][["labevent_id", "hadm_id", "charttime", "valuenum", "flag"]]
    con.register("e", ev)
    con.register("h", hist)
    con.register("adm", adm)
    con.register("icu", icu)
    con.register("panel", clean[["code", "hadm_id", "charttime", "valuenum"]])

    hist_feats = con.execute(f"""
        WITH j AS (
            SELECT e.event_id, h.valuenum AS v, h.flag,
                   date_diff('second', h.charttime, e.t) / 3600.0 AS dh,
                   row_number() OVER (PARTITION BY e.event_id ORDER BY h.charttime DESC,
                                      h.labevent_id DESC) AS rn
            FROM e JOIN h ON h.hadm_id = e.hadm_id
             AND h.charttime < e.t - INTERVAL {gap} MINUTE
             AND h.charttime >= e.t - INTERVAL {look} MINUTE)
        SELECT event_id,
               max(v) FILTER (WHERE rn = 1) AS last,
               max(v) FILTER (WHERE rn = 2) AS second_last,
               max(dh) FILTER (WHERE rn = 1) AS hours_since_last,
               count(*) FILTER (WHERE dh <= 24) AS count_24h,
               count(*) AS count_72h,
               max(CASE WHEN flag = 'abnormal' THEN 1 ELSE 0 END) FILTER (WHERE rn = 1)
                   AS last_flag_abnormal,
               regr_slope(v, -dh / 24.0) FILTER (WHERE dh <= 48) AS slope_48h,
               min(v) FILTER (WHERE dh <= 48) AS min_48h,
               max(v) FILTER (WHERE dh <= 48) AS max_48h,
               stddev_samp(v) FILTER (WHERE dh <= 48) AS std_48h
        FROM j GROUP BY event_id""").df()

    ctx = con.execute(f"""
        SELECT e.event_id,
               a.anchor_age + (year(e.t) - a.anchor_year) AS age,
               CASE WHEN a.gender = 'F' THEN 1 ELSE 0 END AS sex_female,
               CASE WHEN upper(a.admission_type) LIKE '%EMER%' THEN 1 ELSE 0 END AS adm_emergency,
               CASE WHEN upper(a.admission_type) LIKE '%URGENT%' THEN 1 ELSE 0 END AS adm_urgent,
               CASE WHEN upper(a.admission_type) LIKE '%ELECTIVE%'
                      OR upper(a.admission_type) LIKE '%SAME DAY%' THEN 1 ELSE 0 END
                   AS adm_elective,
               CASE WHEN upper(a.admission_type) LIKE '%OBSERVATION%' THEN 1 ELSE 0 END
                   AS adm_observation,
               date_diff('second', a.admittime, e.t) / 3600.0 AS hours_since_admit,
               CASE WHEN EXISTS (SELECT 1 FROM icu i WHERE i.hadm_id = e.hadm_id
                                  AND i.intime < e.t - INTERVAL {gap} MINUTE
                                  AND (i.outtime IS NULL OR i.outtime > e.t))
                    THEN 1 ELSE 0 END AS in_icu,
               hour(e.t) AS hour_of_day,
               isodow(e.t) - 1 AS weekday,
               (SELECT count(*) FROM adm p WHERE p.subject_id = a.subject_id
                  AND p.admittime < a.admittime) AS n_prior_admissions,
               a.gender AS sex, a.race, a.anchor_year_group,
               date_diff('second', a.admittime, a.dischtime) / 86400.0 AS los_days
        FROM e JOIN adm a USING (hadm_id)""").df()

    panel_frames = []
    for code in _panel_codes(cfg, lab):
        panel_frames.append(con.execute(f"""
            SELECT e.event_id, p.valuenum AS "{code}_last",
                   date_diff('second', p.charttime, e.t) / 3600.0 AS "{code}_hrs"
            FROM (SELECT *, t - INTERVAL {gap} MINUTE AS t_cut FROM e) e
            ASOF LEFT JOIN (SELECT * FROM panel WHERE code = '{code}') p
              ON p.hadm_id = e.hadm_id AND e.t_cut > p.charttime""").df()
                            .set_index("event_id"))
    con.close()

    f = labels[["event_id", "a0", "last_id"]].merge(hist_feats, on="event_id", how="left")
    f["delta_last2"] = f["last"] - f["second_last"]
    edges = labels[[c for c in labels.columns if c.startswith("edge_")]].to_numpy()
    a0 = f["a0"].to_numpy().astype(int)
    lo = edges[np.arange(len(f)), a0]
    hi = edges[np.arange(len(f)), a0 + 1]
    with np.errstate(invalid="ignore"):
        f["dist_boundary"] = np.fmin(f["last"].to_numpy() - lo, hi - f["last"].to_numpy())

    # category changes among prior values in the admission (up to last_id)
    amap = action_map(cfg.get("actions"), lab)
    h = hist.sort_values(["hadm_id", "charttime", "labevent_id"]).reset_index(drop=True)
    if amap.kdigo is not None:
        b = creat_baselines(h, gap).set_index("labevent_id").reindex(h["labevent_id"])
        cat = amap.category(h["valuenum"], b["b48"].to_numpy(), b["b7"].to_numpy())
        b_ev = labels[["event_id", "b48", "b7"]]
        f = f.merge(b_ev, on="event_id", how="left")
    else:
        cat = amap.category(h["valuenum"])
    h["cat"] = cat
    h["chg"] = (h["cat"] != h.groupby("hadm_id")["cat"].shift()).astype(int)
    h.loc[h.groupby("hadm_id").head(1).index, "chg"] = 0
    h["n_cat_changes"] = h.groupby("hadm_id")["chg"].cumsum()
    f = f.merge(h[["labevent_id", "n_cat_changes"]].rename(columns={"labevent_id": "last_id"}),
                on="last_id", how="left")

    f = f.merge(ctx[["event_id", *CONTEXT]], on="event_id", how="left")
    f = f.join(pd.concat(panel_frames, axis=1), on="event_id")
    names = manifest(cfg, lab)["name"].tolist()
    feats = f[["event_id", *names]].sort_values("event_id").reset_index(drop=True)
    feats[names] = round_features(feats[names].astype(float))

    meta = labels[["event_id", "subject_id", "hadm_id", "t"]].merge(
        ctx[["event_id", "sex", "age", "race", "in_icu", "anchor_year_group", "los_days"]],
        on="event_id", how="left")
    meta["race_group"] = race_group(meta["race"])
    bands = [float(b) for b in cfg.get("conformal.age_bands")]
    meta["age_band"] = pd.cut(meta["age"], bins=bands, right=False,
                              labels=[f"{int(a)}-{int(b)}" for a, b in zip(bands[:-1], bands[1:],
                                                                           strict=True)]
                              ).astype(str)
    meta["icu"] = np.where(meta["in_icu"] == 1, "icu", "ward")
    meta = meta.drop(columns=["race", "in_icu"]).sort_values("event_id").reset_index(drop=True)
    return feats, meta


def build_features(cfg: Config, force: bool = False) -> dict[str, dict[str, Path]]:
    """Build features + meta for all labs; write manifests to results/features/."""
    paths = {lab: feature_paths(cfg, lab) for lab in cfg.labs}
    if not force and all(p.exists() for d in paths.values() for p in d.values()):
        log.info("features: cached at %s", paths["HGB"]["features"].parent)
        return paths
    labels = build_labels(cfg)
    co, ex = cohort_outputs(cfg), extract_outputs(cfg)
    clean = pd.read_parquet(co["clean_labs"])
    adm = pd.read_parquet(co["admissions"])
    icu = pd.read_parquet(ex["icustays"])
    manifests = {}
    for lab in cfg.labs:
        lab_df = pd.read_parquet(labels[lab])
        feats, meta = build_features_lab(cfg, lab, lab_df, clean, adm, icu)
        feats.to_parquet(paths[lab]["features"], index=False)
        meta.to_parquet(paths[lab]["meta"], index=False)
        manifests[f"manifest_{lab}"] = manifest(cfg, lab)
        log.info("features %s: %d events x %d features", lab, len(feats), feats.shape[1] - 1)
    write_results(cfg, "features", manifests)
    return paths
