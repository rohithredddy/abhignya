"""Per-event labels: y_last, default action a0, decision label D, proxies N and S.

Implements 02_METHODOLOGY §1–§2 (and §4 category intervals). History uses only
same-lab clean values in the same admission strictly before ``t - min_gap``.
Output ``data/interim/labels/<hash>/labels_<LAB>.parquet`` (per event, git-ignored):
event keys, ``t, y, y_last, t_last, a0, cat, D, N, S``, CREAT ``b48, b7``, and
category edges ``edge_0..edge_K`` valid at time t.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from certivoi.config import Config
from certivoi.data.cohort import build_cohort
from certivoi.labels.actions import ActionMap, action_map
from certivoi.utils.io import stage_dir
from certivoi.utils.logging import get_logger

log = get_logger("certivoi.labels")


def labels_path(cfg: Config, lab: str) -> Path:
    return stage_dir(cfg, "labels") / f"labels_{lab}.parquet"


def creat_baselines(hist: pd.DataFrame, gap_min: int) -> pd.DataFrame:
    """For each creatinine row: min prior value in the last 48 h and 7 d (strictly prior).

    Inputs: ``hist`` with labevent_id, hadm_id, charttime, valuenum. Output:
    labevent_id, b48, b7 (NaN when no prior value in the window).
    """
    con = duckdb.connect()
    con.register("h", hist[["labevent_id", "hadm_id", "charttime", "valuenum"]])
    out = con.execute(f"""
        SELECT e.labevent_id,
               min(p.valuenum) FILTER (WHERE p.charttime >= e.charttime - INTERVAL 48 HOUR) AS b48,
               min(p.valuenum) AS b7
        FROM h e LEFT JOIN h p
          ON p.hadm_id = e.hadm_id
         AND p.charttime < e.charttime - INTERVAL {gap_min} MINUTE
         AND p.charttime >= e.charttime - INTERVAL 7 DAY
        GROUP BY e.labevent_id""").df()
    con.close()
    return out


def attach_last(events: pd.DataFrame, hist: pd.DataFrame, gap_min: int) -> pd.DataFrame:
    """Add ``y_last, t_last, last_id``: most recent same-lab value before ``t - gap``."""
    ev = events.copy()
    ev["t_cut"] = ev["t"] - pd.Timedelta(minutes=gap_min)
    h = hist.rename(columns={"charttime": "t_last", "valuenum": "y_last",
                             "labevent_id": "last_id"})[["hadm_id", "t_last", "y_last", "last_id"]]
    m = pd.merge_asof(ev.sort_values("t_cut"), h.sort_values("t_last"), left_on="t_cut",
                      right_on="t_last", by="hadm_id", direction="backward",
                      allow_exact_matches=False)
    return m.drop(columns="t_cut").sort_values(["hadm_id", "t"]).reset_index(drop=True)


def label_events(events: pd.DataFrame, hist: pd.DataFrame, amap: ActionMap,
                 stability_delta: float, gap_min: int) -> pd.DataFrame:
    """Compute a0, cat, D, N, S and edges for one lab's repeat events.

    Inputs: repeat events (event_id, subject_id, hadm_id, t, y, ref_lo, ref_hi),
    clean same-lab history (labevent_id, hadm_id, charttime, valuenum), action map,
    stability delta, min gap. Output: labelled events (one row per event).
    """
    ev = attach_last(events, hist, gap_min)
    n = len(ev)
    if amap.kdigo is not None:
        base = creat_baselines(hist, gap_min).set_index("labevent_id")
        b_now = base.reindex(ev["event_id"])
        b_last = base.reindex(ev["last_id"])
        ev["b48"], ev["b7"] = b_now["b48"].to_numpy(), b_now["b7"].to_numpy()
        ev["a0"] = amap.category(ev["y_last"], b_last["b48"].to_numpy(), b_last["b7"].to_numpy())
        ev["cat"] = amap.category(ev["y"], ev["b48"].to_numpy(), ev["b7"].to_numpy())
        edges = amap.edges(n, ev["b48"].to_numpy(), ev["b7"].to_numpy())
    else:
        ev["a0"] = amap.category(ev["y_last"])
        ev["cat"] = amap.category(ev["y"])
        edges = amap.edges(n)
    ev["D"] = (ev["cat"] != ev["a0"]).astype(int)
    ev["N"] = ((ev["y"] >= ev["ref_lo"]) & (ev["y"] <= ev["ref_hi"])).astype(int)
    ev["S"] = ((ev["y"] - ev["y_last"]).abs() <= stability_delta + 1e-9).astype(int)
    for k in range(edges.shape[1]):
        ev[f"edge_{k}"] = edges[:, k]
    return ev


def build_labels(cfg: Config, force: bool = False) -> dict[str, Path]:
    """Build and cache labels for every target lab."""
    gap = int(cfg.get("cohort.min_gap_minutes"))
    paths = {lab: labels_path(cfg, lab) for lab in cfg.labs}
    if not force and all(p.exists() for p in paths.values()):
        log.info("labels: cached at %s", paths["HGB"].parent)
        return paths
    co = build_cohort(cfg)
    clean = pd.read_parquet(co["clean_labs"])
    for lab, spec in cfg.labs.items():
        events = pd.read_parquet(co[f"events_{lab}"])
        hist = clean[clean["code"] == lab][["labevent_id", "hadm_id", "charttime", "valuenum"]]
        amap = action_map(cfg.get("actions"), lab)
        out = label_events(events, hist, amap, spec.stability_delta, gap)
        out.to_parquet(paths[lab], index=False)
        log.info("labels %s: n=%d, prevalence D=%.3f", lab, len(out), out["D"].mean())
    return paths
