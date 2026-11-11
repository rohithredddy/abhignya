"""Fit all models for one lab and store per-event predictions.

Implements 02_METHODOLOGY §3–§4, §7 and CONTRIBUTING.md §5 (fit on train, early stop
on val, calibrate on val; calib/test untouched). Per-event predictions go to
``data/interim/models/<hash>/<tag>/preds_<LAB>.parquet`` (git-ignored); model
objects are pickled next to them. Hyperparameters -> returned ``info`` dict.
"""

from __future__ import annotations

import itertools
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from certivoi.config import Config
from certivoi.data.split import make_splits
from certivoi.features.build import build_features, manifest
from certivoi.labels.actions import action_map
from certivoi.labels.build import labels_path
from certivoi.models.baselines import LGBMBinary, Logistic, b1_rule_score
from certivoi.models.quantile import QuantileForecaster, lgb_params
from certivoi.utils.io import stage_dir
from certivoi.utils.logging import get_logger
from certivoi.voi.edcp import IsotonicCalibrator, edcp

log = get_logger("certivoi.train")

ALL_MODELS = ("ours", "B1", "B2", "B3", "B4", "B5")
LABEL_COLS = ["y", "y_last", "a0", "cat", "D", "N", "S"]
META_COLS = ["subject_id", "hadm_id", "t", "sex", "age_band", "race_group", "icu",
             "anchor_year_group", "los_days"]


def load_lab_frame(cfg: Config, lab: str) -> tuple[pd.DataFrame, list[str]]:
    """Join features, labels, meta and splits for one lab. Returns (frame, feature names)."""
    fp = build_features(cfg)[lab]
    feats = pd.read_parquet(fp["features"])
    meta = pd.read_parquet(fp["meta"])
    labels = pd.read_parquet(labels_path(cfg, lab))
    edge_cols = [c for c in labels.columns if c.startswith("edge_")]
    splits = pd.read_parquet(make_splits(cfg))
    lab_cols = [c for c in LABEL_COLS if c not in feats.columns]  # a0 is also a feature
    df = (feats.merge(labels[["event_id", *lab_cols, *edge_cols]], on="event_id")
          .merge(meta[["event_id", *META_COLS]], on="event_id")
          .merge(splits, on="subject_id", how="left"))
    return df, manifest(cfg, lab)["name"].tolist()


def edge_matrix(df: pd.DataFrame) -> np.ndarray:
    cols = sorted([c for c in df.columns if c.startswith("edge_")], key=lambda c: int(c[5:]))
    return df[cols].to_numpy(dtype=float)


def _grid(cfg: Config) -> list[dict[str, Any]]:
    g = cfg.get("model.small_grid")
    keys = list(g)
    return [dict(zip(keys, vals, strict=True)) for vals in itertools.product(*g.values())]


def fit_ours(cfg: Config, tr: pd.DataFrame, va: pd.DataFrame, feats: list[str], target: str,
             params: dict[str, Any]) -> QuantileForecaster:
    taus = [float(t) for t in cfg.get("model.quantiles")]
    es = int(cfg.get("model.lightgbm.early_stopping_rounds"))
    return QuantileForecaster(taus, params, es, target).fit(
        tr[feats], tr["y"].to_numpy(), tr["y_last"].to_numpy(),
        va[feats], va["y"].to_numpy(), va["y_last"].to_numpy())


def select_quantile_params(cfg: Config, tr: pd.DataFrame, va: pd.DataFrame, feats: list[str],
                           target: str) -> dict[str, Any]:
    """Pick small-grid params by val pinball loss at tau = 0.5 (only if model.use_grid)."""
    base = lgb_params(cfg.get("model.lightgbm"), cfg.threads, cfg.seed)
    if not cfg.get("model.use_grid", False):
        return base
    es = int(cfg.get("model.lightgbm.early_stopping_rounds"))
    best, best_loss = base, np.inf
    for g in _grid(cfg):
        p = {**base, **g}
        f = QuantileForecaster([0.5], p, es, target).fit(
            tr[feats], tr["y"].to_numpy(), tr["y_last"].to_numpy(),
            va[feats], va["y"].to_numpy(), va["y_last"].to_numpy())
        loss = f.val_pinball(va[feats], va["y"].to_numpy(), va["y_last"].to_numpy())
        if loss < best_loss:
            best, best_loss = p, loss
    return best


def train_models(cfg: Config, lab: str, df: pd.DataFrame, feats: list[str],
                 split_col: str = "split", target: str | None = None,
                 models: tuple[str, ...] = ALL_MODELS) -> tuple[pd.DataFrame, dict, dict]:
    """Fit the requested models; return (predictions for every row, models, info)."""
    df = df[df[split_col].notna()].reset_index(drop=True)
    tr, va = df[df[split_col] == "train"], df[df[split_col] == "val"]
    if len(tr) == 0 or len(va) == 0:
        raise ValueError(f"{lab}: empty train or val for split column '{split_col}'")
    es = int(cfg.get("model.lightgbm.early_stopping_rounds"))
    base = lgb_params(cfg.get("model.lightgbm"), cfg.threads, cfg.seed)
    out = df[["event_id", split_col, *LABEL_COLS, *META_COLS]].rename(columns={split_col: "split"})
    out = pd.concat([out, df[[c for c in df.columns if c.startswith("edge_")]]], axis=1)
    fitted: dict[str, Any] = {}
    info: dict[str, Any] = {"lab": lab, "n_train": len(tr), "n_val": len(va),
                            "n_features": len(feats)}

    if "ours" in models:
        target = target or str(cfg.get("model.target"))
        candidates = [target]
        if lab == "CREAT" and target == "delta" and cfg.get("model.creat_target", "auto") == "auto":
            candidates = ["delta", "absolute"]  # 02_METHODOLOGY §3: pick by val pinball
        best = None
        for tgt in candidates:
            params = select_quantile_params(cfg, tr, va, feats, tgt)
            f = fit_ours(cfg, tr, va, feats, tgt, params)
            loss = f.val_pinball(va[feats], va["y"].to_numpy(), va["y_last"].to_numpy())
            info[f"val_pinball_{tgt}"] = loss
            if best is None or loss < best[1]:
                best = (f, loss, params)
        fq, _, qparams = best
        info.update(target=fq.target, quantile_params={k: v for k, v in qparams.items()
                                                       if k not in ("n_jobs", "verbose")},
                    quantile_best_iterations=dict(zip(map(str, fq.taus), fq.best_iterations,
                                                      strict=True)))
        vals = fq.predict_values(df[feats], df["y_last"].to_numpy())
        for k, tau in enumerate(fq.taus):
            out[f"q_{tau:g}"] = vals[:, k]
        raw = edcp(vals, fq.taus, edge_matrix(df), df["a0"].to_numpy())
        cal = IsotonicCalibrator().fit(raw[(df[split_col] == "val").to_numpy()],
                                       va["D"].to_numpy())
        out["edcp_raw"] = raw
        out["score_ours"] = cal.predict(raw)
        fitted["ours"], fitted["calibrator"] = fq, cal

    if "B1" in models:
        amap = action_map(cfg.get("actions"), lab)
        out["score_B1"] = b1_rule_score(df, amap.no_action_index,
                                        cfg.labs[lab].stability_delta,
                                        float(cfg.get("baselines.b1_max_hours")))
    for name, label, invert in (("B2", "N", True), ("B3", "S", True), ("B5", "D", False)):
        if name not in models:
            continue
        params = base
        if cfg.get("model.use_grid", False):
            best_ll = np.inf
            for g in _grid(cfg):
                m = LGBMBinary({**base, **g}, es).fit(tr[feats], tr[label], va[feats], va[label])
                p = np.clip(m.predict_proba(va[feats]), 1e-6, 1 - 1e-6)
                yv = va[label].to_numpy()
                ll = -np.mean(yv * np.log(p) + (1 - yv) * np.log(1 - p))
                if ll < best_ll:
                    best_ll, params = ll, {**base, **g}
        m = LGBMBinary(params, es).fit(tr[feats], tr[label], va[feats], va[label])
        p = m.predict_proba(df[feats])
        out[f"score_{name}"] = 1.0 - p if invert else p
        fitted[name] = m
        info[f"{name}_best_iteration"] = m.best_iteration
    if "B4" in models:
        m = Logistic(cfg.seed).fit(tr[feats], tr["D"])
        out["score_B4"] = m.predict_proba(df[feats])
        fitted["B4"] = m
    return out, fitted, info


def model_dir(cfg: Config, tag: str) -> Path:
    d = stage_dir(cfg, "models") / tag
    d.mkdir(parents=True, exist_ok=True)
    return d


def train_lab(cfg: Config, lab: str, tag: str = "main", split_col: str = "split",
              drop_groups: tuple[str, ...] = (), target: str | None = None,
              models: tuple[str, ...] = ALL_MODELS, force: bool = False) -> Path:
    """Train (or load cached) models for ``lab``; returns the predictions path."""
    d = model_dir(cfg, tag)
    preds_p, models_p, info_p = (d / f"preds_{lab}.parquet", d / f"models_{lab}.pkl",
                                 d / f"info_{lab}.json")
    if preds_p.exists() and models_p.exists() and not force:
        return preds_p
    df, feats = load_lab_frame(cfg, lab)
    if drop_groups:
        man = manifest(cfg, lab)
        drop = set(man.loc[man["group"].isin(drop_groups), "name"])
        feats = [f for f in feats if f not in drop]
    preds, fitted, info = train_models(cfg, lab, df, feats, split_col, target, models)
    info.update(tag=tag, split_col=split_col, drop_groups=list(drop_groups), features=feats)
    preds.to_parquet(preds_p, index=False)
    with models_p.open("wb") as fh:
        pickle.dump({"models": fitted, "features": feats}, fh)
    info_p.write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    log.info("train %s [%s]: n_train=%d, target=%s", lab, tag, info["n_train"],
             info.get("target", "-"))
    return preds_p


def load_models(cfg: Config, lab: str, tag: str = "main") -> dict[str, Any]:
    with (model_dir(cfg, tag) / f"models_{lab}.pkl").open("rb") as fh:
        return pickle.load(fh)


def load_info(cfg: Config, lab: str, tag: str = "main") -> dict[str, Any]:
    return json.loads((model_dir(cfg, tag) / f"info_{lab}.json").read_text(encoding="utf-8"))
