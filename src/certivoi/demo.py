"""Back end for the minimal demo app (05_IMPLEMENTATION_ROADMAP Phase 8).

Pure Python, no Streamlit import, so it is unit-tested like the rest of the
package. The app (``app/streamlit_app.py``) is a thin layer on top.

Privacy (CONTRIBUTING.md §4): the demo shows individual events, so it refuses to run
when ``data.mode`` is ``full``. It runs only on synthetic or open-demo data.
Only test-split events are shown; thresholds use calib (CRC) and val (high tier).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from certivoi.config import Config
from certivoi.conformal.crc import crc_threshold
from certivoi.explain.decompose import decompose
from certivoi.explain.shap_attr import tree_shap
from certivoi.explain.templates import reasons, render
from certivoi.features.build import manifest
from certivoi.models.train import edge_matrix, load_lab_frame, load_models, train_lab
from certivoi.triage import assign_tiers, high_value_threshold

BANNER = "Research prototype — not for clinical use. Synthetic or open demo data only."


class FullDataRefused(RuntimeError):
    """Raised when the demo is pointed at credentialed (full) MIMIC-IV data."""


def check_mode(cfg: Config) -> None:
    """Implements CONTRIBUTING.md §4: no row-level display of full MIMIC-IV."""
    if cfg.is_full:
        raise FullDataRefused("The demo shows individual events and must not run on full "
                              "MIMIC-IV. Use data.mode: synthetic or demo.")


@dataclass
class LabView:
    """Everything the app needs for one lab (test-split events only)."""

    lab: str
    unit: str
    alpha: float
    lam: float
    tau_h: float
    events: pd.DataFrame          # test events: ids, labels, score, tier, decomposition
    edges: np.ndarray             # (n, K+1) action-category edges per event
    quantiles: np.ndarray         # (n, n_tau) predicted values of the new result
    taus: np.ndarray
    _feat_frame: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)
    _background: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)
    _bundle: dict = field(repr=False, default_factory=dict)
    _desc: dict = field(repr=False, default_factory=dict)
    _interval: tuple[float, float] = (0.1, 0.9)


def prepare_lab(cfg: Config, lab: str, alpha: float | None = None) -> LabView:
    """Train (or load cached) models for ``lab`` and score its test events."""
    check_mode(cfg)
    alpha = float(cfg.get("eval.main_alpha") if alpha is None else alpha)
    lo, hi = (float(x) for x in cfg.get("explain.interval"))
    preds = pd.read_parquet(train_lab(cfg, lab))
    frame, _ = load_lab_frame(cfg, lab)
    bundle = load_models(cfg, lab)
    feats = bundle["features"]
    frame = frame.merge(preds[["event_id", "score_ours"]], on="event_id")
    cal, va = frame[frame["split"] == "calib"], frame[frame["split"] == "val"]
    lam = crc_threshold(cal["score_ours"].to_numpy(), cal["D"].to_numpy(), alpha)
    tau_h = high_value_threshold(va["score_ours"].to_numpy(), va["D"].to_numpy(),
                                 float(cfg.get("triage.high_value_min_precision")))
    te = frame[frame["split"] == "test"].sort_values(["hadm_id", "t"]).reset_index(drop=True)
    taus = np.array(sorted(float(t) for t in cfg.get("model.quantiles")))
    p = preds.set_index("event_id").loc[te["event_id"]]
    q = p[[f"q_{t:g}" for t in taus]].to_numpy(dtype=float)
    edges = edge_matrix(te)
    dec = decompose(p[f"q_{lo:g}"].to_numpy(), p[f"q_{0.5:g}"].to_numpy(),
                    p[f"q_{hi:g}"].to_numpy(), edges, te["a0"].to_numpy())
    events = pd.concat([te[["event_id", "hadm_id", "t", "y_last", "y", "a0", "D",
                            "score_ours"]].rename(columns={"score_ours": "score"}),
                        dec.reset_index(drop=True)], axis=1)
    events["tier"] = assign_tiers(events["score"].to_numpy(), lam, tau_h)
    tr = frame[frame["split"] == "train"]
    bg = tr[feats].sample(n=min(int(cfg.get("explain.shap_background_rows")), len(tr)),
                          random_state=cfg.seed)
    man = manifest(cfg, lab)
    return LabView(lab=lab, unit=cfg.labs[lab].unit, alpha=alpha, lam=lam, tau_h=tau_h,
                   events=events, edges=edges, quantiles=q, taus=taus,
                   _feat_frame=te[feats], _background=bg, _bundle=bundle,
                   _desc=dict(zip(man["name"], man["description"], strict=True)),
                   _interval=(lo, hi))


def admissions(view: LabView) -> list:
    """Admission ids that have at least one test-split event for this lab."""
    return sorted(view.events["hadm_id"].unique().tolist())


def explain_event(view: LabView, i: int) -> dict:
    """Tier, score, interval vs thresholds and template explanation for event row ``i``."""
    lo, hi = view._interval
    fq, feats = view._bundle["models"]["ours"], view._bundle["features"]
    x = view._feat_frame.iloc[[i]]
    attr = tree_shap(fq, x, view._background, (lo, 0.5, hi))
    ev = view.events.iloc[i]
    vals = x.to_numpy()[0]
    text = render(view.lab, view.unit, ev["tier"], view.alpha, ev["y_hat"], ev["q_lo"],
                  ev["q_hi"], ev["margin"], ev["boundary"], float(ev["score"]),
                  reasons(attr.med[0], vals, feats, view._desc),
                  reasons(attr.unc(lo, hi)[0], vals, feats, view._desc, kind="uncertainty"))
    return {"tier": ev["tier"], "score": float(ev["score"]), "y_last": float(ev["y_last"]),
            "y_hat": float(ev["y_hat"]), "q_lo": float(ev["q_lo"]), "q_hi": float(ev["q_hi"]),
            "thresholds": [float(e) for e in view.edges[i][1:-1] if np.isfinite(e)],
            "quantiles": view.quantiles[i], "taus": view.taus, "text": text,
            "observed": float(ev["y"]), "action_changed": bool(ev["D"])}


def interval_figure(view: LabView, info: dict):
    """Matplotlib figure: predicted quantiles of the new value vs action thresholds."""
    from certivoi.viz.style import SINGLE_COL, figure

    fig, ax = figure(SINGLE_COL * 1.6, 1.6)
    q = np.sort(info["quantiles"])
    ax.fill_betweenx([0, 1], info["q_lo"], info["q_hi"], color="0.85", label="80% range")
    ax.plot(q, view.taus, color="black", linewidth=1, label="predicted CDF")
    ax.axvline(info["y_hat"], color="black", linestyle="--", linewidth=1, label="median")
    ax.axvline(info["y_last"], color="0.4", linestyle="-.", linewidth=1, label="last value")
    for k, e in enumerate(info["thresholds"]):
        ax.axvline(e, color="0.5", linestyle=":", linewidth=1.2,
                   label="action threshold" if k == 0 else None)
    ax.set_xlabel(f"{view.lab} ({view.unit})")
    ax.set_ylabel("cumulative prob.")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=6, loc="lower right")
    return fig
