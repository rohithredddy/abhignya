"""Minimal CertiVOI demo (05_IMPLEMENTATION_ROADMAP Phase 8).

Run:  streamlit run app/streamlit_app.py -- --config configs/default.yaml \
          [--override configs/synthetic.yaml]
Synthetic or open-demo data only; refuses data.mode = full (CONTRIBUTING.md §4).
"""

from __future__ import annotations

import argparse
import sys

import streamlit as st

from certivoi.config import load_config
from certivoi.demo import (
    BANNER,
    FullDataRefused,
    admissions,
    explain_event,
    interval_figure,
    prepare_lab,
)
from certivoi.triage import HIGH, LOW
from certivoi.viz.style import LAB_ORDER


def _args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--override", default="configs/synthetic.yaml")
    return ap.parse_known_args(sys.argv[1:])[0]


@st.cache_resource(show_spinner="Training / loading models (first run only)...")
def _view(config: str, override: str | None, lab: str, alpha: float):
    cfg = load_config(config, override_path=override or None)
    return cfg.data_mode, prepare_lab(cfg, lab, alpha)


def main() -> None:
    st.set_page_config(page_title="CertiVOI demo", layout="centered")
    st.title("CertiVOI — will this repeat test change the action?")
    st.warning(BANNER)
    a = _args()
    lab = st.sidebar.selectbox("Lab", LAB_ORDER)
    alpha = st.sidebar.select_slider("α (max. rate of missed action changes)",
                                     options=[0.005, 0.01, 0.02, 0.05, 0.10], value=0.05)
    try:
        mode, view = _view(a.config, a.override, lab, alpha)
    except FullDataRefused as e:
        st.error(str(e))
        st.stop()
    st.sidebar.caption(f"data mode: **{mode}** · test events: {len(view.events)}")
    hadm = st.sidebar.selectbox("Admission (test split)", admissions(view))
    ev = view.events[view.events["hadm_id"] == hadm]
    labels = [f"t = {t:%Y-%m-%d %H:%M} · last {y:.2f}" for t, y in zip(ev["t"], ev["y_last"],
                                                                       strict=True)]
    pick = st.sidebar.selectbox("Repeat test", range(len(ev)), format_func=lambda k: labels[k])
    info = explain_event(view, int(ev.index[pick]))

    badge = {LOW: "🟢 Low value (certified)", HIGH: "🔴 High value"}.get(
        info["tier"], "🟡 Potentially useful")
    c1, c2, c3 = st.columns(3)
    c1.metric("Tier", badge)
    c2.metric("P(action change)", f"{info['score']:.2f}")
    c3.metric("Predicted", f"{info['y_hat']:.2f} {view.unit}",
              f"{info['y_hat'] - info['y_last']:+.2f} vs last")
    st.pyplot(interval_figure(view, info))
    st.write(info["text"])
    st.caption(f"CRC threshold λ̂ = {view.lam:.3f} at α = {view.alpha:g} (calibration split); "
               f"high-value threshold = {view.tau_h:.3f} (validation split).")
    with st.expander("Observed result (hidden from the model)"):
        st.write(f"Observed {info['observed']:.2f} {view.unit}; action changed: "
                 f"**{'yes' if info['action_changed'] else 'no'}**")


main()
