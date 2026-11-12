"""Write the paper's LaTeX tables and figure from results/ (no hand-typed numbers).

Usage: python scripts/paper_tables.py [--main results/challenge2019] [--rep results/eicu_demo]
Outputs: paper/tables/tab_{cohort,perf,external,panel}.tex, paper/figures/fig_online.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

LABS = ["HGB", "K", "CREAT", "PLT"]
NAME = {"HGB": "Hemoglobin", "K": "Potassium", "CREAT": "Creatinine", "PLT": "Platelets"}


def pct(x: float, d: int = 1) -> str:
    return "--" if pd.isna(x) else f"{100 * x:.{d}f}"


def write(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def tab_cohort(m: Path, rep: Path, out: Path) -> None:
    flow = pd.read_csv(m / "cohort" / "flow.csv")
    flow = flow[flow["step"] == flow["step"].max()].set_index("lab")
    e1 = pd.read_csv(m / "E1" / "table2_labels.csv").set_index("lab")
    r1 = pd.read_csv(rep / "E1" / "table2_labels.csv").set_index("lab")
    rows = [f"{NAME[lab]} & {flow.loc[lab, 'n_events']:,} & {flow.loc[lab, 'n_patients']:,} & "
            f"{pct(e1.loc[lab, 'prev_D'])} & {pct(e1.loc[lab, 'pct_D_normal'])} & "
            f"{pct(e1.loc[lab, 'pct_D_stable'])} & {pct(r1.loc[lab, 'pct_D_normal'])} \\\\"
            for lab in LABS]
    write(out / "tab_cohort.tex", [
        f"% source: {m}/cohort/flow.csv, {m}/E1/table2_labels.csv, {rep}/E1/table2_labels.csv",
        r"\begin{table}[t]\centering\footnotesize",
        r"\caption{Repeat tests and how often action changes look ``normal'' or ``stable''.}",
        r"\label{tab:cohort}\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{lrrrrrr}\toprule",
        r" & & & & \multicolumn{2}{c}{Share of $D{=}1$ (\%)} & eICU \\ \cmidrule(lr){5-6}",
        r"Lab & Tests & Patients & $D{=}1$ (\%) & normal & stable & normal \\ \midrule",
        *rows, r"\bottomrule\end{tabular}\end{table}"])


def tab_perf(m: Path, out: Path) -> None:
    p = pd.read_csv(m / "E2" / "table3_performance.csv").set_index(["lab", "model"])
    c = pd.read_csv(m / "E3" / "table4_certified.csv").set_index(["lab", "alpha"])
    rows = []
    for lab in LABS:
        o, b = p.loc[(lab, "ours")], p.loc[(lab, "B5")]
        cells = [f"{o['auroc']:.3f} ({o['auroc_lo']:.3f}--{o['auroc_hi']:.3f})",
                 f"{b['auroc']:.3f}", f"{o['auprc']:.3f}"]
        for a in (0.01, 0.05):
            r = c.loc[(lab, a)]
            cells.append(f"{pct(r['resample_reduction_mean'])} / {pct(r['resample_risk_mean'], 2)}")
        rows.append(f"{NAME[lab]} & " + " & ".join(cells) + r" \\")
    write(out / "tab_perf.tex", [
        f"% source: {m}/E2/table3_performance.csv, {m}/E3/table4_certified.csv",
        r"\begin{table}[t]\centering\scriptsize",
        r"\caption{Discrimination of $D$ on the test patients and the certified low-value tier "
        r"(mean over 100 calibration/test re-splits). Tier cells: tests flagged / missed "
        r"action changes, both in \% of all repeat tests.}",
        r"\label{tab:perf}\setlength{\tabcolsep}{2.5pt}",
        r"\begin{tabular}{lcccc c}\toprule",
        r" & AUROC (95\% CI) & AUROC & AUPRC & \multicolumn{2}{c}{Flagged / missed (\%)} \\"
        r" \cmidrule(lr){5-6}",
        r"Lab & EDCP & direct & EDCP & $\alpha{=}1\%$ & $\alpha{=}5\%$ \\ \midrule",
        *rows, r"\bottomrule\end{tabular}\end{table}"])


def tab_external(m: Path, out: Path) -> None:
    t = pd.read_csv(m / "E4" / "temporal.csv")
    o = pd.read_csv(m / "E7" / "table10_online.csv")
    o = o[(o["status"] == "ok") & (o["order"] == "temporal")]
    rows = []
    for a in (0.01, 0.05):
        for lab in LABS:
            g = t[(t["lab"] == lab) & (t["alpha"] == a)].set_index("setting")
            q = o[(o["lab"] == lab) & (o["alpha"] == a)].set_index("method")
            s, rc = g.loc["shifted"], g.loc["recalibrated"]
            cells = [f"{pct(s['risk'], 2)} / {pct(s['reduction'], 0)}",
                     f"{pct(rc['risk'], 2)} / {pct(rc['reduction'], 0)}"]
            for mth in ("online_p0.2", "online_p0.1"):
                cells.append(f"{pct(q.loc[mth, 'miss_rate_mean'], 2)} / "
                             f"{pct(q.loc[mth, 'reduction_mean'], 0)}")
            lead = rf"\multirow{{4}}{{*}}{{{100 * a:g}\%}}" if lab == "HGB" else ""
            rows.append(f"{lead} & {NAME[lab]} & " + " & ".join(cells) + r" \\")
        rows.append(r"\midrule" if a == 0.01 else "")
    write(out / "tab_external.tex", [
        f"% source: {m}/E4/temporal.csv, {m}/E7/table10_online.csv (order=temporal)",
        r"\begin{table}[t]\centering\footnotesize",
        r"\caption{Transfer to an unseen hospital (models and calibration from hospital A, "
        r"test stream from hospital B). Cells: missed action changes (\%) / tests avoided (\%). "
        r"Online cells average five audit seeds.}",
        r"\label{tab:external}\setlength{\tabcolsep}{2.5pt}",
        r"\begin{tabular}{llcccc}\toprule",
        r"$\alpha$ & Lab & Static & Recalib. 20\% & Online $p{=}0.2$ & Online $p{=}0.1$ \\"
        r" \midrule",
        *[r for r in rows if r], r"\bottomrule\end{tabular}\end{table}"])


def tab_panel(m: Path, out: Path) -> None:
    p = pd.read_csv(m / "E7" / "table9_panel.csv")
    p = p[p["status"] == "ok"].set_index(["panel", "alpha", "method"])
    rows = []
    for panel, labs in (("CBC", "Hgb + Plt"), ("BMP", "K + Creat")):
        for a in (0.01, 0.05):
            cells = []
            for mth in ("naive_per_lab_alpha", "bonferroni_per_lab", "panel_crc_noisy_or"):
                r = p.loc[(panel, a, mth)]
                cells.append(f"{pct(r['risk_mean'], 2)} / {pct(r['draw_reduction_mean'], 0)}")
            rows.append(f"{panel} ({labs}) & {100 * a:g}\\% & " + " & ".join(cells) + r" \\")
    write(out / "tab_panel.tex", [
        f"% source: {m}/E7/table9_panel.csv",
        r"\begin{table}[t]\centering\footnotesize",
        r"\caption{Tube-level decisions. Cells: draws with a missed action change (\%) / draws "
        r"avoided (\%), mean of 100 re-splits.}",
        r"\label{tab:panel}\setlength{\tabcolsep}{2.5pt}",
        r"\begin{tabular}{lcccc}\toprule",
        r"Draw & $\alpha$ & Per-lab at $\alpha$ & Bonferroni & Panel CRC \\ \midrule",
        *rows, r"\bottomrule\end{tabular}\end{table}"])


def fig_online(cfg_path: str, override: str, out: Path) -> None:
    import matplotlib.pyplot as plt

    from certivoi.config import load_config
    from certivoi.conformal.crc import crc_threshold
    from certivoi.conformal.online import online_risk_control, rank_transform, static_result
    from certivoi.models.train import train_lab
    from certivoi.viz.style import SINGLE_COL, figure

    cfg = load_config(cfg_path, override_path=override)
    a, eta = 0.05, float(cfg.get("extensions.online.eta"))
    fig, axes = figure(SINGLE_COL, 1.9, ncols=2)
    for ax, lab in zip(np.atleast_1d(axes), ("HGB", "K"), strict=True):
        df = pd.read_parquet(train_lab(cfg, lab, tag="temporal", split_col="split_temporal",
                                       models=("ours",)))
        cal, te = df[df["split"] == "calib"], df[df["split"] == "test"]
        rng = np.random.default_rng(cfg.seed)
        te = te.iloc[np.lexsort((rng.random(len(te)),))]
        f = rank_transform(cal["score_ours"].to_numpy())
        lam = crc_threshold(cal["score_ours"].to_numpy(), cal["D"].to_numpy(), a)
        lam1 = float(f(np.array([lam]))[0])
        u, d = f(te["score_ours"].to_numpy()), te["D"].to_numpy()
        w = 2000
        runs = {"static": static_result(u, d, lam1),
                "online, $p$=0.2": online_risk_control(u, d, a, eta, 0.2, lam1, rng)}
        for (name, r), ls, col in zip(runs.items(), ("-", "-"), ("#D55E00", "#0072B2"),
                                      strict=True):
            c = np.convolve(r.loss, np.ones(w) / w, mode="valid")
            ax.plot(np.arange(len(c)) + w, 100 * c, ls, color=col, lw=1.1, label=name)
        ax.axhline(100 * a, color="0.3", ls=":", lw=0.9)
        ax.grid(axis="y", color="0.9", lw=0.5)
        ax.set_title(NAME[lab], fontsize=7)
        ax.set_xlabel("tests at hospital B", fontsize=6)
        ax.tick_params(labelsize=5.5)
        ax.set_ylim(0, None)
    np.atleast_1d(axes)[0].set_ylabel("missed changes (%)", fontsize=6)
    np.atleast_1d(axes)[0].legend(fontsize=5.5, loc="center right", frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig_online.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_tradeoff(cfg_path: str, override: str, out: Path) -> None:
    """Miss rate vs share of tests flagged on the test split, EDCP vs B5 vs proxies."""
    import matplotlib.pyplot as plt

    from certivoi.config import load_config
    from certivoi.conformal.crc import crc_threshold, risk_and_reduction
    from certivoi.models.train import train_lab
    from certivoi.viz.style import SINGLE_COL, figure

    cfg = load_config(cfg_path, override_path=override)
    styles = {"ours": ("EDCP", "-", "#0072B2"), "B5": ("Direct", "--", "#E69F00"),
              "B3": ("Stability", "-.", "#009E73"), "B2": ("Normality", ":", "#CC79A7")}
    fig, axes = figure(SINGLE_COL, 1.75, ncols=3)
    for ax, lab in zip(np.atleast_1d(axes), ("HGB", "K", "CREAT"), strict=True):
        df = pd.read_parquet(train_lab(cfg, lab))
        cal, te = df[df["split"] == "calib"], df[df["split"] == "test"]
        d = te["D"].to_numpy()
        for model, (name, ls, col) in styles.items():
            s = te[f"score_{model}"].to_numpy()
            lams = np.unique(np.quantile(s, np.linspace(0, 1, 201)))
            curve = np.array([risk_and_reduction(s, d, lam) for lam in lams])
            ax.plot(100 * curve[:, 1], 100 * curve[:, 0], ls, color=col, lw=1.2, label=name)
        for a, mk in ((0.01, "o"), (0.05, "s")):
            lam = crc_threshold(cal["score_ours"].to_numpy(), cal["D"].to_numpy(), a)
            r, red = risk_and_reduction(te["score_ours"].to_numpy(), d, lam)
            ax.plot(100 * red, 100 * r, mk, ms=3.6, mfc="#D55E00", mec="black", mew=0.5)
        ax.set_title(NAME[lab], fontsize=7)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 8)
        ax.set_xlabel("tests flagged (%)", fontsize=6)
        ax.tick_params(labelsize=5.5)
    np.atleast_1d(axes)[0].set_ylabel("missed changes (%)", fontsize=6)
    np.atleast_1d(axes)[0].legend(fontsize=5, loc="upper left", frameon=False)
    fig.tight_layout(w_pad=0.6)
    fig.savefig(out / "fig_tradeoff.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_examples(cfg_path: str, override: str, out: Path) -> None:
    """Three real potassium test events (one per tier): forecast vs action bands."""
    import matplotlib.pyplot as plt

    from certivoi.config import load_config
    from certivoi.demo import prepare_lab
    from certivoi.triage import HIGH, LOW, POTENTIAL
    from certivoi.viz.style import SINGLE_COL, figure

    cfg = load_config(cfg_path, override_path=override)
    view = prepare_lab(cfg, "K", alpha=0.01)
    ev = view.events
    picks = []
    for tier in (LOW, POTENTIAL, HIGH):
        idx = np.flatnonzero(ev["tier"].to_numpy() == tier)
        j = idx[np.argsort(ev["width"].to_numpy()[idx])[len(idx) // 2]]   # median-width event
        picks.append((tier, j))
    fig, ax = figure(SINGLE_COL, 1.45)
    bands = [(2.0, 3.5, "#56B4E9", "replete"), (3.5, 5.5, "#009E73", "no action"),
             (5.5, 7.0, "#D55E00", "treat")]
    for lo, hi, col, name in bands:
        ax.axvspan(lo, hi, color=col, alpha=0.13, lw=0)
        ax.text((lo + hi) / 2, 2.62, name, ha="center", va="bottom", fontsize=5.5, color=col)
    taus = view.taus
    label = {LOW: "low value", POTENTIAL: "potentially useful", HIGH: "high value"}
    for k, (_tier, j) in enumerate(picks):
        q = np.sort(view.quantiles[j])
        q05, q95 = np.interp([0.05, 0.95], taus, q)
        r = ev.iloc[j]
        ax.plot([q05, q95], [k, k], color="#0072B2", lw=1.2, solid_capstyle="butt")
        ax.plot([r["q_lo"], r["q_hi"]], [k, k], color="#0072B2", lw=4.5, solid_capstyle="butt")
        ax.plot(r["y_hat"], k, "o", ms=3.5, mfc="white", mec="#0072B2", mew=0.9)
        ax.plot(r["y_last"], k, "x", ms=4, color="black", mew=0.9)
        ax.text(6.95, k, f"P = {r['score']:.2f}", ha="right", va="center", fontsize=5.5)
    ax.set_yticks(range(3), [label[t] for t, _ in picks], fontsize=6)
    ax.set_ylim(-0.6, 2.95)
    ax.set_xlim(2.0, 7.0)
    ax.set_xlabel("potassium (mEq/L)", fontsize=6)
    ax.tick_params(axis="x", labelsize=5.5)
    for s in ("left",):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig(out / "fig_examples.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", default="results/challenge2019")
    ap.add_argument("--rep", default="results/eicu_demo")
    ap.add_argument("--override", default="configs/challenge2019.yaml")
    a = ap.parse_args()
    m, rep = Path(a.main), Path(a.rep)
    tabs, figs = Path("paper/tables"), Path("paper/figures")
    tabs.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    tab_cohort(m, rep, tabs)
    tab_perf(m, tabs)
    tab_external(m, tabs)
    tab_panel(m, tabs)
    fig_online("configs/default.yaml", a.override, figs)
    fig_tradeoff("configs/default.yaml", a.override, figs)
    fig_examples("configs/default.yaml", a.override, figs)
    print("tables and figure written")


if __name__ == "__main__":
    main()
