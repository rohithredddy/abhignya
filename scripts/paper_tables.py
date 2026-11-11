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
        r"\begin{table}[t]\centering\footnotesize",
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
        for (name, r), ls, col in zip(runs.items(), ("-", "--"), ("black", "0.45"), strict=True):
            c = np.convolve(r.loss, np.ones(w) / w, mode="valid")
            ax.plot(np.arange(len(c)) + w, 100 * c, ls, color=col, lw=0.9, label=name)
        ax.axhline(100 * a, color="0.6", ls=":", lw=0.8)
        ax.set_title(NAME[lab], fontsize=7)
        ax.set_xlabel("tests at hospital B", fontsize=6)
        ax.tick_params(labelsize=5.5)
        ax.set_ylim(0, None)
    np.atleast_1d(axes)[0].set_ylabel("missed changes (%)", fontsize=6)
    np.atleast_1d(axes)[0].legend(fontsize=5.5, loc="center right", frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig_online.png", dpi=300, bbox_inches="tight")
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
    print("tables and figure written")


if __name__ == "__main__":
    main()
