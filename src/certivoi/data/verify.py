"""Aggregate verification of lab itemids (label, fluid, units).

Implements 03_DATA_AND_COHORT §4 ("fail loudly on mismatch") and CONTRIBUTING.md §11
(itemid/unit mismatches must be reviewed before proceeding). Aggregate output only.
"""

from __future__ import annotations

import pandas as pd

from certivoi.config import Config
from certivoi.data.extract import all_lab_itemids, connect, extract_outputs


def verify_items(cfg: Config) -> tuple[pd.DataFrame, list[str]]:
    """Return (per-itemid × unit aggregate table, list of problems).

    Checks: itemid present in d_labitems, label contains the expected text,
    fluid matches, and for target labs the dominant unit equals the config unit.
    Counts below ``eval.min_cell_count`` are suppressed in full mode.
    """
    paths = extract_outputs(cfg)
    con = connect(cfg)
    ids = ", ".join(str(i) for i in all_lab_itemids(cfg))
    table = con.execute(f"""
        SELECT d.itemid, d.label, d.fluid, d.category, l.valueuom AS unit,
               count(l.itemid) AS n_rows, count(l.valuenum) AS n_numeric
        FROM '{paths['d_labitems'].as_posix()}' d
        LEFT JOIN '{paths['labs'].as_posix()}' l USING (itemid)
        WHERE d.itemid IN ({ids})
        GROUP BY ALL ORDER BY d.itemid, n_rows DESC""").df()
    con.close()

    expected = {int(k): str(v) for k, v in cfg.get("verify.expected_labels").items()}
    fluid = str(cfg.get("verify.fluid"))
    target_units = {s.itemid: (s.code, s.unit) for s in cfg.labs.values()}
    problems: list[str] = []
    for itemid in all_lab_itemids(cfg):
        rows = table[table["itemid"] == itemid]
        if rows.empty:
            problems.append(f"itemid {itemid} not found in d_labitems")
            continue
        label, fl = rows["label"].iloc[0], rows["fluid"].iloc[0]
        if itemid in expected and expected[itemid].lower() not in str(label).lower():
            problems.append(f"itemid {itemid}: label '{label}' does not contain "
                            f"'{expected[itemid]}'")
        if str(fl) != fluid:
            problems.append(f"itemid {itemid}: fluid '{fl}' != '{fluid}'")
        if itemid in target_units and rows["n_rows"].sum() > 0:
            code, unit = target_units[itemid]
            top = rows.sort_values("n_rows", ascending=False)["unit"].iloc[0]
            if top != unit:
                problems.append(f"{code} ({itemid}): dominant unit '{top}' != config '{unit}'")
    if cfg.is_full:
        from certivoi.utils.io import suppress_small_counts

        table = suppress_small_counts(table, ["n_rows", "n_numeric"],
                                      int(cfg.get("eval.min_cell_count")))
    return table, problems
