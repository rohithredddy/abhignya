"""Panel-level (specimen-level) certified deferral.

Implements 02_METHODOLOGY §5.6. Labs drawn in the same tube cannot be skipped
one at a time: HGB and PLT share a CBC draw, K and CREAT share a chemistry
draw. A draw is avoidable only if *every* lab in it is low value, and it misses
an action change if *any* lab changes action:

    D_panel = max_j D_j,      loss_panel(λ) = D_panel · 1[s_panel <= λ].

CRC on panel scores therefore certifies the draw-level miss rate at α. By
contrast, combining per-lab certificates ("skip if every lab is flagged at α")
only bounds it by Σ_j α_j (union bound), i.e. up to 2α for two labs.
CRC validity holds for any panel score; the score only affects efficiency.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from certivoi.conformal.crc import crc_threshold

KEYS = ["subject_id", "hadm_id", "t"]


def pair_panel(frames: dict[str, pd.DataFrame], score: str = "score_ours") -> pd.DataFrame:
    """Inner-join per-lab event frames on (subject, admission, draw time).

    Only draws where every panel lab is a repeat event are kept. Returns one row
    per draw with ``s_<LAB>``, ``D_<LAB>``, ``split`` and ``D_panel``.
    """
    out = None
    for lab, df in frames.items():
        part = df[[*KEYS, "split", score, "D"]].rename(
            columns={score: f"s_{lab}", "D": f"D_{lab}"}).drop_duplicates(KEYS)
        if out is None:
            out = part
        else:
            out = out.merge(part.drop(columns="split"), on=KEYS, how="inner")
    labs = list(frames)
    out = out.reset_index(drop=True)
    out["D_panel"] = out[[f"D_{lab}" for lab in labs]].max(axis=1).astype(int)
    return out


def panel_score(s: np.ndarray, how: str) -> np.ndarray:
    """Combine per-lab P(action change) (n, k) into one draw-level score."""
    s = np.clip(np.asarray(s, float), 0.0, 1.0)
    if how == "max":
        return s.max(axis=1)
    if how == "noisy_or":
        return 1.0 - np.prod(1.0 - s, axis=1)
    raise ValueError(f"unknown panel score '{how}'")


def naive_flags(cal: pd.DataFrame, ev: pd.DataFrame, labs: list[str], alpha_each: float
                ) -> np.ndarray:
    """Skip a draw when every lab is flagged by its own per-lab CRC at ``alpha_each``.

    Per-lab thresholds are fitted on the per-lab marginal of the paired draws.
    """
    flag = np.ones(len(ev), dtype=bool)
    for lab in labs:
        lam = crc_threshold(cal[f"s_{lab}"].to_numpy(), cal[f"D_{lab}"].to_numpy(), alpha_each)
        flag &= ev[f"s_{lab}"].to_numpy() <= lam
    return flag
