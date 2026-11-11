"""Test-facing entry point for synthetic data (CONTRIBUTING.md §4.5).

The generator lives in ``certivoi.data.synthetic`` so the CLI can also build a
synthetic dataset (``data.mode: synthetic``); tests import it from here.
"""

from certivoi.data.synthetic import (  # noqa: F401
    DECOY_K_ITEMID,
    DEFAULT_CONFIG,
    make_synthetic_mimic,
    write_mimic_layout,
)
