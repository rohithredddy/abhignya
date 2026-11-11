"""Global seeding. Implements CONTRIBUTING.md §7 (seeds via ``set_all``)."""

from __future__ import annotations

import os
import random

import numpy as np


def set_all(seed: int) -> np.random.Generator:
    """Seed Python ``random``, NumPy's legacy global RNG and ``PYTHONHASHSEED``.

    Inputs: integer seed. Output: a fresh ``np.random.Generator`` seeded with
    ``seed``; prefer passing this generator explicitly over using global state.
    LightGBM / scikit-learn seeds are passed per model from the config seed.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - legacy global RNG, seeded for third-party code
    return np.random.default_rng(seed)
