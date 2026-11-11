"""Verify lab itemids against d_labitems (aggregate only); exit 1 on any mismatch.

Implements 03_DATA_AND_COHORT §4. Usage:
    python scripts/01_verify_items.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import sys

from certivoi.config import load_config
from certivoi.data.extract import extract
from certivoi.data.verify import verify_items
from certivoi.utils.io import write_results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    extract(cfg)
    table, problems = verify_items(cfg)
    write_results(cfg, "verify_items", {"items": table}, problems=problems)
    print(table.to_string(index=False))  # aggregate counts only
    if problems:
        print("\nMISMATCH — stop and review (CONTRIBUTING.md §11):", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1
    print("\nAll itemids verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
