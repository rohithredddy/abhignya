"""CertiVOI command-line entry point.

Implements 05_IMPLEMENTATION_ROADMAP Phase 0 (CLI with subcommands). Stages are
wired here as phases are implemented; unimplemented stages exit with a clear
message naming the phase that will add them.

Usage: python scripts/run.py <command> --config configs/default.yaml [options]
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from typing import Any

from certivoi.config import TARGET_LABS, Config, load_config
from certivoi.utils.logging import get_logger
from certivoi.utils.seed import set_all

EXPERIMENTS = ("E1", "E2", "E3", "E4", "E5", "E6")
log = get_logger("certivoi.run")


def _not_yet(phase: int, what: str) -> Callable[[Config, argparse.Namespace], int]:
    def _run(cfg: Config, args: argparse.Namespace) -> int:
        log.error("'%s' is not implemented yet (planned in Phase %d).", what, phase)
        return 2

    return _run


def cmd_info(cfg: Config, args: argparse.Namespace) -> int:
    """Print resolved config summary (no data access)."""
    print(f"config:      {cfg.source}")
    print(f"config_hash: {cfg.hash}")
    print(f"data_mode:   {cfg.data_mode}")
    print(f"data_root:   {cfg.data_root}")
    print(f"sample_frac: {cfg.sample_frac}")
    print(f"threads:     {cfg.threads}")
    print(f"seed:        {cfg.seed}")
    print(f"labs:        {', '.join(cfg.labs)}")
    return 0


COMMANDS: dict[str, tuple[str, Callable[[Config, argparse.Namespace], int]]] = {
    "info": ("show resolved config and its hash", cmd_info),
    "extract": ("raw CSV.gz -> filtered Parquet (DuckDB)", _not_yet(1, "extract")),
    "cohort": ("cohort steps + flow table", _not_yet(1, "cohort")),
    "split": ("patient-level and temporal splits", _not_yet(1, "split")),
    "labels": ("action labels a0, D, N, S", _not_yet(2, "labels")),
    "features": ("strictly-prior feature matrices", _not_yet(3, "features")),
    "train": ("fit quantile models and baselines", _not_yet(4, "train")),
    "exp": ("run one experiment E1-E6", _not_yet(2, "exp")),
    "all": ("run the whole pipeline", _not_yet(7, "all")),
}


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default="configs/default.yaml", help="base YAML config")
    common.add_argument("--override", default=None, help="optional YAML merged over --config")
    common.add_argument("--sample-frac", type=float, default=None,
                        help="patient sample fraction for smoke tests, e.g. 0.05")
    common.add_argument("--force", action="store_true", help="ignore cached interim artifacts")

    parser = argparse.ArgumentParser(
        prog="run.py",
        description="CertiVOI pipeline (research prototype, not for clinical use).",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="command")
    for name, (help_text, _) in COMMANDS.items():
        p = sub.add_parser(name, parents=[common], help=help_text, description=help_text)
        if name == "train":
            p.add_argument("--lab", choices=TARGET_LABS, default=None,
                           help="train one lab only (default: all)")
        if name == "exp":
            p.add_argument("exp_id", choices=EXPERIMENTS, help="experiment id")
            p.add_argument("--lab", choices=TARGET_LABS, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    overrides: dict[str, Any] = {}
    if args.sample_frac is not None:
        overrides["data"] = {"sample_frac": args.sample_frac}
    cfg = load_config(args.config, override_path=args.override, overrides=overrides)
    set_all(cfg.seed)
    if cfg.is_full:
        log.warning("data_mode=full: share only aggregate results/ (CONTRIBUTING.md §4).")
    return COMMANDS[args.command][1](cfg, args)


if __name__ == "__main__":
    sys.exit(main())
