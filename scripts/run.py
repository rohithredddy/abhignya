"""CertiVOI command-line entry point.

Implements 05_IMPLEMENTATION_ROADMAP (CLI with subcommands). Every stage builds
(or loads cached) upstream inputs, so any command can run on a fresh checkout.

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

EXPERIMENTS = ("E1", "E2", "E3", "E4", "E5", "E6", "E7")
log = get_logger("certivoi.run")


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


def cmd_extract(cfg: Config, args: argparse.Namespace) -> int:
    from certivoi.data.extract import extract

    extract(cfg, force=args.force)
    return 0


def cmd_verify(cfg: Config, args: argparse.Namespace) -> int:
    from certivoi.data.extract import extract
    from certivoi.data.verify import verify_items
    from certivoi.utils.io import write_results

    extract(cfg)
    table, problems = verify_items(cfg)
    write_results(cfg, "verify_items", {"items": table}, problems=problems)
    for p in problems:
        log.error("itemid check: %s", p)
    if problems:
        log.error("Item verification failed — stop and review (CONTRIBUTING.md §11).")
        return 1
    log.info("All itemids verified.")
    return 0


def cmd_cohort(cfg: Config, args: argparse.Namespace) -> int:
    from certivoi.data.cohort import build_cohort

    build_cohort(cfg, force=args.force)
    return 0


def cmd_split(cfg: Config, args: argparse.Namespace) -> int:
    from certivoi.data.split import make_splits

    make_splits(cfg, force=args.force)
    return 0


def cmd_labels(cfg: Config, args: argparse.Namespace) -> int:
    from certivoi.labels.build import build_labels

    build_labels(cfg, force=args.force)
    return 0


def cmd_features(cfg: Config, args: argparse.Namespace) -> int:
    from certivoi.features.build import build_features

    build_features(cfg, force=args.force)
    return 0


def cmd_train(cfg: Config, args: argparse.Namespace) -> int:
    from certivoi.models.train import train_lab

    for lab in [args.lab] if getattr(args, "lab", None) else list(cfg.labs):
        train_lab(cfg, lab, force=args.force)
    return 0


def run_experiment(cfg: Config, exp_id: str) -> None:
    from certivoi.experiments import e1, e2, e3, e4, e5, e6, e7

    modules = {"E1": e1, "E2": e2, "E3": e3, "E4": e4, "E5": e5, "E6": e6, "E7": e7}
    log.info("running %s", exp_id)
    modules[exp_id].run(cfg)


def cmd_exp(cfg: Config, args: argparse.Namespace) -> int:
    run_experiment(cfg, args.exp_id)
    return 0


def cmd_all(cfg: Config, args: argparse.Namespace) -> int:
    """Full pipeline: verify items, build every stage, run E1–E7."""
    if cmd_verify(cfg, args) != 0:
        return 1
    for step in (cmd_cohort, cmd_split, cmd_labels, cmd_features, cmd_train):
        step(cfg, args)
    for exp_id in EXPERIMENTS:
        run_experiment(cfg, exp_id)
    log.info("all done: results in %s", cfg.results_dir)
    return 0


COMMANDS: dict[str, tuple[str, Callable[[Config, argparse.Namespace], int]]] = {
    "info": ("show resolved config and its hash", cmd_info),
    "extract": ("raw CSV.gz -> filtered Parquet (DuckDB)", cmd_extract),
    "verify": ("aggregate itemid/unit verification (fails on mismatch)", cmd_verify),
    "cohort": ("cohort steps + flow table", cmd_cohort),
    "split": ("patient-level and temporal splits", cmd_split),
    "labels": ("action labels a0, D, N, S", cmd_labels),
    "features": ("strictly-prior feature matrices", cmd_features),
    "train": ("fit quantile models and baselines", cmd_train),
    "exp": ("run one experiment E1-E7", cmd_exp),
    "all": ("run the whole pipeline (verify, stages, E1-E7)", cmd_all),
}


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default="configs/default.yaml", help="base YAML config")
    common.add_argument("--override", default=None, help="optional YAML merged over --config")
    common.add_argument("--sample-frac", type=float, default=None,
                        help="patient sample fraction for smoke tests, e.g. 0.05")
    common.add_argument("--force", action="store_true",
                        help="recompute this stage even if a cached artifact exists")

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
