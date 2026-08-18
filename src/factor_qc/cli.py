"""Command-line interface for factor-qc.

Subcommands:

- ``check``   run the fail-closed gate on a returns series
- ``version`` print version
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .gate import run_gate


def _load_returns(path: str) -> np.ndarray:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(x, (int, float)) for x in value):
        raise ValueError(f"returns must be a JSON list of numbers: {path}")
    return np.asarray(value, dtype=float)


def _load_trials(path: str) -> np.ndarray:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, list) or not value:
        raise ValueError(f"trials must be a non-empty JSON list of lists: {path}")
    rows = []
    for row in value:
        if not isinstance(row, list):
            raise ValueError(f"trials rows must be lists: {path}")
        rows.append([float(x) for x in row])
    matrix = np.asarray(rows, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"trials must be 2D (T x N): {path}")
    return matrix


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qc",
        description="Fail-closed quality gate for backtests.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="run the fail-closed gate")
    check.add_argument("--returns", required=True, help="JSON list of per-period returns")
    check.add_argument("--trials", default=None, help="JSON 2D matrix (T x N) of trial returns")
    check.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="honest number of configurations tried (required, fail-closed)",
    )
    check.add_argument("--periods-per-year", type=int, default=252)
    check.add_argument(
        "--n-blocks",
        type=int,
        default=16,
        help="CSCV blocks for PBO (default 16 = 12,870 splits; use 8 or 10 for speed)",
    )
    check.add_argument("--json", action="store_true", help="machine-readable output")

    sub.add_parser("version", help="print version")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "version":
        print(__version__)
        return 0

    if args.command == "check":
        returns = _load_returns(args.returns)
        trials = _load_trials(args.trials) if args.trials else None
        if args.n_trials is not None and args.n_trials < 1:
            parser.error("--n-trials must be >= 1")
        body: dict[str, Any] = run_gate(
            returns,
            args.n_trials,
            trials_matrix=trials,
            periods_per_year=args.periods_per_year,
            n_blocks=args.n_blocks,
        )
        if args.json:
            print(json.dumps(body, ensure_ascii=False, indent=2))
        else:
            print(body["verdict"])
            if body["report"] is not None:
                print(body["report_text"])
            for check in body["checks"]:
                marker = "PASS" if check["passed"] else "FAIL"
                print(
                    f" [{marker}] {check['severity']} {check['check_id']}: "
                    f"{check['title']} (value={check['value']}, "
                    f"threshold={check['threshold']})"
                )
        return 0 if body["passed"] else 1

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
