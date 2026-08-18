"""factor-qc: fail-closed quality gate for backtests.

One numpy-only engine for the standard backtest-overfit statistics —
Deflated Sharpe Ratio, Probability of Backtest Overfitting (CSCV),
Harvey-Liu multiple-testing haircut, Minimum Track Record Length — wrapped
in a gate that refuses to judge a backtest that does not declare how many
configurations were tried.
"""

from .gate import run_gate
from .stats import (
    build_check_artifact,
    build_overfit_report,
    deflated_sharpe_ratio,
    haircut_sharpe,
    minimum_track_record_length,
    probability_of_backtest_overfitting,
    sharpe_ratio,
    skew_kurt,
)

__version__ = "0.1.0"

__all__ = [
    "build_check_artifact",
    "build_overfit_report",
    "deflated_sharpe_ratio",
    "haircut_sharpe",
    "minimum_track_record_length",
    "probability_of_backtest_overfitting",
    "run_gate",
    "sharpe_ratio",
    "skew_kurt",
]
