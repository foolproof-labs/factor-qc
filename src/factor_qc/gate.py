"""The fail-closed quality gate.

``factor_qc.stats`` computes; this module decides — and its default answer
is *no*.  A backtest that refuses to declare how many configurations were
tried is refused outright: without an honest ``n_trials`` there is no
deflation benchmark, no haircut and no track-record floor, and any verdict
would be theatre.

Checks are graded by severity:

- **P0 (fatal)** — the candidate must not pass:
  - Deflated Sharpe Ratio below threshold (selection-bias corrected),
  - Probability of Backtest Overfitting above threshold (when a trials
    matrix is provided),
  - multiple-testing haircut annual Sharpe below the floor,
  - Minimum Track Record Length longer than the available sample.
- **P1 (warning)** — proceed with eyes open:
  - PSR vs zero below 0.95 (weak evidence even before deflation),
  - sample shorter than one year (252 obs),
  - trials count aggressive relative to sample (n_trials > n_obs / 5).
- **P2 (info)** — recorded, no action:
  - non-normal return moments (skew / kurtosis),
  - very few trials (n_trials < 5).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .stats import (
    DSR_THRESHOLD,
    PBO_THRESHOLD,
    build_overfit_report,
    minimum_track_record_length,
    probabilistic_sharpe_ratio,
    render_report_text,
    sharpe_ratio,
    skew_kurt,
)

SAFETY = {
    "production_effect": False,
    "changes_probability": False,
    "allow_real_trade": False,
}
MIN_YEAR_OBS = 252
ADJUSTED_SHARPE_FLOOR = 0.5
PSR_ZERO_THRESHOLD = 0.95
TRIAL_AGGRESSION_DENOM = 5.0
MIN_TRIALS_INFO = 5


def run_gate(
    returns: np.ndarray,
    n_trials: int | None,
    *,
    trials_matrix: np.ndarray | None = None,
    periods_per_year: int = 252,
    dsr_threshold: float = DSR_THRESHOLD,
    pbo_threshold: float = PBO_THRESHOLD,
    adjusted_sharpe_floor: float = ADJUSTED_SHARPE_FLOOR,
    require_declared_trials: bool = True,
    n_blocks: int = 16,
) -> dict[str, Any]:
    """Run the fail-closed gate. Returns a dict with ``passed``, ``checks``
    and the underlying statistics report.

    When ``require_declared_trials`` is true (default) and ``n_trials`` is
    None, the gate refuses: ``passed=False`` with a single P0 blocker
    ``n_trials_declaration_required``.

    ``n_blocks`` controls the CSCV granularity of the PBO check; note that
    CSCV enumerates C(n_blocks, n_blocks/2) splits (12,870 for the default
    16), so smaller values (e.g. 8 or 10) run much faster on large trial
    matrices.
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    checks: list[dict[str, Any]] = []

    def add(check_id: str, severity: str, title: str, value: Any, threshold: Any, passed: bool) -> None:
        checks.append(
            {
                "check_id": check_id,
                "severity": severity,
                "title": title,
                "value": value,
                "threshold": threshold,
                "passed": passed,
            }
        )

    if n_trials is None:
        if require_declared_trials:
            add(
                "n_trials_declaration_required",
                "P0",
                "honest n_trials declaration",
                None,
                "declared integer >= 1",
                False,
            )
            return {
                "passed": False,
                "verdict": "FAIL - n_trials must be declared before a backtest "
                           "can be judged (fail-closed)",
                "checks": checks,
                "report": None,
                "safety": SAFETY,
            }
        n_trials = 1

    n_obs = int(r.size)
    report = build_overfit_report(
        r,
        n_trials,
        trials_matrix=trials_matrix,
        periods_per_year=periods_per_year,
        dsr_threshold=dsr_threshold,
        pbo_threshold=pbo_threshold,
        n_blocks=n_blocks,
    )
    dsr = report["deflated_sharpe_ratio"]
    pbo = report["pbo"]
    ann = np.sqrt(periods_per_year)
    sr_pp = sharpe_ratio(r)
    skew, kurt = skew_kurt(r)
    mintrl = minimum_track_record_length(sr_pp, 0.0, skew, kurt)
    psr_zero = probabilistic_sharpe_ratio(sr_pp, 0.0, n_obs, skew, kurt)
    adjusted_annual = report["haircut"]["adjusted_sharpe_annual"]

    # P0: fatal
    add(
        "dsr",
        "P0",
        "deflated sharpe ratio >= threshold",
        round(dsr, 4),
        dsr_threshold,
        bool(dsr >= dsr_threshold),
    )
    if pbo is not None:
        add(
            "pbo",
            "P0",
            "probability of backtest overfitting <= threshold",
            round(pbo["pbo"], 4),
            pbo_threshold,
            bool(pbo["pbo"] <= pbo_threshold),
        )
    add(
        "haircut_sharpe",
        "P0",
        "multiple-testing haircut annual sharpe >= floor",
        round(adjusted_annual, 4),
        adjusted_sharpe_floor,
        bool(adjusted_annual >= adjusted_sharpe_floor),
    )
    add(
        "mintrl",
        "P0",
        "minimum track record length <= sample",
        round(mintrl, 1) if np.isfinite(mintrl) else None,
        f"<= {n_obs}",
        bool(np.isfinite(mintrl) and mintrl <= n_obs),
    )

    # P1: warnings
    add(
        "psr_vs_zero",
        "P1",
        "probabilistic sharpe vs zero >= threshold",
        round(psr_zero, 4) if np.isfinite(psr_zero) else None,
        PSR_ZERO_THRESHOLD,
        bool(np.isfinite(psr_zero) and psr_zero >= PSR_ZERO_THRESHOLD),
    )
    add(
        "sample_length",
        "P1",
        "sample >= one year of observations",
        n_obs,
        MIN_YEAR_OBS,
        bool(n_obs >= MIN_YEAR_OBS),
    )
    add(
        "trial_aggression",
        "P1",
        "trials not aggressive relative to sample",
        n_trials,
        f"<= {n_obs / TRIAL_AGGRESSION_DENOM:.0f}",
        bool(n_trials <= n_obs / TRIAL_AGGRESSION_DENOM),
    )

    # P2: info
    add(
        "return_moments",
        "P2",
        "return moments near-normal",
        {"skew": round(skew, 4), "kurtosis": round(kurt, 4)},
        "skew ~ 0, kurtosis ~ 3",
        bool(abs(skew) < 0.5 and abs(kurt - 3.0) < 1.0),
    )
    add(
        "trial_count",
        "P2",
        "trials count >= minimal",
        n_trials,
        MIN_TRIALS_INFO,
        bool(n_trials >= MIN_TRIALS_INFO),
    )

    p0_failed = [check for check in checks if check["severity"] == "P0" and not check["passed"]]
    passed = not p0_failed
    if passed:
        verdict = "PASS - survives multiple-testing correction (no P0 failures)"
    else:
        detail = "; ".join(
            f"{check['check_id']}: {check['value']} vs {check['threshold']}"
            for check in p0_failed
        )
        verdict = f"FAIL - P0 blocker(s): {detail}"
    return {
        "passed": passed,
        "verdict": verdict,
        "checks": checks,
        "report": report,
        "report_text": render_report_text(report),
        "safety": SAFETY,
    }
