"""Independent backtest-overfit statistics (DSR / PBO / haircut / MinTRL).

Implementation of the published methods:

- Deflated / Probabilistic Sharpe Ratio and Minimum Track Record Length:
  Bailey & Lopez de Prado (2012, 2014).
- Probability of Backtest Overfitting (PBO) via Combinatorially-Symmetric
  Cross-Validation: Bailey, Borwein, Lopez de Prado & Zhu (2017).
- Multiple-testing haircut of the Sharpe Ratio: Harvey & Liu (2015).

Implementation notes
--------------------
Only numpy + the standard library are required (no scipy).  The
standard-normal inverse CDF uses Acklam's rational approximation with one
Newton refinement.  All Sharpe ratios are per-observation (not annualised);
the report builder annualises only for display.

This module computes; it never decides.  The fail-closed gate lives in
``factor_qc.gate``.
"""

from __future__ import annotations

import math
from datetime import datetime
from itertools import combinations
from typing import Any, Sequence

import numpy as np

EULER_MASCHERONI = 0.5772156649015328606
DSR_THRESHOLD = 0.95
PBO_THRESHOLD = 0.50


# --------------------------------------------------------------------------- #
# Standard normal helpers (no scipy)
# --------------------------------------------------------------------------- #
def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _norm_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF (Acklam's rational approximation + Newton step)."""
    p = min(max(p, 1e-16), 1.0 - 1e-16)
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]
    plow = 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    elif p <= 1.0 - plow:
        q = p - 0.5
        r = q * q
        x = (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
        ) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    error = _norm_cdf(x) - p
    u = error * math.sqrt(2.0 * math.pi) * math.exp(x * x / 2.0)
    return x - u / (1.0 + x * u / 2.0)


def _norm_isf(p: float) -> float:
    return _norm_ppf(1.0 - p)


# --------------------------------------------------------------------------- #
# Sharpe ratio and moments
# --------------------------------------------------------------------------- #
def sharpe_ratio(returns: np.ndarray, benchmark: float = 0.0) -> float:
    """Per-period Sharpe ratio (ddof=1); NaN when fewer than 2 observations."""
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if r.size < 2:
        return float("nan")
    sd = r.std(ddof=1)
    if sd == 0:
        return float("nan")
    return float((r.mean() - benchmark) / sd)


def skew_kurt(returns: np.ndarray) -> tuple[float, float]:
    """Sample skewness (g1) and non-excess kurtosis (g2, normal == 3)."""
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if r.size < 4:
        return 0.0, 3.0
    mean = r.mean()
    sd = r.std(ddof=0)
    if sd == 0:
        return 0.0, 3.0
    centered = (r - mean) / sd
    return float(np.mean(centered ** 3)), float(np.mean(centered ** 4))


# --------------------------------------------------------------------------- #
# Deflated Sharpe Ratio / PSR / MinTRL
# --------------------------------------------------------------------------- #
def probabilistic_sharpe_ratio(
    observed_sr: float,
    benchmark_sr: float,
    n_obs: int,
    skew: float,
    kurtosis: float,
) -> float:
    """P(SR > SR*) under the non-normal Sharpe estimator standard error."""
    if n_obs < 2 or math.isnan(observed_sr):
        return float("nan")
    denom = 1.0 - skew * observed_sr + ((kurtosis - 1.0) / 4.0) * observed_sr ** 2
    denom = max(denom, 1e-12)
    se = math.sqrt(denom / (n_obs - 1))
    return _norm_cdf((observed_sr - benchmark_sr) / se)


def expected_max_sharpe(sr_variance_across_trials: float, n_trials: int) -> float:
    """E[max SR] across N independent zero-true-SR trials (deflation benchmark)."""
    if n_trials < 2:
        return 0.0
    variance = max(sr_variance_across_trials, 0.0)
    z1 = _norm_ppf(1.0 - 1.0 / n_trials)
    z2 = _norm_ppf(1.0 - 1.0 / (n_trials * math.e))
    return float(
        math.sqrt(variance)
        * ((1.0 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2)
    )


def deflated_sharpe_ratio(
    strategy_returns: np.ndarray,
    n_trials: int,
    *,
    sr_variance_across_trials: float | None = None,
    all_trial_sharpes: Sequence[float] | None = None,
    threshold: float = DSR_THRESHOLD,
) -> dict[str, Any]:
    r = np.asarray(strategy_returns, dtype=float)
    r = r[~np.isnan(r)]
    n = r.size
    sr = sharpe_ratio(r)
    skew, kurt = skew_kurt(r)
    if sr_variance_across_trials is None:
        if all_trial_sharpes is not None and len(all_trial_sharpes) > 1:
            sr_variance_across_trials = float(
                np.var(np.asarray(all_trial_sharpes, dtype=float), ddof=1)
            )
        else:
            denom = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr ** 2
            sr_variance_across_trials = max(denom, 1e-12) / max(n - 1, 1)
    sr0 = expected_max_sharpe(sr_variance_across_trials, n_trials)
    return {
        "observed_sharpe": sr,
        "deflated_benchmark_sr0": sr0,
        "psr_vs_zero": probabilistic_sharpe_ratio(sr, 0.0, n, skew, kurt),
        "deflated_sharpe_ratio": probabilistic_sharpe_ratio(sr, sr0, n, skew, kurt),
        "n_obs": n,
        "n_trials": n_trials,
        "skew": skew,
        "kurtosis": kurt,
        "passed": bool(
            probabilistic_sharpe_ratio(sr, sr0, n, skew, kurt) >= threshold
        ),
    }


def minimum_track_record_length(
    observed_sr: float,
    benchmark_sr: float,
    skew: float,
    kurtosis: float,
    confidence: float = 0.95,
) -> float:
    if observed_sr <= benchmark_sr:
        return float("inf")
    z = _norm_ppf(confidence)
    num = max(
        1.0 - skew * observed_sr + ((kurtosis - 1.0) / 4.0) * observed_sr ** 2,
        1e-12,
    )
    return float(1.0 + num * (z / (observed_sr - benchmark_sr)) ** 2)


# --------------------------------------------------------------------------- #
# Multiple-testing haircut (Harvey & Liu 2015)
# --------------------------------------------------------------------------- #
def _p_from_t(tstat: float) -> float:
    return 2.0 * _norm_sf(abs(tstat))


def _adjusted_p(p: float, n_tests: int, method: str, rank: int = 1) -> float:
    method = method.lower()
    if method == "bonferroni":
        return min(1.0, p * n_tests)
    if method == "holm":
        return min(1.0, p * (n_tests - rank + 1))
    if method == "bhy":
        harmonic = sum(1.0 / i for i in range(1, n_tests + 1))
        return min(1.0, p * n_tests * harmonic / rank)
    raise ValueError(f"unknown method: {method}")


def haircut_sharpe(
    observed_sharpe_per_period: float,
    n_obs: int,
    n_tests: int,
    method: str = "bonferroni",
    rank: int = 1,
) -> dict[str, Any]:
    t_obs = observed_sharpe_per_period * math.sqrt(n_obs)
    p_obs = _p_from_t(t_obs)
    p_adj = _adjusted_p(p_obs, n_tests, method, rank)
    # The inverse-CDF approximation is only meaningful down to ~1e-15; clip
    # extreme significance so the adjusted Sharpe stays finite.
    p_adj = min(1.0, max(p_adj, 1e-15))
    t_adj = math.copysign(_norm_isf(p_adj / 2.0), observed_sharpe_per_period)
    sr_adj = t_adj / math.sqrt(n_obs)
    haircut = (
        1.0 - sr_adj / observed_sharpe_per_period
        if observed_sharpe_per_period
        else float("nan")
    )
    return {
        "method": method,
        "observed_sharpe": observed_sharpe_per_period,
        "adjusted_sharpe": sr_adj,
        "haircut": haircut,
        "observed_pvalue": p_obs,
        "adjusted_pvalue": p_adj,
        "n_tests": n_tests,
    }


# --------------------------------------------------------------------------- #
# Probability of Backtest Overfitting (CSCV, Bailey et al. 2017)
# --------------------------------------------------------------------------- #
def _sharpe_cols(block: np.ndarray) -> np.ndarray:
    mean = np.nanmean(block, axis=0)
    sd = np.nanstd(block, axis=0, ddof=1)
    sd = np.where(sd == 0, np.nan, sd)
    return mean / sd


def probability_of_backtest_overfitting(
    perf_matrix: np.ndarray,
    n_blocks: int = 16,
) -> dict[str, Any]:
    matrix = np.asarray(perf_matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("perf_matrix must be 2D (T x N)")
    t_rows, n_strategies = matrix.shape
    if n_strategies < 2:
        raise ValueError("need at least 2 strategy configurations for PBO")
    if n_blocks % 2 != 0:
        raise ValueError("n_blocks must be even")
    if n_blocks > t_rows:
        raise ValueError("n_blocks cannot exceed observations")

    block_idx = np.array_split(np.arange(t_rows), n_blocks)
    blocks = list(range(n_blocks))
    logits: list[float] = []
    oos_ranks: list[float] = []

    for is_blocks in combinations(blocks, n_blocks // 2):
        is_set = set(is_blocks)
        is_rows = np.concatenate([block_idx[b] for b in blocks if b in is_set])
        oos_rows = np.concatenate([block_idx[b] for b in blocks if b not in is_set])
        is_perf = _sharpe_cols(matrix[is_rows])
        oos_perf = _sharpe_cols(matrix[oos_rows])
        if np.all(np.isnan(is_perf)):
            continue
        n_star = int(np.nanargmax(is_perf))
        valid = ~np.isnan(oos_perf)
        rank = float(np.sum(oos_perf[valid] <= oos_perf[n_star]))
        w = rank / (float(np.sum(valid)) + 1.0)
        w = min(max(w, 1e-6), 1.0 - 1e-6)
        logits.append(float(np.log(w / (1.0 - w))))
        oos_ranks.append(w)

    pbo = (
        float(np.mean([1.0 if lam <= 0 else 0.0 for lam in logits]))
        if logits
        else float("nan")
    )
    return {
        "pbo": pbo,
        "n_splits": len(logits),
        "n_strategies": n_strategies,
        "n_blocks": n_blocks,
        "median_logit": float(np.median(logits)) if logits else float("nan"),
    }


# --------------------------------------------------------------------------- #
# Report builder
# --------------------------------------------------------------------------- #
def build_overfit_report(
    selected_returns: np.ndarray,
    n_trials: int,
    *,
    trials_matrix: np.ndarray | None = None,
    periods_per_year: int = 252,
    dsr_threshold: float = DSR_THRESHOLD,
    pbo_threshold: float = PBO_THRESHOLD,
    n_blocks: int = 16,
    haircut_method: str = "bonferroni",
) -> dict[str, Any]:
    r = np.asarray(selected_returns, dtype=float)
    r = r[~np.isnan(r)]
    ann = math.sqrt(periods_per_year)
    sr_pp = sharpe_ratio(r)
    skew, kurt = skew_kurt(r)

    all_trial_sharpes = None
    if trials_matrix is not None:
        tm = np.asarray(trials_matrix, dtype=float)
        all_trial_sharpes = [
            sharpe_ratio(tm[:, column]) for column in range(tm.shape[1])
        ]
        all_trial_sharpes = [value for value in all_trial_sharpes if not math.isnan(value)]
        n_trials = max(n_trials, len(all_trial_sharpes))

    dsr = deflated_sharpe_ratio(
        r,
        n_trials,
        all_trial_sharpes=all_trial_sharpes,
        threshold=dsr_threshold,
    )
    hc = haircut_sharpe(sr_pp, r.size, n_trials, method=haircut_method)
    mintrl = minimum_track_record_length(sr_pp, 0.0, skew, kurt)

    pbo_block = None
    if trials_matrix is not None and np.asarray(trials_matrix).shape[1] >= 2:
        pbo_block = probability_of_backtest_overfitting(
            np.asarray(trials_matrix, dtype=float),
            n_blocks=n_blocks,
        )

    flags: list[str] = []
    if dsr["deflated_sharpe_ratio"] < dsr_threshold:
        flags.append(f"DSR {dsr['deflated_sharpe_ratio']:.2f} < {dsr_threshold}")
    if pbo_block is not None and pbo_block["pbo"] > pbo_threshold:
        flags.append(f"PBO {pbo_block['pbo']:.2f} > {pbo_threshold}")
    if hc["adjusted_sharpe"] * ann < 0.5:
        flags.append(f"haircut Sharpe {hc['adjusted_sharpe'] * ann:.2f} < 0.5")
    if mintrl > r.size:
        flags.append(f"MinTRL {mintrl:.0f} > sample {r.size}")

    passed = len(flags) == 0
    verdict = (
        "PASS - survives multiple-testing correction"
        if passed
        else "FAIL - likely overfit / selection-biased: " + "; ".join(flags)
    )
    return {
        "verdict": verdict,
        "passed": passed,
        "observed_sharpe_annual": round(sr_pp * ann, 4),
        "skew": round(skew, 4),
        "kurtosis": round(kurt, 4),
        "n_obs": int(r.size),
        "n_trials": int(n_trials),
        "deflated_sharpe_ratio": round(dsr["deflated_sharpe_ratio"], 4),
        "deflation_benchmark_sr0_annual": round(dsr["deflated_benchmark_sr0"] * ann, 4),
        "psr_vs_zero": round(dsr["psr_vs_zero"], 4),
        "haircut": {
            "method": hc["method"],
            "adjusted_sharpe_annual": round(hc["adjusted_sharpe"] * ann, 4),
            "haircut_pct": round(hc["haircut"], 4),
            "observed_pvalue": hc["observed_pvalue"],
            "adjusted_pvalue": hc["adjusted_pvalue"],
        },
        "minimum_track_record_length": round(mintrl, 1),
        "pbo": pbo_block,
    }


def render_report_text(report: dict[str, Any]) -> str:
    lines = [
        "=" * 64,
        " BACKTEST OVERFITTING REPORT",
        "=" * 64,
        f" Verdict : {report['verdict']}",
        "-" * 64,
        f" Observed Sharpe (annual) : {report['observed_sharpe_annual']}",
        f" Trials (multiple tests)  : {report['n_trials']}",
        f" Observations             : {report['n_obs']}",
        f" Skew / Kurtosis          : {report['skew']} / {report['kurtosis']}",
        "-" * 64,
        f" Deflated Sharpe Ratio    : {report['deflated_sharpe_ratio']} "
        f"(benchmark SR0 {report['deflation_benchmark_sr0_annual']} ann.)",
        f" PSR vs 0                 : {report['psr_vs_zero']}",
        f" Haircut Sharpe ({report['haircut']['method']}): "
        f"{report['haircut']['adjusted_sharpe_annual']} "
        f"(-{report['haircut']['haircut_pct']:.0%})",
        f" Min Track Record Length  : {report['minimum_track_record_length']} obs",
    ]
    if report["pbo"] is not None:
        lines.append(
            f" PBO                      : {report['pbo']['pbo']} "
            f"({report['pbo']['n_splits']} CSCV splits)"
        )
    lines.append("=" * 64)
    return "\n".join(lines)


def build_check_artifact(
    *,
    name: str,
    source: str,
    selected_returns: np.ndarray,
    n_trials: int,
    trials_matrix: np.ndarray | None = None,
    periods_per_year: int = 252,
    n_blocks: int = 16,
    returns_meta: dict[str, Any] | None = None,
    trials_meta: dict[str, Any] | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Provenance-wrapped check artifact (schema ``factor_qc.check.v1``)."""
    report = build_overfit_report(
        selected_returns,
        n_trials,
        trials_matrix=trials_matrix,
        periods_per_year=periods_per_year,
        n_blocks=n_blocks,
    )
    artifact: dict[str, Any] = {
        "schema_version": "factor_qc.check.v1",
        "generated_at": generated_at
        or datetime.now().astimezone().isoformat(timespec="seconds"),
        "tool": "factor_qc.stats (independent engine, numpy only)",
        "name": name,
        "declared": {
            "n_trials": n_trials,
            "periods_per_year": periods_per_year,
            "haircut_method": report["haircut"]["method"],
        },
        "inputs": {
            "returns": returns_meta or {},
            "trials": trials_meta,
        },
        "source": source,
        "boundaries": {
            "production_effect": False,
            "changes_probability": False,
            "allow_real_trade": False,
        },
        "report": report,
    }
    if trials_matrix is None and report.get("pbo") is None:
        artifact["limitations"] = [
            "trials matrix not provided; PBO not computed (report pbo=null) and "
            "the DSR cross-trial variance degrades to a conservative "
            "single-trial estimate."
        ]
    return artifact
