"""Numerical-reference tests for the overfit statistics engine.

Reference values come from a deterministic published demo (rng 42,
T=1000, N=200); this implementation is independent and uses those outputs
only as a numerical oracle.
"""

from __future__ import annotations

import numpy as np
import pytest

from factor_qc.stats import (
    build_check_artifact,
    build_overfit_report,
    deflated_sharpe_ratio,
    haircut_sharpe,
    minimum_track_record_length,
    probability_of_backtest_overfitting,
    sharpe_ratio,
)


def _demo_noise() -> tuple[np.ndarray, int]:
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.01, size=(1000, 200))
    sharpes = noise.mean(0) / noise.std(0, ddof=1)
    return noise, int(np.argmax(sharpes))


def test_dsr_haircut_mintrl_match_reference() -> None:
    noise, best = _demo_noise()
    trial_sharpes = [sharpe_ratio(noise[:, column]) for column in range(noise.shape[1])]
    dsr = deflated_sharpe_ratio(
        noise[:, best],
        n_trials=200,
        all_trial_sharpes=trial_sharpes,
    )
    assert dsr["deflated_sharpe_ratio"] == pytest.approx(0.6309, abs=5e-4)
    assert dsr["deflated_benchmark_sr0"] == pytest.approx(1.486 / np.sqrt(252), abs=5e-4)
    assert dsr["psr_vs_zero"] == pytest.approx(0.9995, abs=5e-4)
    assert dsr["observed_sharpe"] * np.sqrt(252) == pytest.approx(1.654, abs=2e-3)

    haircut = haircut_sharpe(dsr["observed_sharpe"], dsr["n_obs"], 200, method="bonferroni")
    assert haircut["adjusted_sharpe"] * np.sqrt(252) == pytest.approx(0.6477, abs=5e-3)
    mintrl = minimum_track_record_length(
        dsr["observed_sharpe"],
        0.0,
        dsr["skew"],
        dsr["kurtosis"],
    )
    assert mintrl == pytest.approx(250.8, abs=0.6)


def test_pbo_noise_about_half_and_edge_low() -> None:
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 1, size=(500, 30))
    pbo_noise = probability_of_backtest_overfitting(noise, n_blocks=10)["pbo"]
    edge = noise.copy()
    edge[:, 0] += 0.15
    pbo_edge = probability_of_backtest_overfitting(edge, n_blocks=10)["pbo"]
    assert 0.2 < pbo_noise < 0.8
    assert pbo_edge < 0.4
    assert pbo_edge < pbo_noise


def test_report_fails_on_negative_edge() -> None:
    rng = np.random.default_rng(5)
    returns = rng.normal(-0.0005, 0.01, 300)
    report = build_overfit_report(returns, n_trials=20)
    assert report["passed"] is False
    assert report["verdict"].startswith("FAIL")


def test_check_artifact_schema_and_boundaries() -> None:
    rng = np.random.default_rng(6)
    returns = rng.normal(0.0004, 0.012, 200)
    artifact = build_check_artifact(
        name="demo",
        source="test",
        selected_returns=returns,
        n_trials=10,
    )
    assert artifact["schema_version"] == "factor_qc.check.v1"
    assert artifact["boundaries"]["production_effect"] is False
    assert artifact["boundaries"]["allow_real_trade"] is False
    assert artifact["report"]["pbo"] is None
    assert artifact["limitations"]


def test_report_with_trials_computes_pbo() -> None:
    rng = np.random.default_rng(7)
    noise = rng.normal(0, 0.01, size=(120, 10))
    report = build_overfit_report(
        noise[:, 0],
        n_trials=10,
        trials_matrix=noise,
        n_blocks=8,
    )
    assert report["pbo"] is not None
    assert report["pbo"]["n_strategies"] == 10
    assert report["pbo"]["n_splits"] > 0
