"""Tests for the fail-closed gate semantics."""

from __future__ import annotations

import numpy as np
import pytest

from factor_qc.gate import run_gate


def test_gate_refuses_without_n_trials_declaration() -> None:
    rng = np.random.default_rng(0)
    returns = rng.normal(0.001, 0.01, 500)
    body = run_gate(returns, None)
    assert body["passed"] is False
    blockers = [check for check in body["checks"] if check["severity"] == "P0" and not check["passed"]]
    assert any(check["check_id"] == "n_trials_declaration_required" for check in blockers)
    assert body["report"] is None


def test_gate_allows_explicit_bypass_of_declaration() -> None:
    rng = np.random.default_rng(0)
    returns = rng.normal(0.001, 0.01, 500)
    body = run_gate(returns, None, require_declared_trials=False)
    assert body["report"] is not None


def test_gate_flags_overfit_noise() -> None:
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.01, size=(1000, 200))
    best = int(np.argmax(noise.mean(0) / noise.std(0, ddof=1)))
    body = run_gate(
        noise[:, best], 200, trials_matrix=noise, periods_per_year=252, n_blocks=8
    )
    assert body["passed"] is False
    p0 = [check for check in body["checks"] if check["severity"] == "P0"]
    assert all(not check["passed"] for check in p0 if check["check_id"] == "dsr")
    assert body["verdict"].startswith("FAIL")


def test_gate_passes_on_realistic_edge() -> None:
    rng = np.random.default_rng(3)
    returns = rng.normal(0.0012, 0.008, 800)
    body = run_gate(returns, 5, periods_per_year=252)
    assert body["passed"] is True
    assert body["verdict"].startswith("PASS")


def test_gate_severity_grading() -> None:
    rng = np.random.default_rng(4)
    returns = rng.normal(0.0004, 0.012, 200)  # short sample, mild edge
    body = run_gate(returns, 3, periods_per_year=252)
    severities = {check["severity"] for check in body["checks"]}
    assert severities == {"P0", "P1", "P2"}
    p1_failed = [check for check in body["checks"] if check["severity"] == "P1" and not check["passed"]]
    assert any(check["check_id"] == "sample_length" for check in p1_failed)
    assert any(check["check_id"] == "trial_count" for check in body["checks"] if check["severity"] == "P2")
