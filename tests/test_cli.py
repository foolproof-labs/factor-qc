"""End-to-end CLI tests for the gate."""

from __future__ import annotations

import json

import numpy as np
import pytest

from factor_qc.cli import main


@pytest.fixture()
def returns_file(tmp_path):
    rng = np.random.default_rng(3)
    returns = rng.normal(0.0012, 0.008, 800).tolist()
    path = tmp_path / "returns.json"
    path.write_text(json.dumps(returns), encoding="utf-8")
    return str(path)


@pytest.fixture()
def trials_file(tmp_path):
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.01, size=(400, 12)).tolist()
    path = tmp_path / "trials.json"
    path.write_text(json.dumps(noise), encoding="utf-8")
    return str(path)


def test_cli_version() -> None:
    assert main(["version"]) == 0


def test_cli_check_requires_n_trials(returns_file, capsys) -> None:
    assert main(["check", "--returns", returns_file]) == 1
    out = capsys.readouterr().out
    assert "n_trials" in out


def test_cli_check_passes_with_declaration(returns_file, capsys) -> None:
    assert main(["check", "--returns", returns_file, "--n-trials", "5"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("PASS")


def test_cli_check_with_trials_fails_on_noise(trials_file, tmp_path, capsys) -> None:
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.01, size=(400, 12))
    best = int(np.argmax(noise.mean(0) / noise.std(0, ddof=1)))
    returns_path = tmp_path / "best.json"
    returns_path.write_text(json.dumps(noise[:, best].tolist()), encoding="utf-8")
    assert (
        main(
            [
                "check",
                "--returns", str(returns_path),
                "--trials", trials_file,
                "--n-trials", "12",
                "--n-blocks", "8",
            ]
        )
        == 1
    )
    out = capsys.readouterr().out
    assert "FAIL" in out


def test_cli_check_json_output(returns_file, capsys) -> None:
    assert main(["check", "--returns", returns_file, "--n-trials", "5", "--json"]) == 0
    body = json.loads(capsys.readouterr().out)
    assert body["passed"] is True
    assert len(body["checks"]) == 8
    assert body["report"]["n_obs"] == 800
