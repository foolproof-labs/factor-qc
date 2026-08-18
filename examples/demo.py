"""End-to-end demo: fail-closed gate on reproducible synthetic cases.

Run with:  python examples/demo.py
No network. Three scenarios:

1. honest edge      — a mildly positive strategy with few trials -> PASS
2. selection bias   — best of 200 pure-noise trials -> FAIL (DSR, MinTRL)
3. refusal          — no n_trials declaration -> FAIL (fail-closed)
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from factor_qc.cli import main as cli_main  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="qc-demo-"))


def _write(name: str, value) -> Path:
    path = TMP / name
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _run(label: str, argv: list[str]) -> None:
    print(f"\n== {label} ==")
    code = cli_main(argv)
    print(f"=> exit code: {code} (0 = gate open, 1 = gate closed)")


def run_demo() -> int:
    rng = np.random.default_rng(3)

    # 1. honest edge: 800 obs, small positive drift, 5 trials declared.
    honest = rng.normal(0.0012, 0.008, 800).tolist()
    honest_path = _write("honest.json", honest)
    _run("1. honest edge (n_trials=5)", ["check", "--returns", str(honest_path), "--n-trials", "5"])

    # 2. selection bias: best of 200 noise trials, with the trials matrix.
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.01, size=(1000, 200))
    best = int(np.argmax(noise.mean(0) / noise.std(0, ddof=1)))
    best_path = _write("best.json", noise[:, best].tolist())
    trials_path = _write("trials.json", noise.tolist())
    _run(
        "2. best of 200 noise trials (the honest question)",
        [
            "check",
            "--returns", str(best_path),
            "--trials", str(trials_path),
            "--n-trials", "200",
            "--n-blocks", "8",
        ],
    )

    # 3. refusal: no n_trials declaration.
    _run("3. no n_trials declared (fail-closed)", ["check", "--returns", str(honest_path)])

    print(f"\nfixtures under: {TMP}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_demo())
