# factor-qc

## 中文说明

`factor-qc` 是一个面向量化回测的质量闸门，也适用于 A 股因子研究。
它把 DSR、PBO、Harvey-Liu 多重检验折减和 MinTRL 放在同一次检查中，
并要求明确填写尝试过的参数组合数量。数据不完整或指标未达到门槛时，
工具会给出分级结果并默认不放行；它不保证策略盈利，也不会替研究者选择因子。

A **fail-closed quality gate** for backtests: one numpy-only engine covering
Deflated Sharpe Ratio, Probability of Backtest Overfitting (CSCV), the
Harvey-Liu multiple-testing haircut and Minimum Track Record Length —graded P0/P1/P2, and **it refuses to judge a backtest that will not declare
how many configurations were tried**. Python 3.11+, one dependency
(`numpy`), Windows / Linux / macOS.

**Status:** v0.1 —alpha. The statistics are battle-tested inside a
production research pipeline and validated against published reference
values, but this standalone package is new: expect the CLI to shift before
v1.0.

## 中文简介

`factor-qc` 是一个面向量化回测的质量闸门，也适用于 A 股因子研究。
它把 DSR、PBO、Harvey-Liu 多重检验折减和 MinTRL 放在同一次检查中，
并要求明确填写尝试过的参数组合数量。数据不完整或指标未达到门槛时，
工具会给出分级结果并默认不放行；它不保证策略盈利，也不会替研究者选择因子。

## Why this exists

The standard story: you try 200 factor configurations, the best one shows a
Sharpe of 1.65, you feel great. The honest story: with 200 trials of pure
noise, *someone* is going to show a Sharpe of 1.65 —the expected maximum of
200 zero-true-SR trials —and it will not be your skill, it will be your
selection bias.

Most backtest tooling computes statistics and prints reports. `factor-qc`
is a **gate**: it decides, with graded severity, whether a candidate may
pass —and its default answer is *no*:

- **P0 (fatal)** —DSR below threshold, PBO above threshold, haircut Sharpe
  below floor, MinTRL longer than the sample -> the candidate must not pass.
- **P1 (warning)** —weak PSR vs zero, short sample, aggressive trial count
  -> proceed with eyes open.
- **P2 (info)** —non-normal moments, tiny trial count -> recorded, no action.

## Philosophy

**Honesty is the default; the gate is fail-closed.**

The one non-negotiable input is `n_trials`: the honest count of
configurations you tried. Without it there is no deflation benchmark
([Bailey & López de Prado 2014](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551)),
no haircut ([Harvey, Liu & Zhu 2016, RFS](https://doi.org/10.1093/rfs/hhv059)),
no track-record floor
([Bailey & López de Prado 2018, JPM](https://ideas.repec.org/a/rsk/journl/0journalpm-v44n5.html))
and no overfitting probability
([Bailey, Borwein, López de Prado & Zhu 2017, JCF](https://escholarship.org/uc/item/4w1110bb)).
Refuse to declare, and the gate refuses to judge —that asymmetry is the
point. `qc check` exits non-zero on any P0 failure, so it drops into CI,
pre-commit hooks and research gates as a hard blocker, not a suggestion.

Two design commitments that keep it honest:

1. **numpy only, no scipy** —the standard-normal inverse CDF is Acklam's
   rational approximation with one Newton refinement; every number in the
   report is reproducible from the code in this repo, no hidden black box.
2. **PBO is optional but explicit** —without a trials matrix the gate says
   PBO *was not computed*, and the DSR cross-trial variance degrades to a
   conservative single-trial estimate. Absence of evidence is reported as
   absence, never as evidence.

## Quick start

```bash
# install from PyPI (once published)
pip install factor-qc

# or run without installing anything:
#   PYTHONPATH=src python -m factor_qc --help

python examples/demo.py   # try it on reproducible synthetic cases
```

Your own backtest:

```bash
# returns.json = JSON list of per-period returns of the selected candidate
# trials.json   = JSON 2D matrix (T x N) of every configuration you tried

qc check --returns returns.json --trials trials.json --n-trials 200
# -> FAIL - P0 blocker(s): dsr: 0.63 vs 0.95; mintrl: 250.8 vs <= 1000; ...

qc check --returns returns.json --n-trials 5 --json   # machine-readable
```

Exit codes: `0` = no P0 failures (P1/P2 may still be failing), `1` = at
least one P0 failure (or missing `n_trials`), `2` = usage error. Wire it
into CI as a hard gate.

## Commands

| Command | What it does |
| --- | --- |
| `check` | Run the gate: DSR, PBO (when `--trials` given), haircut Sharpe, MinTRL as P0; PSR-vs-zero, sample length, trial aggression as P1; moments and trial count as P2. Human-readable or `--json` output |
| `version` | Print version |

Flags: `--returns` (required), `--trials` (optional), `--n-trials`
(required unless `require_declared_trials` is disabled in code),
`--periods-per-year` (default 252), `--n-blocks` (CSCV granularity, default
16).

## The checks

| Check | Severity | Method | Reference |
| --- | --- | --- | --- |
| `dsr` | P0 | Deflated Sharpe Ratio: P(SR > E[max SR of N trials]) under non-normal moments | [Bailey & López de Prado (2014), JPM 40(5)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551) |
| `pbo` | P0 | Probability of Backtest Overfitting via Combinatorially-Symmetric Cross-Validation (12,870 splits at n_blocks=16) | [Bailey, Borwein, López de Prado & Zhu (2017), JCF](https://escholarship.org/uc/item/4w1110bb) |
| `haircut_sharpe` | P0 | Multiple-testing haircut of the Sharpe ratio (Bonferroni/Holm/BHY) | [Harvey & Liu (2015)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2528780) |
| `mintrl` | P0 | Minimum Track Record Length: observations needed before SR is significant | [Bailey & López de Prado (2018), JPM 44(5)](https://ideas.repec.org/a/rsk/journl/0journalpm-v44n5.html) |
| `psr_vs_zero` | P1 | Probabilistic Sharpe vs zero | [Bailey & López de Prado (2012)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2168747) |
| `sample_length` | P1 | ≥252 observations | —|
| `trial_aggression` | P1 | n_trials ≥n_obs / 5 | —|
| `return_moments` | P2 | skew ≥0, kurtosis ≥3 | —|
| `trial_count` | P2 | n_trials ≥5 | —|

The P0 set mirrors the spirit of [Harvey, Liu & Zhu (2016),
"“And the Cross-Section of Expected Returns"](https://doi.org/10.1093/rfs/hhv059):
a factor must survive multiple-testing correction to earn the right to be
called a factor. The gate is the machine version of that editorial stance.

## Performance note

CSCV enumerates C(n_blocks, n_blocks/2) splits —12,870 at the default 16.
On large trial matrices (T=1000, N=200) that takes minutes; use
`--n-blocks 8` (70 splits) or `10` (252 splits) for interactive speed at
slightly coarser granularity.

## Development

```bash
python -m pip install -e . pytest
python -m pytest
```

CI runs the full test suite on Ubuntu, Windows and macOS with Python 3.11
and 3.12. Issues are handled on weekends; pull requests are welcome.

## Related work

- [Bailey & López de Prado (2014), The Deflated Sharpe Ratio](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551)
- [Bailey, Borwein, López de Prado & Zhu (2017), The Probability of Backtest Overfitting](https://escholarship.org/uc/item/4w1110bb)
- [Harvey, Liu & Zhu (2016), “And the Cross-Section of Expected Returns (RFS)](https://doi.org/10.1093/rfs/hhv059)
- [Harvey & Liu (2021), Lucky Factors (JFE)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2528780)
- [Mobarekeh & López de Prado (2024), Backtest Overfitting in the Machine Learning Era (SSRN 4778909)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4778909) —why OOS methods still need honest trial accounting

## Project family

Part of [Foolproof Labs](https://github.com/foolproof-labs) — a toolchain
against self-deception in quantitative research:

- [pit-adjuster](https://github.com/foolproof-labs/pit-adjuster) — PIT back-adjustment with static forward-adjustment drift detection
- [falsification-ledger](https://github.com/foolproof-labs/falsification-ledger) — pre-registration and falsification ledger
- [factor-qc](https://github.com/foolproof-labs/factor-qc) — fail-closed backtest quality gate
- [lesson-book](https://github.com/foolproof-labs/lesson-book) — tuition memory for traders
- [lookahead-free](https://github.com/foolproof-labs/lookahead-free) — verifiable look-ahead-freedom checks
- [ashare-data-immunity](https://github.com/foolproof-labs/ashare-data-immunity) — data immunity for A-share daily bars

## License

MIT
