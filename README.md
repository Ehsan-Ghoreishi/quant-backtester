# Quantitative Trading Strategy Backtester

A from-scratch backtesting framework in Python for evaluating systematic
trading strategies on real market data. Built to apply data-science
rigor to trading: **the goal is not to find a profitable strategy, but
to test ideas honestly and avoid fooling yourself.**

This repository implements two classic strategies — **mean-reversion**
and **momentum (trend-following)** — and evaluates them against a
buy-and-hold benchmark using proper out-of-sample testing.

---

## Why this project

Most retail backtests look impressive and are quietly broken: they peek
at future data, ignore trading costs, and never compare against simply
holding the asset. This project deliberately does the opposite and
treats a trading strategy like any other ML model — train/test split,
honest metrics, and a benchmark.

## Methodology (the part that matters)

Every strategy is evaluated under the same disciplined rules:

- **No look-ahead bias** — signals are shifted by one bar, so a decision
  made today is only acted on tomorrow.
- **Transaction costs** — a configurable cost (in basis points) is
  charged on every position change.
- **Train / test split** — strategies are judged on out-of-sample data
  the logic has never seen. In-sample numbers are treated as unreliable.
- **Benchmark comparison** — every result is compared to buy-and-hold.
  A strategy that cannot beat holding the asset has no value.
- **Standard metrics** — Total Return, Annualized Return, Annualized
  Volatility, Sharpe Ratio, and Maximum Drawdown.

## Strategies

| Strategy | Logic | Works best in |
|---|---|---|
| Mean-Reversion | Buy when price is unusually far below its average (z-score) | Range-bound / choppy markets |
| Momentum | Buy when a fast moving average is above a slow one | Trending markets |

## Results (Dow Jones, ^DJI, out-of-sample)

Both strategies were tested on 8 years of daily Dow Jones data
(2018-10-22 to 2026-10-21). The first 70% is the training period; the
numbers below are for the out-of-sample test period
(2024-05-28 to 2026-10-21), with 2 bps transaction costs:

| Metric | Mean-Reversion | Momentum | Buy & Hold |
|---|---|---|---|
| Total Return | -13.4% | +20.7% | +35.9% |
| Annualized Return | -5.8% | +8.2% | +13.7% |
| Annualized Volatility | 13.2% | 13.7% | 14.5% |
| Sharpe | -0.39 | +0.64 | +0.96 |
| Max Drawdown | -20.5% | -16.4% | -16.4% |

Walk-forward check (`--walk-forward 5`, five consecutive ~16-month
periods): momentum beat buy & hold on Sharpe in 1 of 5 periods, and so
did mean-reversion.

Re-tuning inside each walk-forward training window (`--walk-forward 5
--optimize`: every period uses the parameter set with the best Sharpe on
all earlier data, from a small grid) gives a mixed picture. For momentum
the average Sharpe over the five periods stays the same (0.29 with
re-tuning and with the default 50/200). For mean-reversion it rises from
-0.34 to +0.45, but almost all of that gain comes from a single period
(2021-22, Sharpe 1.80), so it is weak evidence, not a discovery.

Multi-asset check (`python compare_assets.py`, out-of-sample Sharpe,
same parameters everywhere):

| Asset | Mean-Reversion | Momentum | Buy & Hold |
|---|---|---|---|
| ^DJI (Dow Jones) | -0.39 | 0.64 | 0.96 |
| SPY (S&P 500) | 0.22 | 0.75 | 1.14 |
| QQQ (Nasdaq 100) | -0.01 | 0.75 | 1.02 |
| IWM (Russell 2000) | -0.13 | 0.72 | 0.76 |
| GLD (Gold) | -0.31 | 1.13 | 1.12 |

**Key findings:**

- Momentum outperformed mean-reversion on the trending Dow, confirming
  the regime hypothesis: a trend-following rule fits a trending market,
  while mean-reversion is structurally the wrong tool for it.
- Neither simple strategy beat buy-and-hold — a realistic, well-documented
  outcome. Beating the market is hard, and most simple rules don't.
- The pattern holds across assets: momentum has a positive Sharpe on
  all five, mean-reversion on only one, and only momentum on gold
  roughly ties buy & hold (1.13 vs 1.12).
- Mean-reversion lost money in 4 of 5 walk-forward periods, so its poor
  result is not an accident of where the train/test cut falls.
- Fixing a warm-up bug (see below) raised momentum's out-of-sample
  return from +13.0% to +20.7%: the earlier version had been forced to
  sit flat for the first 200 days of the test period.

Documenting strategies that *don't* beat the benchmark, and explaining
*why*, is the point: this is a research process, not a sales pitch.

### Bugs found while reviewing the backtest

Honest backtests need their own code reviewed. Issues fixed so far:

- **Indicator warm-up in the test period.** Signals were recomputed on
  the test slice alone, so a 200-day moving average had no value for the
  first 200 test days. Signals are now computed on the full history
  (still leak-free: rolling windows only look backwards) and then split.
- **Non-standard Sharpe ratio.** It divided the geometric annual return
  by volatility; it now uses the standard mean/std of daily (excess)
  returns × √252.
- **Silent synthetic data.** A failed (often rate-limited) download used
  to swap in random data with one easily-missed line of output. It now
  warns loudly, labels the chart, and can be turned off with
  `--no-fallback`.
- **Free first trade.** A period that started already invested was not
  charged the entry cost.

## Project structure

```
backtester/
  data.py        price loading (yfinance + daily cache, synthetic fallback)
  strategies.py  mean-reversion and momentum signal generators
  engine.py      vectorized backtest, train/test split, walk-forward
  metrics.py     return, volatility, Sharpe, max drawdown
  plotting.py    equity-curve charts
  cli.py         command-line options shared by both scripts
tests/           pytest suite (incl. an automated look-ahead-bias test)
mean_reversion_backtest.py
momentum_backtest.py
compare_assets.py   runs both strategies on several assets
```

## Tech stack

Python, pandas, NumPy, Matplotlib, yfinance; pytest and ruff for
testing and linting, GitHub Actions for CI.

## How to run

```bash
pip install -r requirements.txt
python mean_reversion_backtest.py
python momentum_backtest.py
```

By default both scripts pull real daily data via `yfinance` and cache it
in `.cache/` for the day. If the download fails, they fall back to
synthetic data with a clear warning so the code always runs; pass
`--no-fallback` to fail instead.

Every setting is a command-line option (`--help` lists them all):

```bash
python momentum_backtest.py --ticker SPY --years 15 --fast 20 --slow 100
python mean_reversion_backtest.py --ticker QQQ --entry-z 1.5 --cost-bps 5
python momentum_backtest.py --walk-forward 5
python mean_reversion_backtest.py --walk-forward 5 --optimize
python compare_assets.py --tickers SPY QQQ GLD
```

## Development

```bash
pip install -r requirements-dev.txt
pytest
ruff check .
```

## What I learned

- How to build a leak-free backtest and why look-ahead bias silently
  ruins most amateur results.
- Why a strategy must be matched to the market regime (mean-reversion vs
  trend).
- How to read risk-adjusted performance (Sharpe, drawdown) rather than
  raw return.
- That rigorous evaluation — not clever ideas — is what separates
  systematic trading from gambling.

## Roadmap

- [ ] Replace the hand-coded z-score rule with an ML classifier
      (feature engineering from price/volatility, proper CV).
- [x] Multi-asset testing to check robustness.
- [x] Walk-forward validation across several periods.
- [x] Parameter optimisation inside each walk-forward training window.
- [ ] Volatility-targeted position sizing instead of all-in / all-out.

## Disclaimer

This is an educational and portfolio project. It is **not** financial
advice and nothing here is a recommendation to trade.
