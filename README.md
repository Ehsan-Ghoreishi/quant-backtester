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

Both strategies were tested on ~8 years of real Dow Jones data
(out-of-sample / test period):

| Metric | Mean-Reversion | Momentum | Buy & Hold |
|---|---|---|---|
| Total Return | -16.4% | +6.8% | +36.9% |
| Annualized Return | -7.2% | +2.8% | +14.0% |
| Sharpe | -0.56 | +0.23 | +0.98 |
| Max Drawdown | -20.1% | -16.4% | -16.4% |

**Key findings:**

- Momentum outperformed mean-reversion on the trending Dow, confirming
  the regime hypothesis: a trend-following rule fits a trending market,
  while mean-reversion is structurally the wrong tool for it.
- Neither simple strategy beat buy-and-hold — a realistic, well-documented
  outcome. Beating the market is hard, and most simple rules don't.
- Momentum matched the market's drawdown (-16.4%) while stepping aside
  during downtrends, showing comparable risk with a different return
  profile.

Documenting strategies that *don't* beat the benchmark, and explaining
*why*, is the point: this is a research process, not a sales pitch.

## Tech stack

Python, pandas, NumPy, Matplotlib, yfinance.

## How to run

```bash
pip install -r requirements.txt
python mean_reversion_backtest.py
python momentum_backtest.py
```

By default both scripts pull real daily data via `yfinance`. If the
network or library is unavailable, they fall back to synthetic data so
the code always runs.

To test other assets, change the ticker in `main()`:

```python
prices = load_prices("DIA", years=8)   # Dow ETF, S&P: "SPY", Nasdaq: "QQQ"
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
- [ ] Multi-asset testing to check robustness.
- [ ] Walk-forward (rolling) validation instead of a single split.

## Disclaimer

This is an educational and portfolio project. It is **not** financial
advice and nothing here is a recommendation to trade.
