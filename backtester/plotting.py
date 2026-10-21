"""Equity-curve charts."""

import matplotlib

matplotlib.use("Agg")  # charts are only saved to disk; works without a display

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402


def drawdown(equity: pd.Series) -> pd.Series:
    """Fall from the running peak, as a fraction (0 = at a new high)."""
    return equity / equity.cummax() - 1


def plot_equity(bt: pd.DataFrame, label: str, title: str, path: str) -> None:
    """Save strategy vs buy-and-hold equity curves with a drawdown panel below."""
    fig, (ax, ax_dd) = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})
    ax.plot(bt.index, bt["strat_equity"], label=label)
    ax.plot(bt.index, bt["market_equity"], label="Buy & Hold", alpha=0.7)
    ax.set_title(title)
    ax.set_ylabel("Growth of $1")
    ax.legend()
    ax.grid(alpha=0.3)

    ax_dd.fill_between(bt.index, drawdown(bt["strat_equity"]), 0, alpha=0.5, label=label)
    ax_dd.plot(bt.index, drawdown(bt["market_equity"]), color="C1", alpha=0.7,
               label="Buy & Hold")
    ax_dd.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax_dd.set_ylabel("Drawdown")
    ax_dd.set_xlabel("Date")
    ax_dd.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
