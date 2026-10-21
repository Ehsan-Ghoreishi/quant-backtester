"""Equity-curve charts."""

import matplotlib

matplotlib.use("Agg")  # charts are only saved to disk; works without a display

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def plot_equity(bt: pd.DataFrame, label: str, title: str, path: str) -> None:
    """Save strategy vs buy-and-hold equity curves for one backtest."""
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(bt.index, bt["strat_equity"], label=label)
    ax.plot(bt.index, bt["market_equity"], label="Buy & Hold", alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
