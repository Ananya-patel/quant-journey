"""
backtester.py
─────────────
Takes a signal column and computes full strategy performance.
This is the engine that every quant project relies on.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def run_backtest(df: pd.DataFrame,
                 signal_col: str = 'signal',
                 return_col: str = 'Return',
                 initial_capital: float = 100_000,
                 cost_per_trade: float = 0.0005) -> dict:
    """
    cost_per_trade: 0.05% per trade (one-way)
    Charged whenever signal changes (a new trade happens)
    """
    df = df.copy()

    # Detect trades (signal changes)
    df['trade']       = df[signal_col].diff().abs() > 0
    df['trade_cost']  = df['trade'] * cost_per_trade

    # Strategy return after costs
    df['strat_return'] = (df[signal_col] * df[return_col]) - df['trade_cost']

    # Equity curves
    df['equity']    = initial_capital * (1 + df['strat_return']).cumprod()
    df['bh_equity'] = initial_capital * (1 + df[return_col]).cumprod()

    # ── METRICS ───────────────────────────────────────────────
    n_days       = len(df)
    total_return = df['equity'].iloc[-1] / initial_capital - 1
    ann_return   = (1 + total_return) ** (252 / n_days) - 1
    ann_vol      = df['strat_return'].std() * np.sqrt(252)
    sharpe       = ann_return / ann_vol if ann_vol != 0 else 0

    rolling_max  = df['equity'].cummax()
    drawdown     = (df['equity'] - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    calmar       = ann_return / abs(max_drawdown) if max_drawdown != 0 else 0

    wins         = (df['strat_return'] > 0).sum()
    total_trades = df['trade'].sum()
    win_rate     = wins / total_trades if total_trades > 0 else 0

    metrics = {
        'Total Return':    f"{total_return*100:.2f}%",
        'Ann. Return':     f"{ann_return*100:.2f}%",
        'Ann. Volatility': f"{ann_vol*100:.2f}%",
        'Sharpe Ratio':    f"{sharpe:.3f}",
        'Max Drawdown':    f"{max_drawdown*100:.2f}%",
        'Calmar Ratio':    f"{calmar:.3f}",
        'Win Rate':        f"{win_rate*100:.1f}%",
        'Total Trades':    int(total_trades),
        'Trading Days':    n_days,
    }

    return metrics, df


def print_metrics(metrics: dict, strategy_name: str = "OBI Strategy") -> None:
    """Print metrics in a clean formatted table."""
    print(f"\n{'═'*40}")
    print(f"  {strategy_name}")
    print(f"{'═'*40}")
    for key, val in metrics.items():
        print(f"  {key:<20} {val}")
    print(f"{'═'*40}\n")


def plot_results(df: pd.DataFrame,
                 metrics: dict,
                 ticker: str = "AAPL") -> None:
    """Generate the final results chart."""
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 14))
    fig.patch.set_facecolor('#0d0d14')
    fig.suptitle(f'P01 — OBI Signal Backtest: {ticker}',
                 color='white', fontsize=14, y=0.98)
    
    for ax in axes:
        ax.set_facecolor('#0d0d14')
        ax.tick_params(colors='#888')
        ax.title.set_color('white')
        ax.yaxis.label.set_color('#888')
        for spine in ax.spines.values():
            spine.set_edgecolor('#1e1e30')
    
    # Chart 1: Equity Curve
    axes[0].plot(df.index, df['equity'],    color='#00ff9d', lw=1.8,
                 label='OBI Strategy')
    axes[0].plot(df.index, df['bh_equity'], color='#7b61ff', lw=1.2,
                 linestyle='--', label='Buy & Hold', alpha=0.8)
    axes[0].set_title('Equity Curve')
    axes[0].set_ylabel('Portfolio ($)')
    axes[0].legend(facecolor='#1e1e30', labelcolor='white')
    axes[0].grid(True, alpha=0.1)
    
    # Chart 2: Drawdown
    rolling_max = df['equity'].cummax()
    drawdown = (df['equity'] - rolling_max) / rolling_max * 100
    axes[1].fill_between(df.index, drawdown, 0,
                          color='#ff3c6e', alpha=0.6)
    axes[1].plot(df.index, drawdown, color='#ff3c6e', lw=0.8)
    axes[1].set_title('Drawdown (%)')
    axes[1].set_ylabel('%')
    axes[1].grid(True, alpha=0.1)
    
    # Chart 3: OBI Signal
    axes[2].plot(df.index, df['obi_smooth'], color='#ffcc00', lw=1.2)
    axes[2].axhline(0,    color='white',   lw=0.5)
    axes[2].axhline( 0.1, color='#00ff9d', lw=0.6, linestyle='--')
    axes[2].axhline(-0.1, color='#ff3c6e', lw=0.6, linestyle='--')
    axes[2].set_title('OBI Signal (smoothed)')
    axes[2].set_ylabel('OBI')
    axes[2].grid(True, alpha=0.1)
    
    # Chart 4: Daily Strategy Returns
    colors = ['#00ff9d' if r > 0 else '#ff3c6e'
              for r in df['strat_return']]
    axes[3].bar(df.index, df['strat_return']*100,
                color=colors, width=1.0, alpha=0.8)
    axes[3].set_title('Daily Strategy Returns (%)')
    axes[3].set_ylabel('%')
    axes[3].grid(True, alpha=0.1)
    
    plt.tight_layout()
    
    results_dir = os.path.join(os.path.dirname(__file__),
                               '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, 'p01_backtest_results.png')
    plt.savefig(path, dpi=150, bbox_inches='tight',
                facecolor='#0d0d14')
    plt.show()
    print(f"[backtester] Chart saved → {path}")