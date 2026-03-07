from src.data_loader import load_price_data, save_data
from src.signal      import compute_obi_proxy, compute_signal
from src.backtester  import run_backtest, print_metrics, plot_results

TICKER          = "AAPL"
PERIOD          = "2y"
OBI_SMOOTH      = 20       # ← increased from 5 to 20 (less noise)
OBI_THRESHOLD   = 0.15     # ← slightly higher bar to trigger trade
TREND_WINDOW    = 50       # SMA50 trend filter
VOL_CAP         = 0.40     # go flat if annualized vol > 40%
INITIAL_CAPITAL = 100_000

if __name__ == "__main__":
    print("\n" + "═"*50)
    print("  P01 — OBI Signal Backtest v2 (with filters)")
    print("═"*50)
    print(f"  Ticker:         {TICKER}")
    print(f"  OBI smooth:     {OBI_SMOOTH} days")
    print(f"  Threshold:      {OBI_THRESHOLD}")
    print(f"  Trend window:   {TREND_WINDOW} days")
    print(f"  Vol cap:        {VOL_CAP*100:.0f}%")
    print(f"  Capital:        ${INITIAL_CAPITAL:,}")

    df = load_price_data(TICKER, PERIOD)
    save_data(df, TICKER)

    df = compute_obi_proxy(df, smooth_window=OBI_SMOOTH)
    df = compute_signal(df,
                        threshold=OBI_THRESHOLD,
                        trend_window=TREND_WINDOW,
                        vol_cap=VOL_CAP)

    metrics, df = run_backtest(df, initial_capital=INITIAL_CAPITAL)
    print_metrics(metrics, strategy_name=f"OBI v2 — {TICKER}")
    plot_results(df, metrics, ticker=TICKER)
    