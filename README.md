[README.md](https://github.com/user-attachments/files/31891332/README.md)
# KSE Backtest Platform

A live, multi-asset quantitative strategy backtesting terminal built in a single Streamlit/Python file — no database, no backend server, just live market data and a real cost/risk engine underneath it.

**[Live app →](https://kse-backtest-platform.streamlit.app)**

---

## What it does

Pick a stock or build a portfolio, choose a trading strategy, and see what would have actually happened if you'd traded it — with realistic costs, honest statistics, and validation methods designed to catch overfitting rather than hide it.

## Features

**Strategies**
- Moving-average crossover (trend-following), long or long/short
- Mean reversion (Bollinger/z-score band), long or long/short
- Portfolio: Buy & Hold, Equal Weight, Moving Average, absolute Momentum, and Cross-Sectional Momentum (relative-strength ranking across the universe)

**Execution realism**
- Commission and slippage modeled as separate cost buckets, not lumped together
- Square-root market impact model using live trading volume (cost scales with order size relative to a ticker's liquidity)
- Borrow cost accrual for short positions
- Volatility-targeted position sizing (leverage scales to a target annualized vol, computed with no lookahead)
- Idle cash earns the risk-free rate you set, consistently reflected in Sharpe/Sortino

**Validation & statistics**
- Walk-forward validation (multi-fold, re-optimized per fold, with a fixed warm-up-window bug that a lot of backtesters get wrong)
- Deflated Sharpe Ratio (Bailey & López de Prado, 2014) — corrects for the multiple-testing bias of scanning many parameter combinations and reporting the best one
- Autocorrelation-adjusted Sharpe Ratio (Lo, 2002) — corrects for the fact that trend-following returns aren't independent day to day
- Newey-West (HAC) standard errors on alpha significance, not naive OLS
- Block-bootstrap confidence intervals on Sharpe and CAGR
- Beta/alpha decomposition vs. a selectable benchmark
- Monte Carlo simulation (10,000 trade-resampled paths) with probability-of-profit, probability of beating benchmark, and drawdown probabilities

**Market terminal UI**
- Live, animated ticker tape across 100+ global equities (US, Netherlands, Germany, France, UK)
- Global search (ticker, company name, or sector)
- Clickable everything: tape → stock detail, sector performance → filtered stock list, movers → stock detail
- Stock detail pages with an interactive Plotly range-selector chart (1M/3M/6M/YTD/1Y/5Y/All)
- Session watchlist and one-click "add to portfolio" from anywhere in the app
- Market breadth, sector performance, region performance, top gainers/losers, most volatile

## Data

Live daily closes and volume pulled from Yahoo Finance via `yfinance` — refreshed automatically every 5 minutes, or on demand. There is deliberately **no static CSV fallback**: if the data provider is unreachable, the app shows an explicit error rather than silently serving stale numbers.

## Tech stack

Python · Streamlit · pandas · NumPy · Plotly · yfinance

## Running locally

```bash
git clone https://github.com/KhalidelShrbini/kse-backtest-platform.git
cd kse-backtest-platform
pip install -r requirements.txt
streamlit run app.py
```

## Known limitations

Being upfront about what this is and isn't:

- **Survivorship bias** — the universe is 100+ large, currently-successful companies chosen in hindsight, not a realistic point-in-time universe.
- **No point-in-time data store** — Yahoo Finance's adjusted-close series can be retroactively revised (new dividends/splits recompute historical adjustment factors), so this isn't a system of record for historical prices.
- **Approximate market impact model** — a square-root heuristic, not a full limit-order-book simulation.
- **Single-factor alpha only** — no decomposition against size, value, or other style factors (would need a Fama-French-style dataset this app doesn't have).
- **Ticker-level multiple-testing is uncorrected** — the Deflated Sharpe Ratio corrects for scanning parameter combinations on one ticker, not for scanning the whole universe and picking whichever ticker looked best.

## License

MIT
