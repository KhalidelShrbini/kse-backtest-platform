
"""
Backtest Platform (Streamlit)
=============================
Two tools in one app:
  1. Single-Stock Backtest  -- fast/slow MA crossover, parameter heatmap,
     trade log, trade statistics, Monte Carlo (trade-resampled)
  2. Portfolio Backtest     -- multi-asset weights, 5 strategies,
     rebalancing frequency, risk-free rate

Reads live daily closes from Yahoo Finance (via yfinance) on every session --
there is no bundled CSV and no static-data fallback. If the data provider is
unreachable the app shows an error and a retry button rather than silently
serving stale numbers.

RUN LOCALLY:
    pip install streamlit pandas numpy plotly yfinance
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statistics import NormalDist

st.set_page_config(page_title="Backtest Platform", layout="wide", page_icon="\U0001F4C8")

st.markdown("""
<style>
    .stApp { background-color: #0A0D12; }
    [data-testid="stMetric"] {
        background-color: #151A22;
        border: 1px solid #1D232C;
        border-radius: 6px;
        padding: 12px 16px;
    }
    [data-testid="stMetricLabel"] { color: #6B7684; font-size: 0.75rem; }
    [data-testid="stMetricValue"] { color: #E8ECF1; font-family: ui-monospace, monospace; }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #12161D;
        border-radius: 4px 4px 0 0;
        color: #6B7684;
    }
    .stTabs [aria-selected="true"] { color: #E8ECF1 !important; background-color: #171C24 !important; }
    h1, h2, h3 { color: #E8ECF1 !important; font-family: -apple-system, sans-serif; }
    .stCaption, [data-testid="stCaptionContainer"] { color: #6B7684 !important; }
</style>
""", unsafe_allow_html=True)

PLOTLY_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor="#12161D",
        plot_bgcolor="#12161D",
        font=dict(color="#C9D1D9", family="ui-monospace, monospace", size=11),
        xaxis=dict(gridcolor="#1D232C", zerolinecolor="#2A313C"),
        yaxis=dict(gridcolor="#1D232C", zerolinecolor="#2A313C"),
        legend=dict(bgcolor="#161B22", bordercolor="#232933", borderwidth=1),
        colorway=["#E8A33D", "#8891A0", "#4FAE7A", "#D9724F", "#5B8DD9"],
    )
)

# Curated universe: well-known large-caps across the US, Netherlands, Germany,
# France, and the UK.
# NOTE: these are TODAY's well-known names, not point-in-time historical index
# membership -- see the Methodology tab for the survivorship-bias caveat.
TICKER_INFO = {
    # --- US ---
    "AAPL": ("Apple", "Technology", "US"),
    "MSFT": ("Microsoft", "Technology", "US"),
    "GOOGL": ("Alphabet", "Technology", "US"),
    "AMZN": ("Amazon", "Consumer Discretionary", "US"),
    "NVDA": ("Nvidia", "Technology", "US"),
    "META": ("Meta Platforms", "Technology", "US"),
    "TSLA": ("Tesla", "Automotive/Tech", "US"),
    "JPM": ("JPMorgan Chase", "Financials", "US"),
    "V": ("Visa", "Financials", "US"),
    "JNJ": ("Johnson & Johnson", "Healthcare", "US"),
    "PG": ("Procter & Gamble", "Consumer Staples", "US"),
    "KO": ("Coca-Cola", "Consumer Staples", "US"),
    "WMT": ("Walmart", "Consumer Staples", "US"),
    "XOM": ("ExxonMobil", "Energy", "US"),
    "DIS": ("Disney", "Communication Services", "US"),
    "NFLX": ("Netflix", "Communication Services", "US"),
    "INTC": ("Intel", "Technology", "US"),
    "AMD": ("Advanced Micro Devices", "Technology", "US"),
    "PYPL": ("PayPal", "Financials", "US"),
    "BA": ("Boeing", "Industrials", "US"),
    "SPY": ("S&P 500 ETF", "Benchmark", "US"),
    # --- Euronext Amsterdam ---
    "ASML.AS": ("ASML Holding", "Technology", "NL"),
    "SHELL.AS": ("Shell", "Energy", "NL"),
    "UNA.AS": ("Unilever", "Consumer Staples", "NL"),
    "ADYEN.AS": ("Adyen", "Technology", "NL"),
    "HEIA.AS": ("Heineken", "Consumer Staples", "NL"),
    "PHIA.AS": ("Philips", "Healthcare", "NL"),
    "INGA.AS": ("ING Group", "Financials", "NL"),
    "AD.AS": ("Ahold Delhaize", "Consumer Staples", "NL"),
    "KPN.AS": ("KPN", "Communication Services", "NL"),
    "AKZA.AS": ("AkzoNobel", "Materials", "NL"),
    "WKL.AS": ("Wolters Kluwer", "Industrials", "NL"),
    "ASM.AS": ("ASM International", "Technology", "NL"),
    # --- US (added to expand universe from 33 to 50) ---
    "MA": ("Mastercard", "Financials", "US"),
    "UNH": ("UnitedHealth Group", "Healthcare", "US"),
    "HD": ("Home Depot", "Consumer Discretionary", "US"),
    "COST": ("Costco", "Consumer Staples", "US"),
    "CSCO": ("Cisco Systems", "Technology", "US"),
    "ORCL": ("Oracle", "Technology", "US"),
    "CRM": ("Salesforce", "Technology", "US"),
    "ADBE": ("Adobe", "Technology", "US"),
    "QCOM": ("Qualcomm", "Technology", "US"),
    "TXN": ("Texas Instruments", "Technology", "US"),
    "IBM": ("IBM", "Technology", "US"),
    "GE": ("General Electric", "Industrials", "US"),
    "CAT": ("Caterpillar", "Industrials", "US"),
    "NKE": ("Nike", "Consumer Discretionary", "US"),
    "MCD": ("McDonald's", "Consumer Discretionary", "US"),
    "PEP": ("PepsiCo", "Consumer Staples", "US"),
    "ABT": ("Abbott Laboratories", "Healthcare", "US"),
    # --- US (added to expand universe from 50 to 80) ---
    "PLD": ("Prologis", "Real Estate", "US"),
    "AMT": ("American Tower", "Real Estate", "US"),
    "O": ("Realty Income", "Real Estate", "US"),
    "NEE": ("NextEra Energy", "Utilities", "US"),
    "DUK": ("Duke Energy", "Utilities", "US"),
    "CVX": ("Chevron", "Energy", "US"),
    "COP": ("ConocoPhillips", "Energy", "US"),
    "T": ("AT&T", "Communication Services", "US"),
    "VZ": ("Verizon", "Communication Services", "US"),
    "CMCSA": ("Comcast", "Communication Services", "US"),
    "BAC": ("Bank of America", "Financials", "US"),
    "WFC": ("Wells Fargo", "Financials", "US"),
    "GS": ("Goldman Sachs", "Financials", "US"),
    "MS": ("Morgan Stanley", "Financials", "US"),
    "PFE": ("Pfizer", "Healthcare", "US"),
    "MRK": ("Merck", "Healthcare", "US"),
    "LLY": ("Eli Lilly", "Healthcare", "US"),
    "SBUX": ("Starbucks", "Consumer Discretionary", "US"),
    "LOW": ("Lowe's", "Consumer Discretionary", "US"),
    "CL": ("Colgate-Palmolive", "Consumer Staples", "US"),
    "LIN": ("Linde", "Materials", "US"),
    "HON": ("Honeywell", "Industrials", "US"),
    "UPS": ("United Parcel Service", "Industrials", "US"),
    "AVGO": ("Broadcom", "Technology", "US"),
    # --- Euronext Amsterdam (added to expand universe from 50 to 80) ---
    "RAND.AS": ("Randstad", "Industrials", "NL"),
    "AGN.AS": ("Aegon", "Financials", "NL"),
    "NN.AS": ("NN Group", "Financials", "NL"),
    "BESI.AS": ("BE Semiconductor Industries", "Technology", "NL"),
    "IMCD.AS": ("IMCD", "Materials", "NL"),
    "PRX.AS": ("Prosus", "Consumer Discretionary", "NL"),
    # --- Germany (Xetra) ---
    "SAP.DE": ("SAP", "Technology", "DE"),
    "SIE.DE": ("Siemens", "Industrials", "DE"),
    "ALV.DE": ("Allianz", "Financials", "DE"),
    "DBK.DE": ("Deutsche Bank", "Financials", "DE"),
    "BMW.DE": ("BMW", "Consumer Discretionary", "DE"),
    "MBG.DE": ("Mercedes-Benz Group", "Consumer Discretionary", "DE"),
    "BAS.DE": ("BASF", "Materials", "DE"),
    "DTE.DE": ("Deutsche Telekom", "Communication Services", "DE"),
    "ADS.DE": ("Adidas", "Consumer Discretionary", "DE"),
    # --- France (Euronext Paris) ---
    "MC.PA": ("LVMH", "Consumer Discretionary", "FR"),
    "OR.PA": ("L'Or\u00e9al", "Consumer Staples", "FR"),
    "TTE.PA": ("TotalEnergies", "Energy", "FR"),
    "AIR.PA": ("Airbus", "Industrials", "FR"),
    "SU.PA": ("Schneider Electric", "Industrials", "FR"),
    "SAN.PA": ("Sanofi", "Healthcare", "FR"),
    "BNP.PA": ("BNP Paribas", "Financials", "FR"),
    # --- United Kingdom (LSE) ---
    "HSBA.L": ("HSBC", "Financials", "UK"),
    "AZN.L": ("AstraZeneca", "Healthcare", "UK"),
    "ULVR.L": ("Unilever", "Consumer Staples", "UK"),
    "BP.L": ("BP", "Energy", "UK"),
    "RR.L": ("Rolls-Royce", "Industrials", "UK"),
    "DGE.L": ("Diageo", "Consumer Staples", "UK"),
    "GSK.L": ("GSK", "Healthcare", "UK"),
    "VOD.L": ("Vodafone", "Communication Services", "UK"),
    "BARC.L": ("Barclays", "Financials", "UK"),
}

_PALETTE = ["#5B8DD9", "#4FAE7A", "#D9724F", "#E8A33D", "#9B7EDE", "#8891A0",
            "#5AC8C8", "#D96FA3", "#8FBF5C", "#C97B4A", "#6E9BD9", "#B0885B"]
TICKER_COLORS = {tk: _PALETTE[i % len(_PALETTE)] for i, tk in enumerate(TICKER_INFO)}

MA_FAST_OPTIONS = [10, 20, 50]
MA_SLOW_OPTIONS = [50, 100, 150, 200]
STRATEGIES = ["Buy & Hold", "Equal Weight", "Moving Average", "Momentum", "Cross-Sectional Momentum"]
REBAL_OPTIONS = ["Never", "Monthly", "Quarterly", "Annually"]


def fetch_live_prices(tickers, lookback_days=1100):
    """
    Pull fresh daily closes for the given tickers via yfinance, batched into
    one call. Returns None on any failure (network issue, rate limit, provider
    outage) so the caller can show an explicit error -- this app has no static
    fallback, so a failure here means "no data", not "quietly use old data".
    """
    try:
        import yfinance as yf
        from datetime import datetime, timedelta

        end = datetime.today()
        start = end - timedelta(days=lookback_days)
        raw = yf.download(list(tickers), start=start, end=end, group_by="ticker",
                           threads=True, progress=False, auto_adjust=True)
        if raw is None or raw.empty:
            return None

        rows = []
        for tk in tickers:
            try:
                sub = raw[tk] if len(tickers) > 1 else raw
                data = sub[["Close", "Volume"]].dropna(subset=["Close"])
                if data.empty:
                    continue
                name, sector, _market = TICKER_INFO.get(tk, (tk, "Unknown", "?"))
                for date, row in data.iterrows():
                    vol = row["Volume"]
                    rows.append({
                        "ticker": tk, "sector": sector, "price_date": date, "close": float(row["Close"]),
                        "volume": float(vol) if pd.notna(vol) else np.nan,
                    })
            except (KeyError, TypeError):
                continue

        if not rows:
            return None
        df = pd.DataFrame(rows)
        df["price_date"] = pd.to_datetime(df["price_date"])
        return df
    except Exception:
        return None


@st.cache_data(ttl=300, show_spinner=False)
def fetch_live_prices_cached(tickers, lookback_days=1100):
    """
    Auto-refreshing live fetch used on every normal page load. TTL-cached for
    5 minutes -- Streamlit reruns the whole script on every widget interaction,
    so without this, dragging a slider would fire a fresh Yahoo Finance request
    for 50 tickers on every single rerun. 5 minutes keeps this genuinely "live"
    (prices are never more than a few minutes stale) without hammering the
    data provider or making the UI feel laggy on every interaction.
    """
    return fetch_live_prices(tickers, lookback_days)


def load_prices():
    """
    Live data only -- no CSV, no static fallback. Three paths:
      1. The user hit "Refresh" this session: use that exact snapshot
         (bypasses the 5-min TTL cache, since a manual refresh means
         "I want this right now").
      2. Otherwise, the TTL-cached auto-fetch (still live, just reused for
         up to 5 minutes so the app isn't re-fetching on every rerun).
      3. If Yahoo Finance is unreachable and neither of the above produced
         data, return None -- the caller shows an error + retry rather than
         quietly falling back to something stale.
    """
    if st.session_state.get("live_prices") is not None:
        return st.session_state["live_prices"], st.session_state.get("live_fetched_at")
    df = fetch_live_prices_cached(tuple(TICKER_INFO.keys()))
    if df is not None and not df.empty:
        return df, None
    return None, None


def fmt_pct(v, digits=1):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    s = f"{v*100:.{digits}f}%"
    return ("+" + s) if v > 0 else s


def run_backtest(prices, ma_fast, ma_slow, tx_cost_bps, position_size_pct, initial_capital, roll_window,
                  slippage_bps=0.0, allow_short=False, borrow_rate_annual=0.0,
                  strategy_type="ma_crossover", mr_lookback=20, mr_entry_z=1.5,
                  enable_impact=False, impact_coef=1.0,
                  vol_target_annual=None, max_leverage=2.0, risk_free_pct=0.0):
    """
    Two strategy families, selected by strategy_type:

    "ma_crossover": long (or short, if allow_short) whenever the fast MA is
    above (below) the slow MA -- a trend-following rule.

    "mean_reversion": z-score of price vs. its own rolling mean/std
    (Bollinger-band style). Goes long when price is mr_entry_z standard
    deviations BELOW its rolling mean (oversold), holds until price reverts
    back to the mean (z >= 0), not just until it exits the extreme band --
    that hold-until-reversion behavior is implemented via a sparse signal +
    forward-fill (only the entry and exit rows get an explicit value; rows in
    between stay unset and inherit the held position), which is what makes
    this a genuine "buy the dip and wait for reversion" rule instead of a
    twitchy one that exits the moment price ticks off its extreme. Mirror
    logic on the short side if allow_short=True.

    Position sizing is either a fixed fraction of capital (position_size_pct)
    or, if vol_target_annual is set, dynamically scaled so the position's
    OWN realized volatility (20-day, annualized, using only data through the
    PRIOR close -- no lookahead) tracks that target: leverage = target_vol /
    realized_vol, capped at max_leverage. This is standard risk management --
    the same signal sized the same way delivers very different real risk in
    a calm market vs. a volatile one, and vol targeting normalizes for that
    instead of exposing a fixed dollar amount regardless of conditions.

    Execution cost has three separate components, all charged only on a
    signal transition, matching how a real fill is costed:
      - COMMISSION (tx_cost_bps): flat broker/exchange fee.
      - SLIPPAGE (slippage_bps): flat bid-ask spread / fill-price haircut.
      - MARKET IMPACT (enable_impact, impact_coef): a square-root impact
        model (impact_coef * asset_volatility * sqrt(order_notional / 20-day
        average dollar volume)) -- trading a large position in a thin,
        illiquid name costs more than trading the same size in a heavily
        traded one, which flat bps costs alone can't represent. Needs the
        "volume" column (pulled from yfinance); silently skipped if absent.
        This is still an approximation (Almgren-style square-root law, a
        standard industry heuristic), not a full limit-order-book simulation.

    If allow_short=True, the signal is -1/0/+1 instead of 0/1. A short
    position accrues a daily borrow cost (borrow_rate_annual, annualized) for
    as long as it's held -- the standard cost of a short (paid to borrow the
    shares).
    """
    df = prices.copy().reset_index(drop=True)
    exec_cost = (tx_cost_bps + slippage_bps) / 10000
    base_pos_frac = position_size_pct / 100
    daily_borrow = borrow_rate_annual / 100 / 252

    if strategy_type == "mean_reversion":
        roll_mean = df["close"].rolling(mr_lookback).mean()
        roll_std = df["close"].rolling(mr_lookback).std()
        z = (df["close"] - roll_mean) / roll_std
        df["z_score"] = z

        long_raw = pd.Series(np.nan, index=df.index)
        long_raw[z <= -mr_entry_z] = 1
        long_raw[z >= 0] = 0
        long_state = long_raw.ffill().fillna(0)

        if allow_short:
            short_raw = pd.Series(np.nan, index=df.index)
            short_raw[z >= mr_entry_z] = 1
            short_raw[z <= 0] = 0
            short_state = short_raw.ffill().fillna(0)
        else:
            short_state = pd.Series(0.0, index=df.index)

        df["in_market"] = (long_state - short_state).astype(int)
    else:
        df["ma_fast"] = df["close"].rolling(ma_fast).mean()
        df["ma_slow"] = df["close"].rolling(ma_slow).mean()
        if allow_short:
            df["in_market"] = np.where(df["ma_fast"] > df["ma_slow"], 1, -1)
            df.loc[df["ma_fast"].isna() | df["ma_slow"].isna(), "in_market"] = 0
        else:
            df["in_market"] = (df["ma_fast"] > df["ma_slow"]).astype(int)
    df["daily_return"] = df["close"].pct_change().fillna(0)
    df["prev_signal"] = df["in_market"].shift(1).fillna(0)

    # Volatility-targeted position sizing: leverage computed from realized
    # vol through the PRIOR close only (shifted), so today's sizing decision
    # never uses today's not-yet-known return -- same no-lookahead discipline
    # as the trading signal itself.
    if vol_target_annual and vol_target_annual > 0:
        realized_vol = df["daily_return"].rolling(20).std() * np.sqrt(252)
        leverage = (vol_target_annual / 100) / realized_vol
        leverage = leverage.clip(upper=max_leverage).shift(1).fillna(1.0)
        pos_frac = leverage * base_pos_frac
    else:
        pos_frac = pd.Series(base_pos_frac, index=df.index)

    # Cost scales with the SIZE of the position change, not just whether one
    # happened. A flat->long or long->flat flip trades 1x notional. A direct
    # long->short flip (only possible with allow_short) trades 2x notional --
    # you're closing the long AND opening the short in the same move -- so it
    # should cost twice as much. A flat boolean "did the signal change" flag
    # (an earlier version of this function) charged the same flat fee for
    # both cases, understating the true cost of a direct short flip by half.
    df["signal_change"] = (df["prev_signal"] - df["prev_signal"].shift(1).fillna(0)).abs()
    df["transition"] = df["signal_change"] > 0
    df["eff_return"] = df["prev_signal"] * df["daily_return"] * pos_frac
    # Idle cash: whatever fraction of capital isn't allocated to the position
    # (all of it while flat; the leftover remainder if position_size_pct < 100%
    # or vol-targeting has scaled exposure down) now earns the risk-free rate
    # instead of sitting at a flat 0% -- previously the single-stock engine's
    # cash was always 0% regardless of this setting, an inconsistency with the
    # Portfolio Backtest engine, which already modeled idle cash this way.
    if risk_free_pct and risk_free_pct > 0:
        daily_rf = risk_free_pct / 100 / 252
        uninvested_frac = (1 - df["prev_signal"].abs() * pos_frac).clip(lower=0)
        df["eff_return"] += daily_rf * uninvested_frac
    df["eff_return"] -= exec_cost * pos_frac * df["signal_change"]
    if allow_short and daily_borrow > 0:
        df.loc[df["prev_signal"] == -1, "eff_return"] -= daily_borrow * pos_frac

    df["impact_rate"] = 0.0
    if enable_impact and "volume" in df.columns and df["volume"].notna().any():
        dollar_vol = df["close"] * df["volume"]
        adv20 = dollar_vol.rolling(20).mean()
        asset_vol20 = df["daily_return"].rolling(20).std()
        # Order size uses INITIAL capital as the notional base rather than
        # compounding equity, to avoid a circular dependency (equity itself
        # depends on the return this cost reduces). A simplification worth
        # knowing about: it means impact cost doesn't scale up if the
        # strategy has grown well beyond its starting capital.
        order_notional = initial_capital * pos_frac * df["signal_change"]
        with np.errstate(divide="ignore", invalid="ignore"):
            raw_impact = impact_coef * asset_vol20 * np.sqrt(order_notional / adv20)
        df["impact_rate"] = raw_impact.replace([np.inf, -np.inf], np.nan).fillna(0).clip(upper=0.05)
        df["eff_return"] -= df["impact_rate"]

    df["strat_equity"] = initial_capital * (1 + df["eff_return"]).cumprod()
    df["buyhold_equity"] = initial_capital * (df["close"] / df["close"].iloc[0])

    # Track the actual dollar cost drag, split by source, for turnover reporting.
    df["cost_drag"] = exec_cost * pos_frac * df["signal_change"]
    df["borrow_drag"] = 0.0
    if allow_short and daily_borrow > 0:
        df.loc[df["prev_signal"] == -1, "borrow_drag"] = daily_borrow * pos_frac
    prior_equity = df["strat_equity"].shift(1).fillna(initial_capital)
    df["cost_dollars"] = prior_equity * df["cost_drag"]
    df["borrow_dollars"] = prior_equity * df["borrow_drag"]
    df["impact_dollars"] = df["impact_rate"] * initial_capital

    df["strat_peak"] = df["strat_equity"].cummax()
    df["drawdown_strat"] = (df["strat_equity"] - df["strat_peak"]) / df["strat_peak"]

    df["roll_vol"] = df["eff_return"].rolling(roll_window).std() * np.sqrt(252)
    roll_mean = df["eff_return"].rolling(roll_window).mean() * 252
    df["roll_sharpe"] = roll_mean / df["roll_vol"]
    return df


def get_trades(df):
    """Builds the trade list from prev_signal, which can now be -1/0/+1 (short/
    flat/long) instead of just 0/1. A direct long->short flip with allow_short
    closes one trade and opens another on the same day, rather than needing to
    pass through a flat state -- that's the trade-off in a strategy that's
    always either long or short unless a warmup period leaves the signal at 0."""
    trades = []
    current_side, entry_idx = 0, None
    for i, row in df.iterrows():
        sig = row["prev_signal"]
        if sig != current_side:
            if current_side != 0:
                exit_idx = i - 1
                entry_equity = df["strat_equity"].iloc[entry_idx - 1] if entry_idx > 0 else df["strat_equity"].iloc[0]
                exit_equity = df["strat_equity"].iloc[exit_idx]
                trades.append({
                    "entry_date": df["price_date"].iloc[entry_idx].strftime("%Y-%m-%d"),
                    "exit_date": df["price_date"].iloc[exit_idx].strftime("%Y-%m-%d"),
                    "side": "Long" if current_side == 1 else "Short",
                    "days": int(exit_idx - entry_idx),
                    "ret": exit_equity / entry_equity - 1,
                })
            if sig != 0:
                entry_idx = i
            current_side = sig
    if current_side != 0:
        entry_equity = df["strat_equity"].iloc[entry_idx - 1] if entry_idx > 0 else df["strat_equity"].iloc[0]
        exit_idx = len(df) - 1
        trades.append({
            "entry_date": df["price_date"].iloc[entry_idx].strftime("%Y-%m-%d"),
            "exit_date": df["price_date"].iloc[exit_idx].strftime("%Y-%m-%d") + " (open)",
            "side": "Long" if current_side == 1 else "Short",
            "days": int(exit_idx - entry_idx),
            "ret": df["strat_equity"].iloc[exit_idx] / entry_equity - 1,
        })
    return trades


@st.cache_data(show_spinner=False)
def ticker_backtest_summary(price_df, ma_fast, ma_slow, tx_cost_bps, position_size_pct, initial_capital, roll_window,
                             slippage_bps=0.0, allow_short=False, borrow_rate_annual=0.0,
                             strategy_type="ma_crossover", mr_lookback=20, mr_entry_z=1.5,
                             enable_impact=False, impact_coef=1.0, vol_target_annual=None, max_leverage=2.0,
                             risk_free_pct=0.0):
    """
    Cached per-ticker summary for the "compare all tickers" table.

    This is the expensive part of the Single-Stock Backtest page: with 80
    tickers in the universe, re-running run_backtest() for every ticker on
    every Streamlit rerun (e.g. just clicking a row to switch the active
    ticker, with no parameter change) means redoing ~80 full rolling-window
    backtests for no new information. Caching on (price slice, strategy type
    and all of its parameters, tx cost, slippage, position size, capital,
    roll window, short settings, impact/vol-targeting settings, risk-free
    rate) means a rerun with the same parameters is a cache hit -- only an
    actual change to strategy inputs triggers real recomputation. Returns
    plain scalars, not the full frame, since that's all the comparison table needs.
    """
    bt = run_backtest(price_df, ma_fast, ma_slow, tx_cost_bps, position_size_pct, initial_capital, roll_window,
                       slippage_bps=slippage_bps, allow_short=allow_short, borrow_rate_annual=borrow_rate_annual,
                       strategy_type=strategy_type, mr_lookback=mr_lookback, mr_entry_z=mr_entry_z,
                       enable_impact=enable_impact, impact_coef=impact_coef,
                       vol_target_annual=vol_target_annual, max_leverage=max_leverage, risk_free_pct=risk_free_pct)
    trades = get_trades(bt)
    bh = bt["buyhold_equity"].iloc[-1] / initial_capital - 1
    strat = bt["strat_equity"].iloc[-1] / initial_capital - 1
    eff = bt["eff_return"].iloc[1:]
    vol = eff.std() * np.sqrt(252)
    sharpe = (eff.mean() * 252) / vol if vol > 0 else 0
    return bh, strat, vol, sharpe, len(trades)


def compute_trade_stats(trades):
    if not trades:
        return dict(count=0, win_rate=0, avg_gain=0, avg_loss=0, profit_factor=0, expectancy=0,
                    avg_days=0, longest_win_streak=0, longest_loss_streak=0, avg_trade_return=0)
    rets = np.array([t["ret"] for t in trades])
    days = np.array([t["days"] for t in trades])
    wins, losses = rets[rets > 0], rets[rets <= 0]
    win_rate = len(wins) / len(rets)
    avg_gain = wins.mean() if len(wins) else 0
    avg_loss = losses.mean() if len(losses) else 0
    sum_w, sum_l = wins.sum(), abs(losses.sum())
    profit_factor = sum_w / sum_l if sum_l > 0 else (np.inf if sum_w > 0 else 0)
    expectancy = win_rate * avg_gain + (1 - win_rate) * avg_loss
    longest_win = longest_loss = cur_win = cur_loss = 0
    for r in rets:
        if r > 0:
            cur_win += 1; cur_loss = 0
        else:
            cur_loss += 1; cur_win = 0
        longest_win = max(longest_win, cur_win)
        longest_loss = max(longest_loss, cur_loss)
    return dict(count=len(rets), win_rate=win_rate, avg_gain=avg_gain, avg_loss=avg_loss,
                profit_factor=profit_factor, expectancy=expectancy, avg_days=days.mean(),
                longest_win_streak=longest_win, longest_loss_streak=longest_loss, avg_trade_return=rets.mean())


def run_monte_carlo_trades(trades, initial_capital, n_sims=1000, benchmark_final=None):
    rets = np.array([t["ret"] for t in trades])
    n_trades = len(rets)
    if n_trades < 2:
        return None
    rng = np.random.default_rng()
    sims = rng.choice(rets, size=(n_sims, n_trades), replace=True)
    paths = initial_capital * np.cumprod(1 + sims, axis=1)
    paths = np.hstack([np.full((n_sims, 1), initial_capital), paths])
    finals = paths[:, -1]
    x = list(range(n_trades + 1))
    band_series = {q: [np.percentile(paths[:, i], q) for i in x] for q in [5, 25, 50, 75, 95]}
    actual_path = [initial_capital]
    eq = initial_capital
    for r in rets:
        eq *= (1 + r)
        actual_path.append(eq)

    # Per-simulation max drawdown, for "probability of a >20% drawdown" --
    # needs the full path per simulation, not just the final value.
    running_max = np.maximum.accumulate(paths, axis=1)
    drawdowns = (paths - running_max) / running_max
    max_dd_per_sim = drawdowns.min(axis=1)

    result = dict(x=x, bands=band_series, actual_path=actual_path, finals=finals,
                  median_final=np.median(finals), p5_final=np.percentile(finals, 5),
                  p95_final=np.percentile(finals, 95), loss_prob=(finals < initial_capital).mean(),
                  pos_prob=(finals > initial_capital).mean(),
                  dd20_prob=(max_dd_per_sim <= -0.20).mean())
    if benchmark_final is not None:
        result["beat_bench_prob"] = (finals > benchmark_final).mean()
    return result


def turnover_stats(df, capital):
    """Trading activity and the actual drag it created, for cost transparency."""
    n_transitions = int(df["transition"].sum())
    total_cost_dollars = df["cost_dollars"].sum()
    total_cost_pct = total_cost_dollars / capital
    total_borrow_dollars = df["borrow_dollars"].sum() if "borrow_dollars" in df.columns else 0.0
    total_borrow_pct = total_borrow_dollars / capital
    total_impact_dollars = df["impact_dollars"].sum() if "impact_dollars" in df.columns else 0.0
    total_impact_pct = total_impact_dollars / capital
    n_days = (df["price_date"].iloc[-1] - df["price_date"].iloc[0]).days
    years = max(n_days / 365.25, 1 / 252)
    transitions_per_year = n_transitions / years
    return dict(n_transitions=n_transitions, total_cost_dollars=total_cost_dollars,
                total_cost_pct=total_cost_pct, transitions_per_year=transitions_per_year,
                total_borrow_dollars=total_borrow_dollars, total_borrow_pct=total_borrow_pct,
                total_impact_dollars=total_impact_dollars, total_impact_pct=total_impact_pct)


def return_attribution(stats, benchmark_return):
    """Simple additive decomposition: benchmark return + stock selection effect + strategy timing effect.
    This is an arithmetic approximation (not geometric/compounding-exact) -- standard for a quick read,
    but the three pieces won't sum to the total to the last decimal."""
    selection_effect = stats["buy_hold_return"] - benchmark_return
    timing_effect = stats["strat_return"] - stats["buy_hold_return"]
    return dict(benchmark_return=benchmark_return, selection_effect=selection_effect,
                timing_effect=timing_effect, total=stats["strat_return"])


def summary_stats(df, capital, benchmark_daily_return=None, risk_free_pct=0.0):
    buy_hold_return = df["buyhold_equity"].iloc[-1] / capital - 1
    strat_return = df["strat_equity"].iloc[-1] / capital - 1
    eff = df["eff_return"].iloc[1:]
    # Sharpe/Sortino are computed on EXCESS return (over the risk-free rate),
    # matching the Portfolio Backtest engine's convention and the actual
    # definition of Sharpe. This matters now specifically because idle cash
    # can earn risk_free_pct (see run_backtest): without subtracting rf here
    # too, a strategy that sits in cash a lot would show an artificially
    # inflated Sharpe purely from earning risk-free carry on quiet, low-vol
    # days, which is the opposite of what modeling idle cash honestly should do.
    daily_rf = (risk_free_pct / 100 / 252) if risk_free_pct else 0.0
    excess = eff - daily_rf
    vol = eff.std() * np.sqrt(252)
    sharpe = (excess.mean() * 252) / vol if vol > 0 else 0
    downside = excess[excess < 0]
    downside_dev = np.sqrt((downside ** 2).mean()) * np.sqrt(252) if len(downside) else 0
    sortino = (excess.mean() * 252) / downside_dev if downside_dev > 0 else 0
    max_dd = df["drawdown_strat"].min()
    n_days = (df["price_date"].iloc[-1] - df["price_date"].iloc[0]).days
    years = max(n_days / 365.25, 1 / 252)
    cagr = (df["strat_equity"].iloc[-1] / capital) ** (1 / years) - 1
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    info_ratio = None
    if benchmark_daily_return is not None:
        bench = benchmark_daily_return.values[1:]
        n = min(len(eff), len(bench))
        active = eff.values[:n] - bench[:n]
        ir_vol = active.std() * np.sqrt(252)
        info_ratio = (active.mean() * 252) / ir_vol if ir_vol > 0 else 0
    return dict(buy_hold_return=buy_hold_return, strat_return=strat_return, vol=vol, sharpe=sharpe,
                sortino=sortino, max_dd=max_dd, cagr=cagr, calmar=calmar, info_ratio=info_ratio)


def deflated_sharpe_ratio(trial_daily_sharpes, best_eff_returns):
    """
    Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).

    Scanning N parameter combinations and reporting whichever had the best
    Sharpe is a multiple-comparisons problem: with enough trials, some
    combination will look good purely from noise, even with zero real skill.
    This corrects for that by asking "what Sharpe would the best of N random,
    skill-less trials be expected to produce by chance alone?" (sr0 below),
    then computing the probability that the observed best Sharpe is actually
    above that noise threshold, given the actual sample size and the return
    distribution's skew/kurtosis (which affect how much to trust a Sharpe
    estimate -- fat tails and negative skew make Sharpe ratios less reliable).

    trial_daily_sharpes: daily (non-annualized) Sharpe from every combination
        tried in the search, including the winner.
    best_eff_returns: the daily return series of the winning combination,
        used to estimate its own Sharpe, sample size, skew, and kurtosis.

    Returns None if there's too little data to say anything meaningful.
    """
    n_trials = len(trial_daily_sharpes)
    n_obs = len(best_eff_returns)
    if n_trials < 2 or n_obs < 30:
        return None
    sr_std = float(np.std(trial_daily_sharpes, ddof=1))
    if sr_std == 0:
        return None
    euler_gamma = 0.5772156649
    nd = NormalDist()
    # Expected Sharpe of the BEST of n_trials iid draws from a skill-less
    # (zero true Sharpe) population with this observed cross-trial variance.
    sr0 = sr_std * ((1 - euler_gamma) * nd.inv_cdf(1 - 1 / n_trials) + euler_gamma * nd.inv_cdf(1 - 1 / (n_trials * np.e)))

    sr_hat = float(best_eff_returns.mean() / best_eff_returns.std()) if best_eff_returns.std() > 0 else 0.0
    skew = float(best_eff_returns.skew())
    kurt = float(best_eff_returns.kurtosis()) + 3.0  # pandas gives excess kurtosis; the PSR formula wants raw kurtosis
    denom = np.sqrt(max(1e-12, 1 - skew * sr_hat + ((kurt - 1) / 4) * sr_hat ** 2))
    z = (sr_hat - sr0) * np.sqrt(n_obs - 1) / denom
    psr = nd.cdf(z)
    return dict(dsr=psr, sr0_annualized=sr0 * np.sqrt(252), sr_hat_annualized=sr_hat * np.sqrt(252), n_trials=n_trials)


def lo_adjusted_sharpe(daily_returns, q=252, max_lag=20):
    """
    Autocorrelation-adjusted Sharpe Ratio (Lo, 2002).

    The standard "multiply by sqrt(252)" annualization assumes daily returns
    are independent. Trend-following strategies hold positions for multiple
    consecutive days, which makes daily returns positively autocorrelated --
    and under positive autocorrelation, naive sqrt(q) annualization OVERSTATES
    the true annualized Sharpe (and therefore overstates how statistically
    significant it looks). This computes the correct annualization factor
    given the return series' own autocorrelation structure at lags 1..max_lag
    (sample autocorrelations beyond ~20 lags are usually too noisy to trust,
    so this truncates there rather than summing out to lag q-1 as the exact
    formula technically specifies).
    """
    r = daily_returns.dropna()
    n = len(r)
    if n < max_lag + 30:
        return None
    std_r = r.std()
    if std_r == 0:
        return None
    sr_daily = r.mean() / std_r
    rho = [r.autocorr(lag=k) for k in range(1, max_lag + 1)]
    rho = [0.0 if (v is None or (isinstance(v, float) and np.isnan(v))) else v for v in rho]
    correction = 1 + 2 * sum((1 - k / q) * rho[k - 1] for k in range(1, max_lag + 1))
    correction = max(correction, 1e-6)
    sr_naive = sr_daily * np.sqrt(q)
    sr_adjusted = sr_daily * np.sqrt(q) / np.sqrt(correction)
    return dict(naive=sr_naive, adjusted=sr_adjusted, correction_factor=correction, rho1=rho[0])


def bootstrap_sharpe_cagr_ci(daily_returns, n_boot=500, block_size=20, ci=0.90, seed=7, risk_free_pct=0.0):
    """
    Moving-block bootstrap confidence interval for annualized Sharpe and CAGR.

    A plain day-by-day (iid) bootstrap would implicitly assume returns are
    independent -- exactly the assumption that doesn't hold for a strategy
    that stays in the same position for stretches of days. Resampling whole
    BLOCKS of consecutive returns instead preserves that short-term
    autocorrelation structure in each resampled path, giving a more honest
    (typically wider, more realistic) confidence interval than a naive
    bootstrap would.

    Sharpe is computed net of risk_free_pct (consistent with summary_stats
    elsewhere in this app); CAGR is NOT -- CAGR should reflect actual growth,
    not growth in excess of the risk-free rate, so subtracting rf there would
    mislabel "excess growth" as "CAGR."
    """
    r = daily_returns.dropna().values
    n = len(r)
    if n < block_size * 5:
        return None
    daily_rf = (risk_free_pct / 100 / 252) if risk_free_pct else 0.0
    n_blocks = int(np.ceil(n / block_size))
    rng = np.random.default_rng(seed)
    sharpes, cagrs = [], []
    years = n / 252
    for _ in range(n_boot):
        starts = rng.integers(0, n - block_size, size=n_blocks)
        sample = np.concatenate([r[s:s + block_size] for s in starts])[:n]
        sd = sample.std()
        if sd > 0:
            sharpes.append(((sample.mean() - daily_rf) / sd) * np.sqrt(252))
        total_ret = np.prod(1 + sample) - 1
        if years > 0 and (1 + total_ret) > 0:
            cagrs.append((1 + total_ret) ** (1 / years) - 1)
    if not sharpes or not cagrs:
        return None
    sharpes, cagrs = np.array(sharpes), np.array(cagrs)
    lo_pct, hi_pct = (1 - ci) / 2 * 100, (1 - (1 - ci) / 2) * 100
    return dict(
        sharpe_median=float(np.median(sharpes)), sharpe_lo=float(np.percentile(sharpes, lo_pct)),
        sharpe_hi=float(np.percentile(sharpes, hi_pct)),
        cagr_median=float(np.median(cagrs)), cagr_lo=float(np.percentile(cagrs, lo_pct)),
        cagr_hi=float(np.percentile(cagrs, hi_pct)), n_boot=n_boot, ci=ci,
    )


def newey_west_alpha_se(x, y, alpha_hat, beta_hat, max_lag=None):
    """
    Newey-West (1987) heteroskedasticity- and autocorrelation-consistent
    (HAC) standard error for the OLS intercept (alpha) in y = alpha + beta*x + e.

    Plain OLS standard errors assume the regression residuals are
    independent. If a strategy's returns are autocorrelated -- the same
    phenomenon behind the Lo-adjusted Sharpe elsewhere in this app, and
    plausible for anything that holds positions across multiple days --
    residuals inherit that autocorrelation, and naive OLS standard errors
    come out too small. Too-small standard errors mean too-large t-stats,
    which makes alpha look more statistically significant than it actually is.
    """
    n = len(x)
    if max_lag is None:
        max_lag = int(np.floor(4 * (n / 100) ** (2 / 9)))
    max_lag = max(max_lag, 1)
    X = np.column_stack([np.ones(n), x])
    resid = y - (alpha_hat + beta_hat * x)
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return np.nan
    u = X * resid[:, None]
    S = u.T @ u
    for lag in range(1, min(max_lag, n - 1) + 1):
        w = 1 - lag / (max_lag + 1)
        gamma = u[lag:].T @ u[:-lag]
        S += w * (gamma + gamma.T)
    cov = XtX_inv @ S @ XtX_inv
    return float(np.sqrt(max(cov[0, 0], 0)))


def compute_beta_alpha(strat_returns, bench_returns, freq=252):
    """
    Single-factor (CAPM-style) regression of strategy returns on benchmark
    returns: strat_r = alpha + beta * bench_r + residual. Answers the
    question a quant asks first about any "edge": is this actually generating
    excess return, or is it just a leveraged/correlated bet on the benchmark
    going up? Beta > 1 means more market-sensitive than the benchmark; alpha
    is the annualized return left over after removing that market exposure,
    with a Newey-West (HAC-robust) t-stat for whether it's distinguishable
    from zero (|t| > ~2 is the conventional significance bar at ~95%
    confidence) -- not a plain-OLS t-stat, which would overstate significance
    if the strategy's returns are autocorrelated (see newey_west_alpha_se).
    """
    aligned = pd.concat([strat_returns.rename("strat"), bench_returns.rename("bench")], axis=1, join="inner").dropna()
    if len(aligned) < 30:
        return None
    x, y = aligned["bench"].values, aligned["strat"].values
    x_mean, y_mean = x.mean(), y.mean()
    var_x = ((x - x_mean) ** 2).sum()
    if var_x == 0:
        return None
    beta = ((x - x_mean) * (y - y_mean)).sum() / var_x
    alpha_daily = y_mean - beta * x_mean
    y_pred = alpha_daily + beta * x
    ss_res = ((y - y_pred) ** 2).sum()
    ss_tot = ((y - y_mean) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    n = len(x)
    se_alpha_nw = newey_west_alpha_se(x, y, alpha_daily, beta)
    t_alpha = alpha_daily / se_alpha_nw if se_alpha_nw and se_alpha_nw > 0 else np.nan
    return dict(beta=float(beta), alpha_annual=float(alpha_daily * freq), r2=float(r2),
                t_alpha=float(t_alpha) if pd.notna(t_alpha) else None, n=n)


def compute_rebalance_flags(dates, freq):
    n = len(dates)
    flags = np.zeros(n, dtype=bool)
    flags[0] = True
    if freq == "Never":
        return flags
    prev_period = None
    for i, d in enumerate(dates):
        if freq == "Monthly":
            period = (d.year, d.month)
        elif freq == "Quarterly":
            period = (d.year, (d.month - 1) // 3)
        elif freq == "Annually":
            period = d.year
        else:
            period = None
        if period != prev_period:
            flags[i] = True
            prev_period = period
    return flags


def strategy_weights_at(t, strategy, tickers, base_weights, price_matrix, ma_fast, ma_slow, momentum_lookback, top_k=None):
    if strategy == "Buy & Hold":
        return dict(base_weights)
    if strategy == "Equal Weight":
        n = len(tickers)
        return {tk: 1 / n for tk in tickers}
    if strategy == "Moving Average":
        active = []
        for tk in tickers:
            closes = price_matrix[tk]
            if t < ma_slow - 1:
                continue
            fast_ma = closes[t - ma_fast + 1: t + 1].mean()
            slow_ma = closes[t - ma_slow + 1: t + 1].mean()
            if fast_ma > slow_ma:
                active.append(tk)
        if not active:
            return {tk: 0 for tk in tickers}
        w = 1 / len(active)
        return {tk: (w if tk in active else 0) for tk in tickers}
    if strategy == "Momentum":
        active = []
        for tk in tickers:
            closes = price_matrix[tk]
            if t < momentum_lookback:
                continue
            mom = closes[t] / closes[t - momentum_lookback] - 1
            if mom > 0:
                active.append(tk)
        if not active:
            return {tk: 0 for tk in tickers}
        w = 1 / len(active)
        return {tk: (w if tk in active else 0) for tk in tickers}
    if strategy == "Cross-Sectional Momentum":
        # Unlike "Momentum" above (a per-asset filter: is THIS asset's own
        # trailing return positive?), this ranks assets AGAINST EACH OTHER
        # and holds only the relative winners -- the standard definition of
        # cross-sectional momentum in factor research, as opposed to
        # absolute/time-series momentum.
        k = top_k or max(1, len(tickers) // 2)
        momentums = {}
        for tk in tickers:
            closes = price_matrix[tk]
            if t >= momentum_lookback:
                momentums[tk] = closes[t] / closes[t - momentum_lookback] - 1
        if not momentums:
            return {tk: 0 for tk in tickers}
        ranked = sorted(momentums.items(), key=lambda kv: kv[1], reverse=True)
        # Only hold names that are BOTH top-ranked AND have positive momentum --
        # in a broad selloff where nothing has positive trailing returns, the
        # "least-bad" names still shouldn't get bought just to fill k slots.
        top = [tk for tk, m in ranked[:k] if m > 0]
        if not top:
            return {tk: 0 for tk in tickers}
        w = 1 / len(top)
        return {tk: (w if tk in top else 0) for tk in tickers}
    return {tk: 1 / len(tickers) for tk in tickers}


def run_portfolio_backtest(prices_by_ticker, tickers, base_weights, strategy, rebal_freq,
                            ma_fast, ma_slow, momentum_lookback, tx_cost_bps, initial_capital, risk_free_pct,
                            top_k=None):
    common_dates = None
    for tk in tickers:
        d = set(prices_by_ticker[tk]["price_date"])
        common_dates = d if common_dates is None else (common_dates & d)
    common_dates = sorted(common_dates)
    n = len(common_dates)

    price_matrix = {}
    for tk in tickers:
        s = prices_by_ticker[tk].set_index("price_date")["close"]
        price_matrix[tk] = s.reindex(common_dates).values

    rebal_flags = compute_rebalance_flags(common_dates, rebal_freq)
    tx_cost_frac = tx_cost_bps / 10000

    weights = strategy_weights_at(0, strategy, tickers, base_weights, price_matrix, ma_fast, ma_slow, momentum_lookback, top_k)
    invested_frac = sum(weights.values())
    if invested_frac == 0:
        weights = {tk: 1 / len(tickers) for tk in tickers}
        invested_frac = 1.0

    shares = {tk: (initial_capital * weights[tk]) / price_matrix[tk][0] if price_matrix[tk][0] > 0 else 0 for tk in tickers}
    cash = initial_capital * (1 - invested_frac)
    portfolio_value = np.zeros(n)
    portfolio_value[0] = initial_capital
    weight_snapshots = [dict(weights)]
    rebalance_idxs = [0]
    turnover_history = []
    total_cost_dollars = 0.0

    for t in range(1, n):
        invested_val = sum(shares[tk] * price_matrix[tk][t] for tk in tickers)
        val = invested_val + cash
        if rebal_flags[t]:
            new_weights = strategy_weights_at(t - 1, strategy, tickers, base_weights, price_matrix, ma_fast, ma_slow, momentum_lookback, top_k)
            current_weights = {tk: (shares[tk] * price_matrix[tk][t] / val if val > 0 else 0) for tk in tickers}
            turnover_amt = sum(abs(new_weights[tk] - current_weights[tk]) for tk in tickers)
            cost = turnover_amt * tx_cost_frac * val
            total_cost_dollars += cost
            turnover_history.append(turnover_amt)
            val -= cost
            for tk in tickers:
                shares[tk] = (val * new_weights[tk]) / price_matrix[tk][t] if price_matrix[tk][t] > 0 else 0
            cash = val * (1 - sum(new_weights.values()))
            weights = new_weights
            weight_snapshots.append(dict(weights))
            rebalance_idxs.append(t)
        portfolio_value[t] = val

    portfolio_value = pd.Series(portfolio_value, index=common_dates)
    daily_return = portfolio_value.pct_change().fillna(0)
    rf_daily = (risk_free_pct / 100) / 252
    excess_return = daily_return - rf_daily

    vol = daily_return.std() * np.sqrt(252)
    excess_vol = excess_return.std() * np.sqrt(252)
    sharpe = (excess_return.mean() * 252) / excess_vol if excess_vol > 0 else 0
    downside = excess_return[excess_return < 0]
    downside_dev = np.sqrt((downside ** 2).mean()) * np.sqrt(252) if len(downside) else 0
    sortino = (excess_return.mean() * 252) / downside_dev if downside_dev > 0 else 0

    peak = portfolio_value.cummax()
    drawdown = (portfolio_value - peak) / peak
    max_dd = drawdown.min()

    n_days = (common_dates[-1] - common_dates[0]).days
    years = max(n_days / 365.25, 1 / 252)
    total_return = portfolio_value.iloc[-1] / initial_capital - 1
    cagr = (portfolio_value.iloc[-1] / initial_capital) ** (1 / years) - 1
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    avg_turnover = float(np.mean(turnover_history)) if turnover_history else 0.0

    return dict(dates=common_dates, portfolio_value=portfolio_value, drawdown=drawdown,
                weight_snapshots=weight_snapshots, rebalance_idxs=rebalance_idxs,
                stats=dict(total_return=total_return, cagr=cagr, vol=vol, sharpe=sharpe, sortino=sortino,
                           calmar=calmar, max_dd=max_dd, n_rebalances=len(rebalance_idxs),
                           avg_turnover=avg_turnover, total_cost_dollars=total_cost_dollars,
                           total_cost_pct=total_cost_dollars / initial_capital))


import plotly.io as pio

st.markdown("""
<style>
/* KSE TERMINAL V4 */
.block-container { max-width: 1500px; padding-top: 3.2rem; }
[data-testid="stMetric"] {
    background: #12161D;
    border: 1px solid #242B35;
    border-radius: 5px;
    padding: 10px 12px;
}
.section-label {
    color:#737D8B; font:600 10px ui-monospace, monospace;
    letter-spacing:1.4px; margin:16px 0 8px;
}
/* Animated marquee ticker tape: continuous horizontal scroll, pauses on
   hover so you can actually read a name, edges fade instead of hard-cutting
   so it reads as a deliberate design element rather than clipped overflow. */
.kse-tape {
    position: relative; overflow: hidden;
    border: 1px solid #1D232C; border-radius: 5px; margin-bottom: 10px;
    -webkit-mask-image: linear-gradient(90deg, transparent 0, black 24px, black calc(100% - 24px), transparent 100%);
    mask-image: linear-gradient(90deg, transparent 0, black 24px, black calc(100% - 24px), transparent 100%);
}
.kse-tape-track {
    display: flex; width: max-content;
    animation: kse-tape-scroll linear infinite;
}
.kse-tape:hover .kse-tape-track { animation-play-state: paused; }
@keyframes kse-tape-scroll {
    from { transform: translateX(0); }
    to { transform: translateX(-50%); }
}
.kse-tape-item {
    display: flex; align-items: baseline; gap: 6px;
    padding: 8px 16px; white-space: nowrap;
    border-right: 1px solid #1D232C;
    font-family: ui-monospace, monospace; font-size: 12px;
}
/* Clickable ticker links (tape, top movers, watchlist, search results) --
   real navigation via query params under the hood, styled so it reads as a
   terminal row, not a default underlined blue web link. */
a.kse-link, a.kse-link:visited {
    text-decoration: none; color: inherit; cursor: pointer;
}
a.kse-link:hover { opacity: 0.75; }
.kse-mover-row {
    display: flex; justify-content: space-between; align-items: center;
    font-family: ui-monospace, monospace; font-size: 12px;
    padding: 5px 8px; border-bottom: 1px solid #1D232C; border-radius: 3px;
}
.kse-mover-row:hover { background: #12161D; }
/* Selectbox styling to match the terminal look (labels are handled per
   widget via label_visibility, not hidden globally, so other dropdowns
   like Ticker/Strategy keep their labels). */
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background:#0D1117 !important; border:1px solid #242B35 !important;
    border-radius:4px !important; color:#E7EBF0 !important;
    font:600 12px ui-monospace, monospace !important; letter-spacing:.6px;
}
</style>
""", unsafe_allow_html=True)

pio.templates["backtest_dark"] = go.layout.Template(**PLOTLY_TEMPLATE)
pio.templates.default = "backtest_dark"

all_prices, live_fetched_at = load_prices()

if all_prices is None or all_prices.empty:
    st.markdown("""
    <div style="font-family:ui-monospace,monospace;font-size:12px;color:#6B7684;letter-spacing:1.5px;">
      KSE CAPITAL MARKETS TERMINAL
    </div>
    <div style="font-size:30px;font-weight:700;color:#E8ECF1;line-height:1.25;margin-bottom:16px;">BACKTEST PLATFORM</div>
    """, unsafe_allow_html=True)
    st.error(
        "Live market data unavailable right now (Yahoo Finance is unreachable, rate-limited, or the "
        "`yfinance` package isn't installed). This app depends entirely on live data -- there is no "
        "bundled CSV to fall back to -- so nothing further can render until this succeeds."
    )
    st.caption("If this keeps happening: check your network connection, confirm `pip install yfinance` "
               "succeeded, or wait a few minutes in case Yahoo Finance is rate-limiting requests.")
    if st.button("\u21BB Retry live fetch", type="primary"):
        fetch_live_prices_cached.clear()
        st.session_state.pop("live_prices", None)
        st.session_state.pop("live_fetched_at", None)
        st.rerun()
    st.stop()

all_tickers = sorted(all_prices["ticker"].unique())

# --- Cross-page navigation via query params ---
# Streamlit has no real onClick-to-Python-callback from raw injected HTML
# (the ticker tape, top movers, watchlist, search results are all plain
# markdown/HTML, not native widgets). A plain <a href="?goto_ticker=...">
# link is the actual mechanism: clicking it reloads the app with that param
# set, we consume it once here (seeding session_state so the relevant
# selectbox opens pre-filled), then clear the URL so a later manual refresh
# doesn't re-trigger the same navigation. This is a real page reload under
# the hood, not a seamless SPA transition -- Streamlit's rerun-the-whole-
# script model doesn't support that regardless of how this is built.
import urllib.parse


def nav_link(ticker, module="SINGLE-STOCK BACKTEST"):
    return f"?goto_ticker={urllib.parse.quote(ticker)}&goto_module={urllib.parse.quote(module)}"


def watch_toggle_link(ticker):
    return f"?toggle_watch={urllib.parse.quote(ticker)}"


def view_link(ticker):
    """Opens the Stock Detail panel on the Markets page -- the 'click a
    ticker anywhere and land on its profile, not straight into a backtest'
    flow. A separate, explicit Backtest button inside that panel does the
    nav_link() jump when that's actually what the person wants."""
    return f"?view_ticker={urllib.parse.quote(ticker)}"


def sector_link(sector):
    return f"?filter_sector={urllib.parse.quote(sector)}"


qp = st.query_params
if "goto_ticker" in qp and qp["goto_ticker"] in all_tickers:
    st.session_state["main_ticker"] = qp["goto_ticker"]
    st.session_state["module_select"] = qp.get("goto_module", "SINGLE-STOCK BACKTEST")
    st.query_params.clear()
    st.rerun()
if "toggle_watch" in qp:
    tk = qp["toggle_watch"]
    watchlist = st.session_state.setdefault("watchlist", set())
    if tk in watchlist:
        watchlist.discard(tk)
    elif tk in all_tickers:
        watchlist.add(tk)
    st.query_params.clear()
    st.rerun()
if "view_ticker" in qp and qp["view_ticker"] in all_tickers:
    st.session_state["viewing_ticker"] = qp["view_ticker"]
    st.session_state["module_select"] = "MARKETS"  # Stock Detail only renders on Markets; the
    st.query_params.clear()                        # search box is visible on every page, so a
    st.rerun()                                      # click from elsewhere needs to land there too.
if "filter_sector" in qp:
    st.session_state["filter_sector"] = qp["filter_sector"]
    st.session_state["module_select"] = "MARKETS"  # same reasoning as view_ticker above
    st.session_state.pop("viewing_ticker", None)
    st.query_params.clear()
    st.rerun()

# --- Terminal-style header (extra top padding above avoids clipping under
# Streamlit's own toolbar) ---
from datetime import datetime as _dt
st.markdown(f"""
<div style="display:flex;justify-content:space-between;align-items:flex-end;
border-bottom:1px solid #2A313C;padding-bottom:10px;margin-bottom:10px;">
  <div>
    <div style="font-family:ui-monospace,monospace;font-size:12px;color:#6B7684;letter-spacing:1.5px;">
      KSE CAPITAL MARKETS TERMINAL
    </div>
    <div style="font-size:30px;font-weight:700;color:#E8ECF1;line-height:1.25;">BACKTEST PLATFORM</div>
  </div>
  <div style="font-family:ui-monospace,monospace;text-align:right;color:#8891A0;font-size:12px;">
    <span style="color:#4FAE7A;">\u25CF DATA ONLINE</span><br>
    {_dt.now().strftime('%H:%M')}
  </div>
</div>
""", unsafe_allow_html=True)

# --- Global search: type a ticker or company name, get a result card with
# quick actions (open backtest, add to portfolio, watchlist) -- the
# "search box" from the vision doc, not just a filtered dropdown. ---
search_col, _sp = st.columns([2, 4])
search_q = search_col.text_input("\U0001F50E Search securities...", value="", key="global_search",
                                  placeholder="AAPL, ASML, Nvidia...", label_visibility="collapsed")
if search_q.strip():
    q = search_q.strip().lower()
    all_sectors = sorted(set(v[1] for v in TICKER_INFO.values()))
    sector_matches = [s for s in all_sectors if q in s.lower()][:2]
    ticker_matches = [tk for tk in all_tickers if q in tk.lower() or q in TICKER_INFO.get(tk, (tk,))[0].lower()][:6]
    if not sector_matches and not ticker_matches:
        st.caption(f"No matches for \u201c{search_q}\u201d.")
    for sec in sector_matches:
        n_in_sector = sum(1 for v in TICKER_INFO.values() if v[1] == sec)
        st.markdown(
            f"<a class='kse-link' href='{sector_link(sec)}' target='_self'><div class='kse-mover-row'>"
            f"<span style='color:#E7EBF0;font-weight:600;'>\U0001F4C1 {sec}</span>"
            f"<span style='color:#8891A0;'>{n_in_sector} stocks in this sector \u2014 view all \u2192</span></div></a>",
            unsafe_allow_html=True,
        )
    for tk in ticker_matches:
        d = all_prices[all_prices["ticker"] == tk].sort_values("price_date")
        name, sector, market = TICKER_INFO.get(tk, (tk, "Unknown", "?"))
        with st.container():
            rcol1, rcol2, rcol3, rcol4, rcol5, rcol6 = st.columns([2, 1.6, 1.2, 1.2, 1, 1])
            if len(d) >= 2:
                px = float(d["close"].iloc[-1])
                chg = float(d["close"].iloc[-1] / d["close"].iloc[-2] - 1)
                rcol1.markdown(f"**{tk}** \u2014 {name}")
                rcol2.markdown(f"{sector} \u00b7 {market}")
                rcol3.markdown(f"{px:,.2f} " + (":green[\u25B2]" if chg >= 0 else ":red[\u25BC]") + f" {chg*100:+.2f}%")
            else:
                rcol1.markdown(f"**{tk}** \u2014 {name}")
                rcol2.markdown(f"{sector} \u00b7 {market}")
                rcol3.markdown("\u2014")
            if rcol4.button("Open Backtest", key=f"search_open_{tk}"):
                st.session_state["main_ticker"] = tk
                st.session_state["module_select"] = "SINGLE-STOCK BACKTEST"
                st.rerun()
            portfolio_staging = st.session_state.setdefault("portfolio_staging", [])
            if rcol5.button("+ Portfolio", key=f"search_addport_{tk}", disabled=tk in portfolio_staging,
                             help="Add to Portfolio Backtest's asset picker"):
                portfolio_staging.append(tk)
                st.session_state["module_select"] = "PORTFOLIO BACKTEST"
                st.rerun()
            watchlist = st.session_state.setdefault("watchlist", set())
            star_label = "\u2605" if tk in watchlist else "\u2606"
            if rcol6.button(star_label, key=f"search_watch_{tk}", help="Toggle watchlist"):
                if tk in watchlist:
                    watchlist.discard(tk)
                else:
                    watchlist.add(tk)
                st.rerun()

# --- Single navigation system: one control drives everything below. No
# second radio/nav bar, and no fake decorative nav that duplicates it.
# Placed before the ticker tape so the page reads top-to-bottom as
# header -> search -> navigation -> tape, matching how someone actually
# orients on a terminal (find your bearings, then see the live feed). ---
MODULES = ["MARKETS", "SINGLE-STOCK BACKTEST", "PORTFOLIO BACKTEST"]
nav_cols = st.columns([1.4, 2.2, 1])
with nav_cols[0]:
    label_col, select_col = st.columns([1, 2.2])
    label_col.markdown(
        "<div style='padding-top:8px;font:600 11px ui-monospace,monospace;color:#737D8B;letter-spacing:1px;'>MODULE</div>",
        unsafe_allow_html=True,
    )
    page = select_col.selectbox("Module", MODULES, label_visibility="collapsed", key="module_select")
with nav_cols[1]:
    if live_fetched_at:
        st.caption(f"\U0001F7E2 Live data \u00b7 refreshed {live_fetched_at.strftime('%d %b %H:%M')} \u00b7 {len(all_tickers)} tickers")
    else:
        st.caption(f"\U0001F7E2 Live data \u00b7 auto-refreshes every 5 min \u00b7 {len(all_tickers)} tickers")
with nav_cols[2]:
    if st.button("\u21BB Refresh", width="stretch", help="Pull the latest prices live, right now"):
        with st.spinner(f"Fetching {len(TICKER_INFO)} tickers from Yahoo Finance..."):
            fresh = fetch_live_prices(list(TICKER_INFO.keys()))
        if fresh is not None and not fresh.empty:
            st.session_state["live_prices"] = fresh
            st.session_state["live_fetched_at"] = _dt.now()
            st.success(f"Live data loaded: {fresh['ticker'].nunique()} tickers.")
            st.rerun()
        else:
            st.error("Live fetch failed (data provider unreachable or rate-limited). Still showing the last "
                      "successful live pull -- there is no static fallback, so try again in a moment.")

# --- Animated ticker tape: ALL tickers in the universe, continuously
# scrolling -- not a curated subset like before. Still visually distinct
# from the Market Breadth/Sector/Movers sections below (thin single line vs.
# full sections), so it reads as a live tape, not a repeat of them. The item
# list is rendered twice back-to-back and the CSS animates exactly -50% of
# the track's width, so the loop is seamless (the moment the first copy
# scrolls fully offscreen, the second copy is sitting in the same position
# the first one started in). Each item is now a real link -- pauses on
# hover (existing behavior), click to jump straight to that ticker's backtest.
items_html = ""
for tk in all_tickers:
    d = all_prices[all_prices["ticker"] == tk].sort_values("price_date")
    if len(d) >= 2:
        last = float(d["close"].iloc[-1])
        day = float(d["close"].iloc[-1] / d["close"].iloc[-2] - 1)
        color = "#4FAE7A" if day >= 0 else "#D9724F"
        arrow = "\u25B2" if day >= 0 else "\u25BC"
        items_html += (
            f"<a class='kse-link' href='{view_link(tk)}' target='_self'><div class='kse-tape-item'>"
            f"<span style='color:#E8ECF1; font-weight:700;'>{tk}</span>"
            f"<span style='color:#8891A0;'>{last:,.2f}</span>"
            f"<span style='color:{color};'>{arrow} {day*100:+.2f}%</span></div></a>"
        )
# Scroll speed scales with ticker count so the pace per-ticker stays roughly
# constant as the universe grows, rather than one fixed duration making an
# the ticker count feel rushed (or a smaller universe feel sluggish).
tape_duration = max(30, round(len(all_tickers) * 1.4))
st.markdown(
    f"<div class='kse-tape'><div class='kse-tape-track' style='animation-duration:{tape_duration}s;'>"
    f"{items_html}{items_html}</div></div>",
    unsafe_allow_html=True,
)

# --- Market Command Center: the ONE place big instrument cards are shown,
# only under the MARKETS module, so it never appears twice on screen. ---
if page == "MARKETS":
    # --- Watchlist: only shown once the user has starred something, so it
    # doesn't take up space for a new user with an empty list. Toggling a
    # star anywhere in the app (search results, this panel) uses the same
    # session_state["watchlist"] set, so it's consistent everywhere. ---
    watchlist = st.session_state.setdefault("watchlist", set())
    if watchlist:
        st.markdown('<div class="section-label">MY WATCHLIST</div>', unsafe_allow_html=True)
        wl_tickers = sorted(t for t in watchlist if t in all_tickers)
        for tk in wl_tickers:
            d = all_prices[all_prices["ticker"] == tk].sort_values("price_date")
            name = TICKER_INFO.get(tk, (tk,))[0]
            if len(d) >= 2:
                px = float(d["close"].iloc[-1])
                chg = float(d["close"].iloc[-1] / d["close"].iloc[-2] - 1)
                color = "#4FAE7A" if chg >= 0 else "#D9724F"
                arrow = "\u25B2" if chg >= 0 else "\u25BC"
                row_html = (
                    f"<div class='kse-mover-row'><a class='kse-link' href='{view_link(tk)}' target='_self' style='flex:1;'>"
                    f"<span style='color:#E7EBF0;font-weight:600;'>{tk}</span> "
                    f"<span style='color:#6B7684;'>{name}</span></a>"
                    f"<span style='color:#8891A0;'>{px:,.2f}</span>&nbsp;&nbsp;"
                    f"<span style='color:{color};'>{arrow} {chg*100:+.2f}%</span>&nbsp;&nbsp;"
                    f"<a class='kse-link' href='{watch_toggle_link(tk)}' target='_self' title='Remove from watchlist'>\u2605</a></div>"
                )
                st.markdown(row_html, unsafe_allow_html=True)

    # --- Stock Detail: the "stock becomes an object you interact with, not
    # just an input to the backtester" panel. Opened by clicking a ticker
    # anywhere (tape, movers, watchlist, sector list, search). Only an
    # explicit Backtest/+ Portfolio button here jumps elsewhere -- landing
    # here first, rather than straight into a backtest, is the point. ---
    viewing = st.session_state.get("viewing_ticker")
    if viewing and viewing in all_tickers:
        vd = all_prices[all_prices["ticker"] == viewing].sort_values("price_date").reset_index(drop=True)
        vname, vsector, vmarket = TICKER_INFO.get(viewing, (viewing, "Unknown", "?"))
        watchlist = st.session_state.setdefault("watchlist", set())
        st.markdown('<div class="section-label">STOCK DETAIL</div>', unsafe_allow_html=True)
        with st.container():
            vh1, vh2, vh3, vh4 = st.columns([3, 2, 0.6, 0.6])
            if len(vd) >= 2:
                vpx = float(vd["close"].iloc[-1])
                vchg = float(vd["close"].iloc[-1] / vd["close"].iloc[-2] - 1)
                vcolor = "green" if vchg >= 0 else "red"
                varrow = "\u25B2" if vchg >= 0 else "\u25BC"
                vh1.markdown(f"### {viewing} \u2014 {vname}")
                vh1.markdown(f"`{vsector}` \u00b7 `{vmarket}`")
                vh2.markdown(f"### {vpx:,.2f}")
                vh2.markdown(f":{vcolor}[{varrow} {vchg*100:+.2f}% today]")
            else:
                vh1.markdown(f"### {viewing} \u2014 {vname}")
                vh1.markdown(f"`{vsector}` \u00b7 `{vmarket}`")
            star_label = "\u2605" if viewing in watchlist else "\u2606"
            if vh3.button(star_label, key="detail_watch_header", help="Toggle watchlist"):
                if viewing in watchlist:
                    watchlist.discard(viewing)
                else:
                    watchlist.add(viewing)
                st.rerun()
            if vh4.button("\u2715", key="close_stock_detail", help="Close"):
                st.session_state.pop("viewing_ticker", None)
                st.rerun()

            if len(vd) >= 5:
                fig_v = go.Figure()
                fig_v.add_trace(go.Scatter(x=vd["price_date"], y=vd["close"], line=dict(color="#E8A33D", width=2), name=viewing))
                fig_v.update_layout(
                    height=340, margin=dict(t=10, b=10, l=10, r=10),
                    xaxis=dict(rangeselector=dict(
                        buttons=[
                            dict(count=1, label="1M", step="month", stepmode="backward"),
                            dict(count=3, label="3M", step="month", stepmode="backward"),
                            dict(count=6, label="6M", step="month", stepmode="backward"),
                            dict(count=1, label="YTD", step="year", stepmode="todate"),
                            dict(count=1, label="1Y", step="year", stepmode="backward"),
                            dict(count=5, label="5Y", step="year", stepmode="backward"),
                            dict(step="all", label="All"),
                        ],
                        bgcolor="#12161D", activecolor="#E8A33D", font=dict(color="#C9D1D9"),
                    ), rangeslider=dict(visible=False)),
                )
                st.plotly_chart(fig_v, width='stretch')
                # No intraday data available (daily closes only), so 1D/1W
                # range buttons from the original design aren't offered here
                # -- 1M is the shortest window this data can actually support.

                closes = vd["close"]
                perf_cols = st.columns(3)
                for label, n_days, col in [("1M", 21, perf_cols[0]), ("6M", 126, perf_cols[1]), ("1Y", 252, perf_cols[2])]:
                    if len(closes) > n_days:
                        chg = float(closes.iloc[-1] / closes.iloc[-1 - n_days] - 1)
                        col.metric(f"{label} return", fmt_pct(chg))
                    else:
                        col.metric(f"{label} return", "\u2014")

                daily_ret = closes.pct_change().dropna()
                risk_cols = st.columns(4)
                ann_vol = float(daily_ret.std() * np.sqrt(252)) if len(daily_ret) > 5 else None
                risk_cols[0].metric("Ann. volatility", f"{ann_vol*100:.1f}%" if ann_vol is not None else "\u2014")
                running_peak = closes.cummax()
                max_dd_v = float(((closes - running_peak) / running_peak).min())
                risk_cols[1].metric("Max drawdown", fmt_pct(max_dd_v))
                sharpe_v = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else None
                risk_cols[2].metric("Buy & hold Sharpe", f"{sharpe_v:.2f}" if sharpe_v is not None else "\u2014",
                                     help="Raw, no risk-free adjustment -- a simple descriptive figure, distinct from "
                                          "the risk-free-adjusted Sharpe shown in the Single-Stock Backtest results.")
                if viewing != "SPY" and "SPY" in all_tickers:
                    spy_d = all_prices[all_prices["ticker"] == "SPY"].sort_values("price_date")
                    spy_ret = spy_d["close"].pct_change().dropna()
                    ba_v = compute_beta_alpha(daily_ret, spy_ret)
                    risk_cols[3].metric("Beta vs SPY", f"{ba_v['beta']:.2f}" if ba_v else "\u2014")
                else:
                    risk_cols[3].metric("Beta vs SPY", "\u2014")

            act1, act2 = st.columns(2)
            if act1.button("\U0001F4CA BACKTEST", key="detail_backtest", width="stretch"):
                st.session_state["main_ticker"] = viewing
                st.session_state["module_select"] = "SINGLE-STOCK BACKTEST"
                st.rerun()
            portfolio_staging = st.session_state.setdefault("portfolio_staging", [])
            if act2.button("+ ADD TO PORTFOLIO", key="detail_addport", disabled=viewing in portfolio_staging, width="stretch"):
                portfolio_staging.append(viewing)
                st.session_state["module_select"] = "PORTFOLIO BACKTEST"
                st.rerun()

    # Single pass to build the per-ticker 1D-return snapshot everything below
    # is derived from (breadth, sector performance, movers, heatmap) --
    # computed once rather than four separate times.
    heat = []
    for tk in all_tickers:
        d = all_prices[all_prices["ticker"] == tk].sort_values("price_date")
        if len(d) >= 2:
            _name, sector, market = TICKER_INFO.get(tk, (tk, d["sector"].iloc[-1], "?"))
            heat.append({"ticker": tk, "ret": float(d["close"].iloc[-1] / d["close"].iloc[-2] - 1), "sector": sector, "market": market})

    if heat:
        # --- Market breadth: one glance at whether it's broadly a green or red day ---
        n_up = sum(1 for r in heat if r["ret"] > 0)
        n_down = sum(1 for r in heat if r["ret"] < 0)
        n_flat = len(heat) - n_up - n_down
        avg_ret = sum(r["ret"] for r in heat) / len(heat)
        st.markdown('<div class="section-label">MARKET BREADTH</div>', unsafe_allow_html=True)
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Advancers", n_up)
        b2.metric("Decliners", n_down)
        b3.metric("Unchanged", n_flat)
        b4.metric("Universe avg. 1D return", f"{avg_ret*100:+.2f}%")

        # --- Sector performance: genuinely new information the tape/heatmap don't show ---
        st.markdown('<div class="section-label">SECTOR PERFORMANCE</div>', unsafe_allow_html=True)
        st.caption("Click a sector to see its stocks.")
        sector_rets = {}
        for r in heat:
            sector_rets.setdefault(r["sector"], []).append(r["ret"])
        sector_avg = sorted(
            [(s, sum(v) / len(v), len(v)) for s, v in sector_rets.items()],
            key=lambda x: x[1], reverse=True,
        )
        max_abs_sector = max(abs(s[1]) for s in sector_avg) or 1.0
        sector_html = "<div style='display:flex;flex-direction:column;gap:3px;'>"
        for sector, avg, n in sector_avg:
            pct_of_max = min(abs(avg) / max_abs_sector, 1.0) * 100
            color = "#4FAE7A" if avg >= 0 else "#D9724F"
            bar_side = "left: 50%;" if avg >= 0 else "right: 50%;"
            sector_html += (
                f"<a class='kse-link' href='{sector_link(sector)}' target='_self'>"
                "<div class='kse-mover-row' style='font-family:ui-monospace,monospace;font-size:12px;gap:10px;'>"
                f"<div style='width:170px;color:#C9D1D9;font-weight:600;'>{sector} <span style='color:#5B6472;'>({n})</span></div>"
                f"<div style='width:64px;color:{color};'>{avg*100:+.2f}%</div>"
                "<div style='position:relative;flex:1;height:14px;background:#161B22;border-radius:2px;overflow:hidden;'>"
                f"<div style='position:absolute;top:0;bottom:0;{bar_side}width:{pct_of_max/2:.1f}%;background:{color};opacity:0.85;'></div>"
                "</div></div></a>"
            )
        sector_html += "</div>"
        st.markdown(sector_html, unsafe_allow_html=True)

        # --- Sector filter list: shown after clicking a sector above. ---
        filter_sector = st.session_state.get("filter_sector")
        if filter_sector:
            fs_head, fs_clear = st.columns([5, 1])
            fs_head.markdown(f"<div class='section-label' style='margin-top:14px;'>{filter_sector.upper()} \u2014 STOCKS IN THIS SECTOR</div>", unsafe_allow_html=True)
            if fs_clear.button("\u2715 Clear", key="clear_sector_filter"):
                st.session_state.pop("filter_sector", None)
                st.rerun()
            sector_ticks = sorted([r["ticker"] for r in heat if r["sector"] == filter_sector],
                                   key=lambda tk: next(r["ret"] for r in heat if r["ticker"] == tk), reverse=True)
            if not sector_ticks:
                st.caption("No tickers in this sector in the current universe.")
            for tk in sector_ticks:
                r = next(r for r in heat if r["ticker"] == tk)
                d = all_prices[all_prices["ticker"] == tk].sort_values("price_date")
                px = float(d["close"].iloc[-1]) if len(d) else float("nan")
                color = "#4FAE7A" if r["ret"] >= 0 else "#D9724F"
                st.markdown(
                    f"<a class='kse-link' href='{view_link(tk)}' target='_self'><div class='kse-mover-row'>"
                    f"<span style='color:#E7EBF0;font-weight:600;'>{tk}</span>"
                    f"<span style='color:#8891A0;'>{px:,.2f}</span>"
                    f"<span style='color:{color};'>{r['ret']*100:+.2f}%</span></div></a>",
                    unsafe_allow_html=True,
                )

        # --- Region performance: US vs. Netherlands vs. Germany vs. France vs. UK ---
        st.markdown('<div class="section-label">REGION PERFORMANCE</div>', unsafe_allow_html=True)
        region_names = {"US": "\U0001F1FA\U0001F1F8 United States", "NL": "\U0001F1F3\U0001F1F1 Netherlands",
                         "DE": "\U0001F1E9\U0001F1EA Germany", "FR": "\U0001F1EB\U0001F1F7 France",
                         "UK": "\U0001F1EC\U0001F1E7 United Kingdom"}
        region_rets = {}
        for r in heat:
            region_rets.setdefault(r["market"], []).append(r["ret"])
        region_avg = sorted(
            [(m, sum(v) / len(v), len(v)) for m, v in region_rets.items()],
            key=lambda x: x[1], reverse=True,
        )
        max_abs_region = max(abs(m[1]) for m in region_avg) or 1.0
        region_html = "<div style='display:flex;flex-direction:column;gap:3px;'>"
        for market, avg, n in region_avg:
            pct_of_max = min(abs(avg) / max_abs_region, 1.0) * 100
            color = "#4FAE7A" if avg >= 0 else "#D9724F"
            bar_side = "left: 50%;" if avg >= 0 else "right: 50%;"
            label = region_names.get(market, market)
            region_html += (
                "<div style='display:flex;align-items:center;gap:10px;font-family:ui-monospace,monospace;font-size:12px;'>"
                f"<div style='width:170px;color:#C9D1D9;font-weight:600;'>{label} <span style='color:#5B6472;'>({n})</span></div>"
                f"<div style='width:64px;color:{color};'>{avg*100:+.2f}%</div>"
                "<div style='position:relative;flex:1;height:14px;background:#161B22;border-radius:2px;overflow:hidden;'>"
                f"<div style='position:absolute;top:0;bottom:0;{bar_side}width:{pct_of_max/2:.1f}%;background:{color};opacity:0.85;'></div>"
                "</div></div>"
            )
        region_html += "</div>"
        st.markdown(region_html, unsafe_allow_html=True)

        # --- Top movers: data-driven leaderboard instead of a hand-picked card grid ---
        # Click a row to jump straight to that ticker's backtest (▲/▼ movers per the vision doc).
        st.markdown('<div class="section-label">TOP MOVERS</div>', unsafe_allow_html=True)
        gainers = sorted(heat, key=lambda r: r["ret"], reverse=True)[:5]
        losers = sorted(heat, key=lambda r: r["ret"])[:5]

        # Most Volatile: trailing 20-day annualized realized volatility, NOT
        # just today's biggest % move (that's already Top Gainers/Losers by
        # a different name) -- a genuinely different, useful cut of the
        # universe: which names have been choppy lately, regardless of
        # direction.
        vol_list = []
        for tk in all_tickers:
            d = all_prices[all_prices["ticker"] == tk].sort_values("price_date")
            if len(d) >= 21:
                rets = d["close"].pct_change().dropna().iloc[-20:]
                vol_list.append({"ticker": tk, "vol": float(rets.std() * np.sqrt(252))})
        most_volatile = sorted(vol_list, key=lambda r: r["vol"], reverse=True)[:5]

        mv1, mv2, mv3 = st.columns(3)
        with mv1:
            st.markdown("<div style='font:600 11px ui-monospace,monospace;color:#4FAE7A;letter-spacing:.6px;margin-bottom:6px;'>TOP GAINERS</div>", unsafe_allow_html=True)
            for r in gainers:
                st.markdown(
                    f"<a class='kse-link' href='{view_link(r['ticker'])}' target='_self'><div class='kse-mover-row'>"
                    f"<span style='color:#E7EBF0;font-weight:600;'>{r['ticker']}</span>"
                    f"<span style='color:#4FAE7A;'>+{r['ret']*100:.2f}%</span></div></a>",
                    unsafe_allow_html=True,
                )
        with mv2:
            st.markdown("<div style='font:600 11px ui-monospace,monospace;color:#D9724F;letter-spacing:.6px;margin-bottom:6px;'>TOP LOSERS</div>", unsafe_allow_html=True)
            for r in losers:
                st.markdown(
                    f"<a class='kse-link' href='{view_link(r['ticker'])}' target='_self'><div class='kse-mover-row'>"
                    f"<span style='color:#E7EBF0;font-weight:600;'>{r['ticker']}</span>"
                    f"<span style='color:#D9724F;'>{r['ret']*100:.2f}%</span></div></a>",
                    unsafe_allow_html=True,
                )
        with mv3:
            st.markdown("<div style='font:600 11px ui-monospace,monospace;color:#E8A33D;letter-spacing:.6px;margin-bottom:6px;'>MOST VOLATILE (20D)</div>", unsafe_allow_html=True)
            if not most_volatile:
                st.caption("Not enough history yet.")
            for r in most_volatile:
                st.markdown(
                    f"<a class='kse-link' href='{view_link(r['ticker'])}' target='_self'><div class='kse-mover-row'>"
                    f"<span style='color:#E7EBF0;font-weight:600;'>{r['ticker']}</span>"
                    f"<span style='color:#E8A33D;'>{r['vol']*100:.1f}%</span></div></a>",
                    unsafe_allow_html=True,
                )

    st.markdown('<div class="section-label">MARKET HEATMAP \u2014 CURRENT UNIVERSE</div>', unsafe_allow_html=True)
    if heat:
        heat.sort(key=lambda r: r["ret"], reverse=True)
        max_abs = max(abs(r["ret"]) for r in heat) or 1.0
        heat_html = ("<div style='display:flex;flex-direction:column;gap:3px;"
                      "max-height:420px;overflow-y:auto;padding-right:4px;'>")
        for r in heat:
            tk, ret = r["ticker"], r["ret"]
            pct_of_max = min(abs(ret) / max_abs, 1.0) * 100
            color = "#4FAE7A" if ret >= 0 else "#D9724F"
            bar_side = "left: 50%;" if ret >= 0 else f"right: 50%;"
            heat_html += (
                "<div style='display:flex;align-items:center;gap:10px;font-family:ui-monospace,monospace;font-size:12px;'>"
                f"<div style='width:70px;color:#C9D1D9;font-weight:600;'>{tk}</div>"
                f"<div style='width:64px;color:{color};'>{ret*100:+.2f}%</div>"
                "<div style='position:relative;flex:1;height:14px;background:#161B22;border-radius:2px;overflow:hidden;'>"
                f"<div style='position:absolute;top:0;bottom:0;{bar_side}width:{pct_of_max/2:.1f}%;background:{color};opacity:0.85;'></div>"
                "</div></div>"
            )
        heat_html += "</div>"
        st.markdown(heat_html, unsafe_allow_html=True)

if page == "SINGLE-STOCK BACKTEST":
    # Apply any pending ticker switch from a comparison-card click, BEFORE the
    # selectbox widget below is instantiated (Streamlit forbids changing a
    # widget's session_state after it's already been created this run).
    if "pending_ticker" in st.session_state:
        st.session_state["main_ticker"] = st.session_state.pop("pending_ticker")

    # NOTE: a second "buy & hold" ticker tape used to render here, duplicating
    # the top price tape with a different metric (total-period return vs 1D
    # change). Removed -- that data already lives in the sortable "Buy & Hold"
    # column of the compare-all-tickers table below, so showing it twice as
    # an unlabeled bar was just confusing, not additive.

    st.session_state.setdefault("main_ticker", "NVDA" if "NVDA" in all_tickers else all_tickers[0])

    strat_col, _sp = st.columns([1.3, 5.7])
    strategy_label = strat_col.selectbox(
        "Strategy", ["MA Crossover", "Mean Reversion (Z-Score)"], key="single_strategy",
        help="MA Crossover: trend-following. Mean Reversion: buy oversold dips, hold until price reverts to its own rolling average.",
    )
    strategy_type = "mean_reversion" if strategy_label.startswith("Mean Reversion") else "ma_crossover"

    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    ticker = c1.selectbox("Ticker", all_tickers, key="main_ticker")
    if strategy_type == "ma_crossover":
        ma_fast = c2.selectbox("MA Fast", MA_FAST_OPTIONS, index=1)
        ma_slow = c3.selectbox("MA Slow", [s for s in MA_SLOW_OPTIONS if s > ma_fast], index=1)
        mr_lookback, mr_entry_z = 20, 1.5  # unused at these settings, kept for a consistent function signature
    else:
        ma_fast, ma_slow = 20, 100  # unused at these settings, kept for a consistent function signature
        mr_lookback = c2.selectbox("Lookback (days)", [10, 20, 30, 50, 60], index=1,
                                    help="Window for the rolling mean/std the z-score is measured against.")
        mr_entry_z = c3.slider("Entry Z-score", 1.0, 3.0, 1.5, step=0.25,
                                help="Go long when price is this many standard deviations BELOW its rolling mean.")
    tx_cost = c4.number_input("Commission (bps)", 0, 100, 5, help="Broker/exchange fee, charged per trade.")
    capital = c5.number_input("Initial capital ($)", 100, step=100, value=10000)
    roll_window = c6.slider("Rolling window (days)", 20, 120, 60)
    benchmark_ticker = c7.selectbox("Benchmark", all_tickers, index=all_tickers.index("SPY") if "SPY" in all_tickers else 0)

    # Execution realism: commission and slippage are separate cost buckets,
    # shorting is an actual option (with a borrow cost accrued while held),
    # and market impact / vol-targeted sizing are opt-in below -- see Methodology.
    d1, d2, d3, d4 = st.columns([1, 1, 1, 2])
    slippage_bps = d1.number_input("Slippage (bps)", 0, 100, 3, help="Bid-ask spread / fill-price impact, charged per trade alongside commission.")
    allow_short = d2.checkbox("Allow short", value=False,
                               help="MA Crossover: short below the slow MA. Mean Reversion: short when overbought (z above +entry).")
    borrow_rate = d3.number_input("Borrow rate (%/yr)", 0.0, 20.0, 3.0, step=0.5, disabled=not allow_short,
                                   help="Annualized cost of borrowing shares to short, accrued daily while short.")
    position_pct = d4.slider("Position size (%)", 10, 100, 100)

    e1, e2, e3, e4 = st.columns([1, 1, 1, 1])
    enable_impact = e1.checkbox("Market impact cost", value=False,
                                 help="Adds a square-root impact cost on top of commission+slippage, sized against each "
                                      "trade's notional relative to the ticker's 20-day average dollar volume. Needs "
                                      "live volume data; has no effect if that's unavailable.")
    impact_coef = e2.slider("Impact coefficient", 0.1, 3.0, 1.0, step=0.1, disabled=not enable_impact,
                             help="Calibration constant on the impact model. Higher = trading costs more for the same "
                                  "order size relative to liquidity.")
    vol_target_on = e3.checkbox("Volatility targeting", value=False,
                                 help="Scale position size dynamically so the position's own realized volatility tracks "
                                      "a target, instead of a fixed % of capital regardless of how volatile conditions are.")
    vol_target_annual = e4.number_input("Target vol (%/yr)", 5.0, 60.0, 15.0, step=1.0, disabled=not vol_target_on)

    f1, _fsp = st.columns([1, 5])
    risk_free_pct = f1.number_input(
        "Risk-free rate (%/yr)", 0.0, 10.0, 2.0, step=0.1,
        help="Idle cash (whenever the strategy is flat, or under-invested relative to Position size / vol targeting) "
             "earns this rate instead of 0% -- matching how the Portfolio Backtest engine already treats cash. "
             "Sharpe/Sortino are computed net of this rate, so it doesn't artificially inflate risk-adjusted return.")

    prices = all_prices[all_prices["ticker"] == ticker].reset_index(drop=True)
    df = run_backtest(prices, ma_fast, ma_slow, tx_cost, position_pct, capital, roll_window,
                       slippage_bps=slippage_bps, allow_short=allow_short, borrow_rate_annual=borrow_rate,
                       strategy_type=strategy_type, mr_lookback=mr_lookback, mr_entry_z=mr_entry_z,
                       enable_impact=enable_impact, impact_coef=impact_coef,
                       vol_target_annual=vol_target_annual if vol_target_on else None, max_leverage=2.0,
                       risk_free_pct=risk_free_pct)
    trades = get_trades(df)
    trade_stats = compute_trade_stats(trades)
    turnover = turnover_stats(df, capital)

    bench_prices = all_prices[all_prices["ticker"] == benchmark_ticker].reset_index(drop=True)
    benchmark_daily_return = bench_prices["close"].pct_change().fillna(0) if ticker != benchmark_ticker else df["daily_return"]
    stats = summary_stats(df, capital, benchmark_daily_return, risk_free_pct=risk_free_pct)
    benchmark_total_return = bench_prices["close"].iloc[-1] / bench_prices["close"].iloc[0] - 1
    attribution = return_attribution(stats, benchmark_total_return)

    row1 = st.columns(5)
    row1[0].metric("Buy & Hold", f"{stats['buy_hold_return']*100:+.1f}%")
    row1[1].metric("Strategy", f"{stats['strat_return']*100:+.1f}%")
    row1[2].metric("CAGR", f"{stats['cagr']*100:+.1f}%")
    row1[3].metric("Ann. Volatility", f"{stats['vol']*100:.1f}%")
    row1[4].metric("Sharpe", f"{stats['sharpe']:.2f}")
    row2 = st.columns(4)
    row2[0].metric("Sortino", f"{stats['sortino']:.2f}")
    row2[1].metric("Calmar", f"{stats['calmar']:.2f}")
    row2[2].metric("Information Ratio", f"{stats['info_ratio']:.2f}" if stats['info_ratio'] is not None else "-")
    row2[3].metric("Max Drawdown", f"{stats['max_dd']*100:.1f}%")

    # Risk diagnostics: VaR/CVaR are simple historical estimates on the strategy return stream.
    hist_rets = df["eff_return"].iloc[1:].dropna()
    if len(hist_rets) >= 30:
        var_95 = float(hist_rets.quantile(0.05))
        tail = hist_rets[hist_rets <= var_95]
        cvar_95 = float(tail.mean()) if len(tail) else var_95
        rr1, rr2, rr3 = st.columns(3)
        rr1.metric("1D Historical VaR (95%)", f"{var_95*100:.2f}%")
        rr2.metric("1D Historical CVaR (95%)", f"{cvar_95*100:.2f}%")
        rr3.metric("Worst Day", f"{hist_rets.min()*100:.2f}%")

    # --- Compare all tickers at current settings: a sortable table, click a row to switch ---
    # Uses ticker_backtest_summary() (cached) rather than calling run_backtest()
    # directly -- with 50 tickers, re-running the full backtest for all of them
    # on every rerun (e.g. just clicking a row) would get noticeably slow.
    st.markdown("<div style='font-size:12px; color:#6B7684; margin:14px 0 6px;'>COMPARE ALL TICKERS AT CURRENT SETTINGS</div>", unsafe_allow_html=True)
    compare_rows = []
    for tk in all_tickers:
        tk_prices = all_prices[all_prices["ticker"] == tk].reset_index(drop=True)
        min_len = (mr_lookback + 5) if strategy_type == "mean_reversion" else (ma_slow + 5)
        if len(tk_prices) < min_len:
            continue
        tk_bh, tk_strat, tk_vol, tk_sharpe, tk_ntrades = ticker_backtest_summary(
            tk_prices, ma_fast, ma_slow, tx_cost, position_pct, capital, roll_window,
            slippage_bps=slippage_bps, allow_short=allow_short, borrow_rate_annual=borrow_rate,
            strategy_type=strategy_type, mr_lookback=mr_lookback, mr_entry_z=mr_entry_z,
            enable_impact=enable_impact, impact_coef=impact_coef,
            vol_target_annual=vol_target_annual if vol_target_on else None, max_leverage=2.0,
            risk_free_pct=risk_free_pct)
        name, sector, market = TICKER_INFO.get(tk, (tk, tk_prices["sector"].iloc[0], "?"))
        compare_rows.append({
            "Ticker": tk, "Name": name, "Market": market, "Sector": sector,
            "Buy & Hold": tk_bh, "Strategy": tk_strat, "Vol": tk_vol, "Sharpe": tk_sharpe, "Trades": tk_ntrades,
        })
    compare_df = pd.DataFrame(compare_rows).sort_values("Strategy", ascending=False).reset_index(drop=True)
    active_row_idx = compare_df.index[compare_df["Ticker"] == ticker].tolist()

    selection = st.dataframe(
        compare_df,
        width="stretch",
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "Buy & Hold": st.column_config.NumberColumn(format="%+.1f%%"),
            "Strategy": st.column_config.NumberColumn(format="%+.1f%%"),
            "Vol": st.column_config.NumberColumn(format="%.0f%%"),
            "Sharpe": st.column_config.NumberColumn(format="%.2f"),
        },
        key="compare_table",
    )
    st.caption("Click a row to switch the ticker analyzed above and in every tab below.")

    # Research-terminal summary: highlights the current candidate before the detailed tabs.
    if not compare_df.empty:
        leader = compare_df.iloc[0]
        current = compare_df[compare_df["Ticker"] == ticker].iloc[0]
        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("Best strategy return", f"{leader['Ticker']}  {leader['Strategy']*100:+.1f}%")
        rc2.metric("Current strategy", f"{current['Strategy']*100:+.1f}%")
        rc3.metric("Current Sharpe", f"{current['Sharpe']:.2f}")
    if selection and selection.get("selection", {}).get("rows"):
        picked = compare_df.iloc[selection["selection"]["rows"][0]]["Ticker"]
        if picked != ticker:
            st.session_state["pending_ticker"] = picked
            st.rerun()

    sub1, sub2, sub3, sub4, sub5, sub6, sub7 = st.tabs(
        ["Overview", "Trade Log", "Trade Statistics", "Parameter Heatmap", "Monte Carlo", "Out-of-Sample", "Methodology"])

    with sub1:
        strat_label = f"MA {ma_fast}/{ma_slow}" if strategy_type == "ma_crossover" else f"Mean Reversion, z={mr_entry_z}, lookback={mr_lookback}d"
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["price_date"], y=df["strat_equity"], name="Strategy", line=dict(color="#E8A33D", width=2)))
        fig.add_trace(go.Scatter(x=df["price_date"], y=df["buyhold_equity"], name="Buy & hold", line=dict(color="#8891A0", width=1.5, dash="dash")))
        long_entries = df[df["transition"] & (df["prev_signal"] == 1)]
        short_entries = df[df["transition"] & (df["prev_signal"] == -1)]
        flat_exits = df[df["transition"] & (df["prev_signal"] == 0)]
        fig.add_trace(go.Scatter(x=long_entries["price_date"], y=long_entries["strat_equity"], mode="markers", name="Long entry",
                                  marker=dict(color="#4FAE7A", symbol="triangle-up", size=10)))
        if allow_short:
            fig.add_trace(go.Scatter(x=short_entries["price_date"], y=short_entries["strat_equity"], mode="markers", name="Short entry",
                                      marker=dict(color="#D9724F", symbol="triangle-down", size=10)))
        fig.add_trace(go.Scatter(x=flat_exits["price_date"], y=flat_exits["strat_equity"], mode="markers", name="Exit to flat",
                                  marker=dict(color="#8891A0", symbol="circle", size=8)))
        fig.update_layout(height=420, title=f"Equity curve: {ticker} ({strat_label})",
                           hovermode="x unified", margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig, width='stretch')

        cc1, cc2 = st.columns(2)
        with cc1:
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(x=df["price_date"], y=df["drawdown_strat"] * 100, fill="tozeroy",
                                         line=dict(color="#D9724F"), name="Drawdown"))
            fig_dd.update_layout(height=300, title="Drawdown (%)", margin=dict(t=40, b=10, l=10, r=10))
            st.plotly_chart(fig_dd, width='stretch')
        with cc2:
            fig_rs = go.Figure()
            fig_rs.add_trace(go.Scatter(x=df["price_date"], y=df["roll_sharpe"], line=dict(color="#5B8DD9"), name="Rolling Sharpe"))
            fig_rs.update_layout(height=300, title=f"Rolling Sharpe ({roll_window}d)", margin=dict(t=40, b=10, l=10, r=10))
            st.plotly_chart(fig_rs, width='stretch')

        st.markdown("##### Beta & alpha vs. benchmark")
        st.caption(
            f"Single-factor regression of daily strategy returns on {benchmark_ticker}'s daily returns: "
            "is this generating genuine excess return, or is it just a correlated bet on the benchmark?"
        )
        ba = compute_beta_alpha(df["eff_return"], benchmark_daily_return)
        if ba is None:
            st.info("Not enough overlapping history with the benchmark to estimate beta/alpha reliably.")
        else:
            ba_cols = st.columns(4)
            ba_cols[0].metric(f"Beta vs {benchmark_ticker}", f"{ba['beta']:.2f}")
            ba_cols[1].metric("Annualized alpha", fmt_pct(ba['alpha_annual']))
            ba_cols[2].metric("R\u00b2", f"{ba['r2']:.2f}")
            sig = "Yes" if (ba["t_alpha"] is not None and abs(ba["t_alpha"]) >= 2) else ("N/A" if ba["t_alpha"] is None else "No")
            ba_cols[3].metric("Alpha significant? (|t|\u22652)", sig)
            st.caption(
                f"Beta of {ba['beta']:.2f} means the strategy's daily moves are on average {ba['beta']:.2f}\u00d7 "
                f"{benchmark_ticker}'s. R\u00b2 of {ba['r2']:.2f} means {ba['r2']*100:.0f}% of the strategy's daily "
                "variance is explained by that benchmark exposure alone -- the rest, including the alpha figure "
                "above, is what's NOT explained by simply being correlated with the benchmark. The t-stat uses "
                "Newey-West (HAC) standard errors, not plain OLS -- if the strategy's returns are autocorrelated "
                "(plausible for anything holding positions across multiple days), plain OLS would understate the "
                "standard error and overstate how significant alpha looks."
            )

        csv_cols = ["price_date", "close", "in_market", "strat_equity", "buyhold_equity", "drawdown_strat"]
        if strategy_type == "ma_crossover":
            csv_cols[2:2] = ["ma_fast", "ma_slow"]
        else:
            csv_cols[2:2] = ["z_score"]
        st.download_button(
            "Download equity curve (CSV)",
            df[csv_cols].to_csv(index=False),
            file_name=f"{ticker}_equity_curve_{strategy_type}.csv",
            mime="text/csv",
        )

    with sub2:
        if trades:
            trade_df = pd.DataFrame(trades)[["entry_date", "exit_date", "side", "days", "ret"]]
            trade_df.columns = ["Entry date", "Exit date", "Side", "Days held", "Return"]
            trade_df["Return"] = (trade_df["Return"] * 100).map("{:+.2f}%".format)
            st.dataframe(trade_df.iloc[::-1], width='stretch', hide_index=True)
            st.download_button(
                "Download trade log (CSV)",
                pd.DataFrame(trades).to_csv(index=False),
                file_name=f"{ticker}_trade_log_{strategy_type}.csv",
                mime="text/csv",
            )
        else:
            st.info("No trades at these settings.")

    with sub3:
        pf = f"{trade_stats['profit_factor']:.2f}" if np.isfinite(trade_stats['profit_factor']) else "inf"
        stats_rows = [
            ("Number of trades", f"{trade_stats['count']}"),
            ("Average holding period", f"{trade_stats['avg_days']:.0f} days"),
            ("Longest win streak", f"{trade_stats['longest_win_streak']}"),
            ("Longest loss streak", f"{trade_stats['longest_loss_streak']}"),
            ("Average trade return", f"{trade_stats['avg_trade_return']*100:+.2f}%"),
            ("Win rate", f"{trade_stats['win_rate']*100:.0f}%"),
            ("Profit factor", pf),
            ("Average gain", f"{trade_stats['avg_gain']*100:+.2f}%"),
            ("Average loss", f"{trade_stats['avg_loss']*100:+.2f}%"),
            ("Expectancy per trade", f"{trade_stats['expectancy']*100:+.2f}%"),
        ]
        st.table(pd.DataFrame(stats_rows, columns=["Metric", "Value"]).set_index("Metric"))

        st.markdown("##### Autocorrelation-adjusted Sharpe (Lo, 2002)")
        st.caption(
            "The naive Sharpe elsewhere in this app annualizes by multiplying by \u221a252, which assumes daily "
            "returns are independent. A strategy that holds positions for multiple days in a row has "
            "autocorrelated returns, which makes naive annualization overstate the true Sharpe. This corrects for it."
        )
        lo_sr = lo_adjusted_sharpe(df["eff_return"].iloc[1:] - (risk_free_pct / 100 / 252 if risk_free_pct else 0.0))
        if lo_sr is None:
            st.info("Not enough data at these settings to compute a reliable autocorrelation adjustment.")
        else:
            lo_cols = st.columns(3)
            lo_cols[0].metric("Naive Sharpe (\u221a252)", f"{lo_sr['naive']:.2f}")
            lo_cols[1].metric("Lo-adjusted Sharpe", f"{lo_sr['adjusted']:.2f}")
            overstatement = (lo_sr['naive'] - lo_sr['adjusted']) / lo_sr['naive'] * 100 if lo_sr['naive'] != 0 else 0
            lo_cols[2].metric("Overstatement", f"{overstatement:+.0f}%")
            st.caption(f"Lag-1 autocorrelation of daily returns: {lo_sr['rho1']:+.3f}. "
                       f"{'Positive autocorrelation means the naive Sharpe above is inflated.' if lo_sr['rho1'] > 0.02 else 'Autocorrelation is low here, so the naive and adjusted Sharpe are close.'}")

        st.markdown("##### Turnover & transaction costs")
        st.caption("How much trading this strategy actually did, and what it cost (commission + slippage combined; "
                    "borrow cost and market impact, if enabled, are broken out separately below).")
        turn_cols = st.columns(4)
        turn_cols[0].metric("Round-trip signal flips", turnover["n_transitions"])
        turn_cols[1].metric("Flips per year", f"{turnover['transitions_per_year']:.1f}")
        turn_cols[2].metric("Commission + slippage paid", f"${turnover['total_cost_dollars']:,.0f}")
        turn_cols[3].metric("Cost as % of capital", f"{turnover['total_cost_pct']*100:.2f}%")
        if allow_short:
            bc1, bc2 = st.columns(2)
            bc1.metric("Borrow cost paid (shorts)", f"${turnover['total_borrow_dollars']:,.0f}")
            bc2.metric("Borrow cost as % of capital", f"{turnover['total_borrow_pct']*100:.2f}%")
        if enable_impact:
            ic1, ic2 = st.columns(2)
            ic1.metric("Market impact cost paid", f"${turnover['total_impact_dollars']:,.0f}")
            ic2.metric("Impact cost as % of capital", f"{turnover['total_impact_pct']*100:.2f}%")

        st.markdown("##### Annualized return attribution")
        st.caption(
            f"Simple decomposition vs. holding {benchmark_ticker}: how much of the total return came from picking "
            f"{ticker} over the benchmark, versus how much came from the MA-crossover timing itself. Additive "
            f"approximation, not geometrically exact."
        )
        attr_rows = [
            (f"Benchmark return ({benchmark_ticker})", fmt_pct(attribution["benchmark_return"])),
            (f"Selection effect ({ticker} vs. benchmark)", fmt_pct(attribution["selection_effect"])),
            ("Timing effect (strategy vs. buy & hold)", fmt_pct(attribution["timing_effect"])),
            ("Total strategy return", fmt_pct(attribution["total"])),
        ]
        st.table(pd.DataFrame(attr_rows, columns=["Component", "Contribution"]).set_index("Component"))

    with sub4:
        if strategy_type != "ma_crossover":
            st.info(
                "The parameter grid search below sweeps MA Fast \u00d7 MA Slow combinations, so it's specific to "
                "the MA Crossover strategy. Switch the Strategy selector above to MA Crossover to use it. "
                "(A Mean Reversion version of this \u2014 sweeping Lookback \u00d7 Entry Z-score \u2014 would be the "
                "natural next addition here.)"
            )
        else:
            st.caption("Sharpe ratio across every fast/slow MA combination, other settings held fixed.")
            grid = np.full((len(MA_FAST_OPTIONS), len(MA_SLOW_OPTIONS)), np.nan)
            trial_daily_sharpes = []
            best_annualized, best_returns, best_combo_grid = -np.inf, None, None
            for i, fast in enumerate(MA_FAST_OPTIONS):
                for j, slow in enumerate(MA_SLOW_OPTIONS):
                    if fast >= slow:
                        continue
                    r = run_backtest(prices, fast, slow, tx_cost, position_pct, capital, roll_window,
                                      slippage_bps=slippage_bps, allow_short=allow_short, borrow_rate_annual=borrow_rate,
                                      enable_impact=enable_impact, impact_coef=impact_coef,
                                      vol_target_annual=vol_target_annual if vol_target_on else None, max_leverage=2.0,
                                      risk_free_pct=risk_free_pct)
                    eff = r["eff_return"].iloc[1:]
                    daily_rf = (risk_free_pct / 100 / 252) if risk_free_pct else 0.0
                    excess = eff - daily_rf
                    sd = eff.std()
                    daily_sharpe = (excess.mean() / sd) if sd > 0 else 0.0
                    trial_daily_sharpes.append(daily_sharpe)
                    annualized = daily_sharpe * np.sqrt(252)
                    grid[i, j] = annualized
                    if annualized > best_annualized:
                        best_annualized, best_returns, best_combo_grid = annualized, excess, (fast, slow)
            fig_heat = go.Figure(data=go.Heatmap(
                z=grid, x=[f"{s}d" for s in MA_SLOW_OPTIONS], y=[f"{f}d" for f in MA_FAST_OPTIONS],
                colorscale=[[0, "#D9724F"], [0.5, "#161B22"], [1, "#4FAE7A"]],
                text=np.round(grid, 2), texttemplate="%{text}", hoverongaps=False,
            ))
            fig_heat.update_layout(height=380, xaxis_title="MA Slow", yaxis_title="MA Fast", margin=dict(t=20, b=10, l=10, r=10))
            st.plotly_chart(fig_heat, width='stretch')

            st.markdown("##### Deflated Sharpe Ratio \u2014 correcting for the parameter search itself")
            st.caption(
                f"This grid just tried {len(trial_daily_sharpes)} parameter combinations and is about to show you "
                "whichever looked best. That's a multiple-comparisons problem: some of that \"best\" Sharpe is "
                "just the best of several noisy draws, not necessarily real skill. The Deflated Sharpe Ratio "
                "(Bailey & L\u00f3pez de Prado, 2014) corrects for it."
            )
            dsr = deflated_sharpe_ratio(trial_daily_sharpes, best_returns) if best_returns is not None else None
            if dsr is None:
                st.info("Not enough data at these settings to compute a reliable Deflated Sharpe Ratio.")
            else:
                dsr_cols = st.columns(4)
                dsr_cols[0].metric(f"Best combo (MA {best_combo_grid[0]}/{best_combo_grid[1]})", f"{dsr['sr_hat_annualized']:.2f} Sharpe")
                dsr_cols[1].metric(f"Expected best-of-{dsr['n_trials']} by chance alone", f"{dsr['sr0_annualized']:.2f} Sharpe")
                dsr_cols[2].metric("Deflated Sharpe Ratio", f"{dsr['dsr']*100:.1f}%")
                dsr_cols[3].metric("Statistically significant?", "Yes" if dsr["dsr"] >= 0.95 else "No (< 95%)")
                st.caption(
                    "Deflated Sharpe Ratio is the probability the strategy's TRUE Sharpe is actually above zero, "
                    "after accounting for having tried multiple parameter combinations and for this return series' "
                    "own sample size, skew, and fat tails. Convention treats \u2265 95% as a genuinely significant "
                    "result; below that, what looks like an edge in the heatmap above may just be noise that a "
                    f"12-cell search was always going to find somewhere."
                )

    with sub5:
        st.caption("10,000 simulations, built by resampling the actual sequence of trades with replacement.")
        if st.button("Run Monte Carlo simulation"):
            benchmark_final = capital * (1 + benchmark_total_return)
            mc = run_monte_carlo_trades(trades, capital, n_sims=10000, benchmark_final=benchmark_final)
            if mc is None:
                st.warning("Not enough trades at these settings to simulate.")
            else:
                fig_mc = go.Figure()
                fig_mc.add_trace(go.Scatter(x=mc["x"] + mc["x"][::-1], y=mc["bands"][95] + mc["bands"][5][::-1],
                                             fill="toself", fillcolor="rgba(232,163,61,0.12)", line=dict(width=0), showlegend=False))
                fig_mc.add_trace(go.Scatter(x=mc["x"] + mc["x"][::-1], y=mc["bands"][75] + mc["bands"][25][::-1],
                                             fill="toself", fillcolor="rgba(232,163,61,0.25)", line=dict(width=0), showlegend=False))
                fig_mc.add_trace(go.Scatter(x=mc["x"], y=mc["bands"][50], line=dict(color="#E8A33D", dash="dash"), name="Median simulated"))
                fig_mc.add_trace(go.Scatter(x=mc["x"], y=mc["actual_path"], line=dict(color="#E8ECF1", width=2), name="Actual path"))
                fig_mc.update_layout(height=380, xaxis_title="Trade number", margin=dict(t=20, b=10, l=10, r=10))
                st.plotly_chart(fig_mc, width='stretch')

                st.markdown("##### Outcome probabilities")
                p1, p2, p3 = st.columns(3)
                p1.metric("Probability of a positive return", f"{mc['pos_prob']*100:.0f}%")
                if "beat_bench_prob" in mc:
                    p2.metric(f"Probability of beating {benchmark_ticker}", f"{mc['beat_bench_prob']*100:.0f}%")
                else:
                    p2.metric("Probability of beating benchmark", "\u2014")
                p3.metric("Probability of a >20% drawdown", f"{mc['dd20_prob']*100:.0f}%")

                st.markdown("##### Distribution of simulated outcomes")
                final_rets = mc["finals"] / capital - 1
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(x=final_rets * 100, nbinsx=60, marker=dict(color="#E8A33D", opacity=0.75)))
                fig_hist.add_vline(x=0, line=dict(color="#8891A0", dash="dash"))
                fig_hist.add_vline(x=(mc["median_final"] / capital - 1) * 100, line=dict(color="#E8ECF1", width=2),
                                    annotation_text="Median", annotation_position="top")
                fig_hist.update_layout(height=300, xaxis_title="Simulated total return (%)", yaxis_title="Simulations",
                                        margin=dict(t=20, b=10, l=10, r=10), bargap=0.02)
                st.plotly_chart(fig_hist, width='stretch')

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("5th percentile", fmt_pct(mc['p5_final'] / capital - 1), help=f"${mc['p5_final']:,.0f}")
                m2.metric("Median outcome", fmt_pct(mc['median_final'] / capital - 1), help=f"${mc['median_final']:,.0f}")
                m3.metric("95th percentile", fmt_pct(mc['p95_final'] / capital - 1), help=f"${mc['p95_final']:,.0f}")
                m4.metric("Probability of a loss", f"{mc['loss_prob']*100:.0f}%")

        st.markdown("##### Block-bootstrap confidence intervals: Sharpe & CAGR")
        st.caption(
            "500 resamples, built by shuffling BLOCKS of 20 consecutive daily returns (not individual days) so the "
            "short-term autocorrelation this strategy's returns have -- from holding a position across multiple "
            "days -- carries through into each resampled path. A day-by-day bootstrap would implicitly (and "
            "incorrectly) assume returns are independent; this doesn't. The point estimates shown in Overview and "
            "Trade Statistics are single numbers -- these ranges are the honest uncertainty around them."
        )
        if st.button("Run block-bootstrap CI"):
            boot = bootstrap_sharpe_cagr_ci(df["eff_return"].iloc[1:], n_boot=500, block_size=20, ci=0.90, risk_free_pct=risk_free_pct)
            if boot is None:
                st.warning("Not enough data at these settings to bootstrap a reliable confidence interval.")
            else:
                bt_cols = st.columns(2)
                bt_cols[0].metric("Sharpe (median of resamples)", f"{boot['sharpe_median']:.2f}",
                                   help=f"90% CI: [{boot['sharpe_lo']:.2f}, {boot['sharpe_hi']:.2f}]")
                bt_cols[1].metric("CAGR (median of resamples)", fmt_pct(boot['cagr_median']),
                                   help=f"90% CI: [{fmt_pct(boot['cagr_lo'])}, {fmt_pct(boot['cagr_hi'])}]")
                st.caption(
                    f"90% confidence interval \u2014 Sharpe: [{boot['sharpe_lo']:.2f}, {boot['sharpe_hi']:.2f}]. "
                    f"CAGR: [{fmt_pct(boot['cagr_lo'])}, {fmt_pct(boot['cagr_hi'])}]. "
                    f"({boot['n_boot']} resamples.) If this range comfortably excludes zero (for Sharpe) or the "
                    "risk-free rate (for CAGR), that's a meaningfully stronger claim than a single point estimate."
                )

    with sub6:
        if strategy_type != "ma_crossover":
            st.info(
                "Walk-forward validation below re-searches MA Fast \u00d7 MA Slow combinations in each fold, so "
                "it's specific to the MA Crossover strategy. Switch the Strategy selector above to MA Crossover "
                "to use it."
            )
        else:
            st.caption(
                "Walk-forward validation: history is split into several rolling folds. In each fold, MA parameters "
                "are chosen using ONLY the training window up to that point (an expanding window), then tested on "
                "the untouched period right after it \u2014 and the parameter search re-runs fresh in every fold. "
                "This is the standard real-quant validation method, and it's meaningfully more robust than checking "
                "a single train/test split: one split can get lucky or unlucky by chance; several folds show "
                "whether an edge actually holds up consistently."
            )
            n_folds = st.slider("Number of walk-forward folds", 2, 6, 4, key="wf_folds")
            n_rows = len(prices)
            seg_size = n_rows // (n_folds + 1)
            min_needed = max(MA_SLOW_OPTIONS) + 5

            if seg_size < min_needed:
                st.warning(f"Not enough history for {n_folds} folds at these MA settings. Try fewer folds.")
            else:
                fold_rows = []
                oos_returns_parts, oos_dates_parts = [], []
                for k in range(1, n_folds + 1):
                    train_end = seg_size * k
                    test_end = seg_size * (k + 1) if k < n_folds else n_rows
                    train_seg = prices.iloc[:train_end].reset_index(drop=True)
                    test_seg = prices.iloc[train_end:test_end].reset_index(drop=True)
                    if len(train_seg) < min_needed or len(test_seg) < min_needed:
                        continue

                    best_sharpe_k, best_combo_k = -np.inf, (ma_fast, ma_slow)
                    for fast in MA_FAST_OPTIONS:
                        for slow in MA_SLOW_OPTIONS:
                            if fast >= slow:
                                continue
                            r = run_backtest(train_seg, fast, slow, tx_cost, position_pct, capital, roll_window,
                                              slippage_bps=slippage_bps, allow_short=allow_short, borrow_rate_annual=borrow_rate,
                                              enable_impact=enable_impact, impact_coef=impact_coef,
                                              vol_target_annual=vol_target_annual if vol_target_on else None, max_leverage=2.0,
                                              risk_free_pct=risk_free_pct)
                            eff_ = r["eff_return"].iloc[1:]
                            sd_ = eff_.std()
                            sharpe_ = (eff_.mean() / sd_ * np.sqrt(252)) if sd_ > 0 else 0
                            if sharpe_ > best_sharpe_k:
                                best_sharpe_k, best_combo_k = sharpe_, (fast, slow)

                    # Warm-up fix: a test segment sliced fresh at train_end has
                    # no valid MA for its first best_combo_k[1] days (the
                    # rolling window has nothing before the slice to look
                    # back on), so those days were previously forced flat --
                    # silently wasting each fold's own early days instead of
                    # carrying real warm-up context forward from training.
                    # Extending the slice backward by the warm-up window means
                    # the indicator is already valid AT the true fold
                    # boundary; the prefix itself is discarded before scoring.
                    warmup_needed = best_combo_k[1]
                    ext_start = max(0, train_end - warmup_needed)
                    test_seg_ext = prices.iloc[ext_start:test_end].reset_index(drop=True)
                    n_prefix = train_end - ext_start

                    test_r_full = run_backtest(test_seg_ext, best_combo_k[0], best_combo_k[1], tx_cost, position_pct, capital, roll_window,
                                                slippage_bps=slippage_bps, allow_short=allow_short, borrow_rate_annual=borrow_rate,
                                                enable_impact=enable_impact, impact_coef=impact_coef,
                                                vol_target_annual=vol_target_annual if vol_target_on else None, max_leverage=2.0,
                                                risk_free_pct=risk_free_pct)
                    # eff_return at row n_prefix is a legitimately-computed
                    # return against the prior (warm-up) day, not the usual
                    # "first row of a slice" pct_change artifact -- so unlike
                    # elsewhere in this app, no .iloc[1:] offset is needed here.
                    test_eff = test_r_full["eff_return"].iloc[n_prefix:].reset_index(drop=True)
                    test_dates = test_r_full["price_date"].iloc[n_prefix:].reset_index(drop=True)
                    test_sd = test_eff.std()
                    test_sharpe_k = (test_eff.mean() / test_sd * np.sqrt(252)) if test_sd > 0 else 0
                    test_equity = capital * (1 + test_eff).cumprod()
                    test_ret_k = float(test_equity.iloc[-1] / capital - 1) if len(test_equity) else 0.0

                    fold_rows.append({
                        "Fold": k,
                        "Train period": f"{train_seg['price_date'].iloc[0].date()} \u2192 {train_seg['price_date'].iloc[-1].date()}",
                        "Test period": f"{test_seg['price_date'].iloc[0].date()} \u2192 {test_seg['price_date'].iloc[-1].date()}",
                        "Best params": f"MA {best_combo_k[0]}/{best_combo_k[1]}",
                        "Train Sharpe": round(best_sharpe_k, 2),
                        "Test Sharpe": round(test_sharpe_k, 2),
                        "Test return": test_ret_k,
                    })
                    oos_returns_parts.append(test_eff)
                    oos_dates_parts.append(test_dates)

                if not fold_rows:
                    st.warning("No fold had enough data at these MA settings. Try fewer folds.")
                else:
                    fold_df = pd.DataFrame(fold_rows)
                    display_df = fold_df.copy()
                    display_df["Test return"] = display_df["Test return"].map(fmt_pct)
                    st.dataframe(display_df, width='stretch', hide_index=True)

                    pos_folds = sum(1 for r in fold_rows if r["Test Sharpe"] > 0)
                    avg_test_sharpe = float(np.mean([r["Test Sharpe"] for r in fold_rows]))
                    avg_train_sharpe = float(np.mean([r["Train Sharpe"] for r in fold_rows]))
                    wf_cols = st.columns(3)
                    wf_cols[0].metric("Folds with positive OOS Sharpe", f"{pos_folds}/{len(fold_rows)}")
                    wf_cols[1].metric("Average OOS Sharpe across folds", f"{avg_test_sharpe:.2f}")
                    wf_cols[2].metric("Avg. Sharpe degradation (train\u2192test)", f"{avg_train_sharpe - avg_test_sharpe:+.2f}")

                    if avg_test_sharpe < avg_train_sharpe * 0.5 or pos_folds < len(fold_rows) / 2:
                        st.warning(
                            "The edge doesn't hold up consistently out-of-sample across folds \u2014 a real sign of "
                            "overfitting rather than a durable strategy, not just one unlucky split."
                        )
                    else:
                        st.info("The edge holds up reasonably consistently across out-of-sample folds.")

                    # Stitched OOS equity curve: chain every fold's test-period
                    # returns together with continuous compounding, so this is
                    # what you'd have actually experienced re-optimizing on a
                    # rolling basis -- not each fold artificially restarting at
                    # the same capital like the panel above's table implies.
                    all_oos_returns = pd.concat(oos_returns_parts, ignore_index=True)
                    all_oos_dates = pd.concat(oos_dates_parts, ignore_index=True)
                    stitched_equity = capital * (1 + all_oos_returns).cumprod()
                    fig_wf = go.Figure()
                    fig_wf.add_trace(go.Scatter(x=all_oos_dates, y=stitched_equity, line=dict(color="#E8A33D", width=2),
                                                 name="Walk-forward OOS equity"))
                    fig_wf.update_layout(height=380, title="Stitched out-of-sample equity curve (each fold re-optimized on prior data only)",
                                          margin=dict(t=40, b=10, l=10, r=10))
                    st.plotly_chart(fig_wf, width='stretch')

    with sub7:
        if strategy_type == "mean_reversion":
            strategy_desc = (
                f"Go long {ticker} (100% of the position size selected above) when its price falls to "
                f"{mr_entry_z} standard deviations below its own {mr_lookback}-day rolling mean (a z-score "
                f"threshold), and hold until price reverts back up to that rolling mean (z-score \u2265 0) -- "
                f"not just until it exits the extreme band"
            )
            strategy_desc += (
                f"; mirror the same logic on the short side when price is {mr_entry_z} standard deviations "
                f"above the mean." if allow_short else "; otherwise hold cash."
            )
        else:
            strategy_desc = (
                f"Go long {ticker} (100% of the position size selected above) whenever its {ma_fast}-day moving "
                f"average is above its {ma_slow}-day moving average"
            )
            strategy_desc += (
                f"; go short otherwise." if allow_short else "; otherwise hold cash."
            )
        st.markdown(f"""
##### What the strategy does
{strategy_desc} The signal is computed at the close of each day, but the
strategy only acts on it starting the *next* trading day, so no decision ever uses information that wasn't yet
public at the time.

##### Transaction costs
Commission ({tx_cost} bps) and slippage ({slippage_bps} bps) are charged separately on the notional value traded
each time the position changes, together totaling {tx_cost + slippage_bps} bps per unit of notional traded.
A direct long\u2192short (or short\u2192long) flip trades twice the notional of a flat\u2192long or long\u2192flat
move -- you're closing one side and opening the other in the same step -- so it's charged double. This is still
**not a market-impact model**: it doesn't account for order size relative to trading volume, which would make
real-world costs higher for less liquid names or larger position sizes than tested here.
{"" if not allow_short else f"""
##### Short selling & borrow cost
With Allow short enabled, the strategy holds a short position (rather than cash) below the slow MA, and accrues
a borrow cost of {borrow_rate}%/yr, applied daily while short. This approximates the real cost of borrowing
shares to short; it does not model margin requirements or the risk of the position being recalled."""}

##### Cash and risk-free rate
Capital not invested in the position (all of it while flat; the leftover remainder if Position size < 100% or
vol-targeting has scaled exposure down) earns the risk-free rate you set above -- this now matches how the
Portfolio Backtest engine already treated idle cash, rather than the single-stock engine assuming 0% regardless
of the setting. Sharpe and Sortino are computed net of that risk-free rate (excess return / volatility), so
letting cash earn something doesn't mechanically inflate risk-adjusted performance -- it would if the numerator
weren't also corrected. Sharpe, Sortino, and Information Ratio all annualize using 252 trading days per year.

##### Benchmark
Information Ratio and the return attribution above compare this strategy to simply holding **{benchmark_ticker}**
over the same period. The benchmark is selectable in the controls above; it is not hard-coded to SPY.

##### Out-of-sample testing (walk-forward)
The "Out-of-Sample" tab runs walk-forward validation: history is split into several rolling folds, and in each
one the MA parameters are re-searched using only that fold's training window, then tested on the untouched
period right after it. This is the honest way to check whether a parameter search found a real, durable pattern
or just fit noise in one dataset -- and using multiple folds rather than a single split guards against the
result being an artifact of where that one split happened to fall.

##### Deflated Sharpe Ratio
The "Parameter Heatmap" tab also reports a Deflated Sharpe Ratio (Bailey & L\u00f3pez de Prado, 2014): scanning
several MA combinations and reporting whichever had the best Sharpe is a multiple-comparisons problem, and this
corrects for it by estimating how good the best of that many random, skill-less trials would be expected to look
by chance alone, then reporting the probability the observed best Sharpe is genuinely above that noise floor.

##### Data source & provenance
Daily closing prices for the {len(TICKER_INFO)}-ticker universe (large-caps across the US, Netherlands, Germany,
France, and the UK),
pulled live via the `yfinance` Python package (Yahoo Finance) -- auto-refreshed every 5 minutes, or immediately
via the Refresh button. There is no static CSV and no offline fallback: if Yahoo Finance is unreachable, the app
shows an error rather than silently serving old numbers. Current data covers
{all_prices['price_date'].min().date()} to {all_prices['price_date'].max().date()}.
Prices are as returned by Yahoo Finance's default adjustment (split-adjusted; dividend adjustment depends on
Yahoo's own methodology and was not independently verified here). Note: Yahoo's adjusted-close series can be
*retroactively revised* when a new dividend or split is applied, since adjustment factors are recalculated back
through history -- so a backtest run today over 2015-2020 is not guaranteed to use the exact same historical
values as the same backtest run a year from now. This app doesn't snapshot data by fetch date, so it is not a
true point-in-time database; treat it as a research/exploration tool, not a system of record for historical prices.

##### Execution cost model
Commission and slippage are flat per-trade haircuts (in bps), charged whenever the position changes -- this part
doesn't scale with order size or liquidity. Optionally, a market impact model can be enabled on top of those:
a square-root impact cost (`impact_coef \u00d7 asset's 20-day volatility \u00d7 \u221a(order notional / 20-day
average dollar volume)`), so trading a large position in a thin name costs more than the same size in a heavily
traded one. This uses live trading volume (pulled from Yahoo Finance) and is still an approximation -- the
square-root law is a standard industry heuristic, not a full limit-order-book simulation, and it sizes each
order against INITIAL capital rather than compounding equity to avoid a circular dependency, so impact cost
doesn't scale up automatically if the strategy grows well beyond its starting capital.

##### Volatility-targeted position sizing
Optionally, position size can scale dynamically with realized volatility instead of staying fixed: leverage =
target annual vol \u00f7 the asset's own trailing 20-day realized vol, capped at a maximum, computed using only
data through the prior close (no lookahead, same discipline as the trading signal itself). This is standard risk
management -- a fixed position size delivers very different real risk depending on how volatile conditions are;
vol targeting normalizes for that.

##### Short selling & borrow cost
With "Allow short" enabled, the strategy goes short (rather than flat) below the slow MA, and accrues a daily
borrow cost (annualized rate you set, applied while short) -- the standard real-world cost of holding a short.
This is a simplified approximation: it doesn't model margin requirements, the risk of the position being
recalled, or borrow rates that vary by ticker and over time (hard-to-borrow names can cost far more than the
flat rate here).

##### Autocorrelation-adjusted Sharpe (Lo, 2002)
The Sharpe ratios shown elsewhere in this app (Overview, Trade Statistics headline numbers, Deflated Sharpe
Ratio) annualize with the standard \u221a252 multiplier, which assumes daily returns are independent. A strategy
that holds a position across multiple consecutive days has autocorrelated returns, and under positive
autocorrelation naive annualization OVERSTATES the true Sharpe. The Trade Statistics tab reports both the naive
and the Lo-corrected value side by side.

##### Bootstrap confidence intervals
The Monte Carlo tab also offers a block-bootstrap confidence interval on Sharpe and CAGR: it resamples BLOCKS of
20 consecutive daily returns (rather than individual days) so short-term autocorrelation carries through into
each resampled path, then reports a 90% interval. Point estimates elsewhere in this app are single numbers with
no stated uncertainty; this is the one place that gives you an actual range.

##### Beta & alpha vs. benchmark
The Overview tab reports a single-factor (CAPM-style) regression of the strategy's daily returns on the selected
benchmark's: beta (market sensitivity), annualized alpha (return left over after removing that market exposure),
R\u00b2 (how much of the daily variance the benchmark exposure alone explains), and a t-stat for whether alpha is
distinguishable from zero. This is a single-factor model only -- it does not decompose returns against style
factors like size or value (that needs a Fama-French-style factor dataset this app doesn't have access to).

##### Known limitations
- **Survivorship bias**: every ticker in the universe is a large, currently-successful company chosen in
  hindsight. A strategy tested only on "winners that are still around" will look better than one tested on a
  realistic, unbiased, point-in-time universe.
- **Approximate market impact model**: see Execution cost model above.
- **Simplified borrow cost**: see Short selling & borrow cost above.
- **No point-in-time data store**: see Data source & provenance above.
- **Multiple-testing correction is parameter-level only**: the Deflated Sharpe Ratio corrects for scanning MA
  combinations, but scanning the full ticker universe and reporting whichever ticker looked best is the same
  kind of multiple-comparisons problem, uncorrected here.
- **Single-factor beta/alpha only**: see Beta & alpha vs. benchmark above.
- **In-sample results elsewhere in this app** (Overview, Trade Statistics, Parameter Heatmap, Monte Carlo) use
  the full dataset and the MA parameters you picked by hand \u2014 treat those as descriptive/exploratory, not
  as a validated result. The Out-of-Sample tab is the one place in this app built specifically to avoid that bias.
""")

if page == "PORTFOLIO BACKTEST":
    st.subheader("Portfolio assets & weights")
    default_included = ["AAPL", "NVDA", "SPY"]
    default_included = [t for t in default_included if t in all_tickers] or all_tickers[:3]

    # Merge anything staged via "+ Portfolio" buttons elsewhere in the app
    # (search results) into the widget's backing state, once, before it
    # renders -- consumed with pop() so it doesn't keep re-adding a ticker
    # the user has since manually removed from the picker.
    staged = st.session_state.pop("portfolio_staging", [])
    if staged:
        current = st.session_state.get("port_tickers_select", default_included)
        merged = list(dict.fromkeys(list(current) + [t for t in staged if t in all_tickers]))
        st.session_state["port_tickers_select"] = merged

    port_tickers = st.multiselect(
        "Choose assets (search by ticker)", all_tickers, default=default_included, key="port_tickers_select",
        format_func=lambda tk: f"{tk} \u2014 {TICKER_INFO.get(tk, (tk,))[0]}",
    )

    raw_weights = {}
    if port_tickers:
        even_weight = round(100 / len(port_tickers))
        weight_cols = st.columns(min(len(port_tickers), 6))
        for i, tk in enumerate(port_tickers):
            with weight_cols[i % len(weight_cols)]:
                raw_weights[tk] = st.number_input(
                    f"{tk} weight (%)", 0, 100, even_weight, key=f"w_{tk}"
                )

    st.subheader("Strategy controls")
    p1, p2, p3, p4, p5, p6, p7 = st.columns(7)
    strategy = p1.selectbox("Strategy", STRATEGIES)
    rebal_freq = p2.selectbox("Rebalancing", REBAL_OPTIONS, index=2)
    p_ma_fast = p3.number_input("MA Fast", 5, 100, 20)
    p_ma_slow = p4.number_input("MA Slow", 20, 300, 100)
    momentum_lookback = p5.number_input("Momentum lookback (d)", 20, 250, 90)
    p_tx_cost = p6.number_input("TX cost (bps)", 0, 100, 5, key="ptx")
    p_capital = p7.number_input("Capital ($)", 100, step=100, value=10000, key="pcap")
    r1c, r2c = st.columns([1, 3])
    default_top_k = max(1, len(port_tickers) // 2) if port_tickers else 1
    top_k = r1c.number_input(
        "Top K assets", 1, max(1, len(port_tickers)) if port_tickers else 1, min(default_top_k, max(1, len(port_tickers)) if port_tickers else 1),
        disabled=strategy != "Cross-Sectional Momentum",
        help="Cross-Sectional Momentum only: hold the K assets in this portfolio with the strongest trailing "
             "return relative to each other (not just positive in isolation), equal-weighted among them.",
    )
    risk_free_pct = r2c.slider("Risk-free rate (%/yr)", 0.0, 10.0, 2.0, step=0.1)

    if not port_tickers:
        st.info("Select at least one asset.")
    else:
        total_w = sum(raw_weights[tk] for tk in port_tickers)
        base_weights = {tk: (raw_weights[tk] / total_w if total_w > 0 else 1 / len(port_tickers)) for tk in port_tickers}
        prices_by_ticker = {tk: all_prices[all_prices["ticker"] == tk] for tk in port_tickers}

        result = run_portfolio_backtest(prices_by_ticker, port_tickers, base_weights, strategy, rebal_freq,
                                         p_ma_fast, p_ma_slow, momentum_lookback, p_tx_cost, p_capital, risk_free_pct,
                                         top_k=top_k)
        s = result["stats"]

        r1 = st.columns(4)
        r1[0].metric("Total Return", f"{s['total_return']*100:+.1f}%")
        r1[1].metric("CAGR", f"{s['cagr']*100:+.1f}%")
        r1[2].metric("Volatility", f"{s['vol']*100:.1f}%")
        r1[3].metric("Sharpe", f"{s['sharpe']:.2f}")
        r2 = st.columns(4)
        r2[0].metric("Sortino", f"{s['sortino']:.2f}")
        r2[1].metric("Calmar", f"{s['calmar']:.2f}")
        r2[2].metric("Max Drawdown", f"{s['max_dd']*100:.1f}%")
        r2[3].metric("Rebalances", f"{s['n_rebalances']}")

        st.markdown("##### Turnover & transaction costs")
        r3 = st.columns(3)
        r3[0].metric("Avg. turnover per rebalance", f"{s['avg_turnover']*100:.0f}%")
        r3[1].metric("Total cost paid", f"${s['total_cost_dollars']:,.0f}")
        r3[2].metric("Cost as % of capital", f"{s['total_cost_pct']*100:.2f}%")
        st.caption("Turnover is the fraction of the portfolio bought/sold at each rebalance to reach the new target weights.")

        fig_p = go.Figure()
        fig_p.add_trace(go.Scatter(x=result["dates"], y=result["portfolio_value"].values,
                                    line=dict(color="#E8A33D", width=2), name="Portfolio"))
        for idx in result["rebalance_idxs"][1:]:
            fig_p.add_vline(x=result["dates"][idx], line=dict(color="#3A4250", width=0.6, dash="dot"))
        fig_p.update_layout(height=400, title=f"Portfolio equity curve ({strategy}, {rebal_freq} rebalancing)",
                             margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig_p, width='stretch')

        cc1, cc2 = st.columns(2)
        with cc1:
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(x=result["dates"], y=result["drawdown"].values * 100, fill="tozeroy",
                                         line=dict(color="#D9724F"), name="Drawdown"))
            fig_dd.update_layout(height=320, title="Drawdown (%)", margin=dict(t=40, b=10, l=10, r=10))
            st.plotly_chart(fig_dd, width='stretch')
        with cc2:
            last_weights = result["weight_snapshots"][-1]
            cash_pct = max(0, 100 - sum(last_weights.values()) * 100)
            title = "Current target allocation (%)" + (f"  ({cash_pct:.0f}% in cash)" if cash_pct > 0.5 else "")
            fig_alloc = go.Figure(data=go.Bar(
                x=port_tickers, y=[last_weights[tk] * 100 for tk in port_tickers],
                marker_color=[TICKER_COLORS.get(tk, "#8891A0") for tk in port_tickers]))
            fig_alloc.update_layout(height=320, title=title, yaxis_range=[0, 100], margin=dict(t=40, b=10, l=10, r=10))
            st.plotly_chart(fig_alloc, width='stretch')

        st.caption(
            "Method note: assets are combined by dollar value each day; on a rebalance date the strategy re-evaluates "
            "target weights using the prior day's data (no lookahead), and any turnover incurs the transaction cost "
            "above (commission only -- no slippage or market impact modeled). Uninvested capital sits in cash "
            "earning 0% between rebalances. Sharpe and Sortino subtract the risk-free rate before annualizing. "
            "Data: daily closes via yfinance, split-adjusted."
        )

        portfolio_csv = pd.DataFrame({
            "date": result["dates"],
            "portfolio_value": result["portfolio_value"].values,
            "drawdown": result["drawdown"].values,
        })
        st.download_button(
            "Download portfolio equity curve (CSV)",
            portfolio_csv.to_csv(index=False),
            file_name=f"portfolio_{strategy.replace(' ', '_')}_{rebal_freq}.csv",
            mime="text/csv",
        )
