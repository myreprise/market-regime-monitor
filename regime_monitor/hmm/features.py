"""Feature construction for the 4-state regime HMM.

This is a faithful, self-contained port of the vectorised feature builder the
model was trained with (``_build_training_features`` in the original
``RegimeHMMService``). Computing the features here — directly from price frames
as a full time series — means training and inference operate on exactly the
same feature distribution, and lets us recompute the entire regime history
reproducibly from public price data alone.

The 18-feature contract (order is fixed — it must match the fitted scaler/PCA):

    0  spy_ema21_spread          (close - EMA21)  / EMA21
    1  spy_ema50_spread          (close - EMA50)  / EMA50
    2  spy_ema200_spread         (close - EMA200) / EMA200
    3  qqq_spy_rel_21d           QQQ 21d return - SPY 21d return
    4  vix_pctl_252d             rolling 252-bar VIX percentile [0,1]
    5  vix_slope_10d             (VIX[t] - VIX[t-10]) / (10 * VIX[t])
    6  breadth_ema50             fraction of 11 SPDR sectors above their EMA50
    7  spy_return_126d           SPY 126-day return (fraction)
    8  credit_spread_21d         HYG 21d return - IEF 21d return
    9  tlt_return_21d            TLT 21-day return (fraction)
    10 copper_gold_21d           CPER 21d return - GLD 21d return
    11 vix_accel_5d              vix_slope_10d.diff(5)
    12 credit_spread_momentum    credit_spread_21d.diff(5)
    13 breadth_acceleration      breadth_ema50.diff(5)
    14 tsy_3m_yield_slope        ^IRX 5-day slope
    15 fed_funds_futures_slope   0.0 (not populated; matches training)
    16 yield_curve_slope_10y2y   ^TNX - ^IRX (10y - 3m proxy)
    17 recession_probability_nyfed  0.0 (not populated; matches training)
"""

from __future__ import annotations

from typing import Dict

import pandas as pd

FEATURE_NAMES = [
    "spy_ema21_spread",
    "spy_ema50_spread",
    "spy_ema200_spread",
    "qqq_spy_rel_21d",
    "vix_pctl_252d",
    "vix_slope_10d",
    "breadth_ema50",
    "spy_return_126d",
    "credit_spread_21d",
    "tlt_return_21d",
    "copper_gold_21d",
    "vix_accel_5d",
    "credit_spread_momentum",
    "breadth_acceleration",
    "tsy_3m_yield_slope",
    "fed_funds_futures_slope",
    "yield_curve_slope_10y2y",
    "recession_probability_nyfed",
]

# EMA-200 needs >=200 bars; VIX rolling percentile needs 252.
WARMUP_BARS = 252

SPDR_TICKERS = [
    "XLB", "XLE", "XLF", "XLI", "XLK",
    "XLP", "XLU", "XLV", "XLY", "XLC", "XLRE",
]

# Every ticker the feature builder consumes (sector ETFs + SPY/VIX + auxiliaries).
REQUIRED_TICKERS = ["SPY", "^VIX"] + SPDR_TICKERS
AUX_TICKERS = ["QQQ", "HYG", "IEF", "TLT", "GLD", "CPER", "^IRX", "^TNX"]
ALL_TICKERS = REQUIRED_TICKERS + AUX_TICKERS


def _close(df: pd.DataFrame) -> pd.Series | None:
    if df is None or df.empty:
        return None
    for col in ("close", "Close", "CLOSE", "adj close", "Adj Close"):
        if col in df.columns:
            return df[col]
    return None


def _sector_breadth(prices: Dict[str, pd.DataFrame], target_index: pd.Index) -> pd.Series:
    """Fraction of SPDR sectors trading above their own EMA50, per bar."""
    series = {}
    for ticker in SPDR_TICKERS:
        close = _close(prices.get(ticker))
        if close is None or len(close) < 60:
            continue
        ema50 = close.ewm(span=50, adjust=False).mean()
        series[ticker] = (close > ema50).astype(float)
    if not series:
        return pd.Series(0.5, index=target_index, dtype=float)
    breadth_df = pd.DataFrame(series).reindex(target_index).ffill()
    return breadth_df.mean(axis=1)


def build_feature_matrix(prices: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build the full (n_bars, 18) feature time series from price frames.

    ``prices`` maps ticker -> OHLCV DataFrame (lowercase columns, DatetimeIndex).
    Returns a DataFrame indexed by date with columns in ``FEATURE_NAMES`` order,
    after warm-up truncation and NaN drop.
    """
    spy_close = _close(prices.get("SPY"))
    vix_close = _close(prices.get("^VIX"))
    if spy_close is None or vix_close is None:
        raise ValueError("SPY or ^VIX close prices unavailable.")

    idx = spy_close.index

    ema21 = spy_close.ewm(span=21, adjust=False).mean()
    ema50 = spy_close.ewm(span=50, adjust=False).mean()
    ema200 = spy_close.ewm(span=200, adjust=False).mean()
    ema21_spread = (spy_close - ema21) / ema21
    ema50_spread = (spy_close - ema50) / ema50
    ema200_spread = (spy_close - ema200) / ema200

    vix_aligned = vix_close.reindex(idx).ffill()
    vix_pctl = vix_aligned.rolling(252, min_periods=60).rank(pct=True)
    vix_slope = (vix_aligned - vix_aligned.shift(10)) / (10.0 * vix_aligned)

    breadth = _sector_breadth(prices, idx)

    def ret21(ticker: str) -> pd.Series:
        close = _close(prices.get(ticker))
        if close is None:
            return pd.Series(0.0, index=idx)
        return close.pct_change(21).reindex(idx).ffill().fillna(0.0)

    def slope5(ticker: str) -> pd.Series:
        close = _close(prices.get(ticker))
        if close is None:
            return pd.Series(0.0, index=idx)
        slope = (close - close.shift(5)) / close.replace(0, 1e-8)
        return slope.reindex(idx).ffill().fillna(0.0)

    spy_ret_126 = spy_close.pct_change(126).fillna(0.0)
    qqq_spy_rel = ret21("QQQ") - spy_close.pct_change(21).fillna(0.0)
    credit_spread = ret21("HYG") - ret21("IEF")
    tlt_ret21 = ret21("TLT")
    copper_gold = ret21("CPER") - ret21("GLD")

    tsy_3m_slope = slope5("^IRX")

    tnx_close = _close(prices.get("^TNX"))
    irx_close = _close(prices.get("^IRX"))
    if tnx_close is not None and irx_close is not None:
        yc_slope = (tnx_close - irx_close).reindex(idx).ffill().fillna(0.0)
    else:
        yc_slope = pd.Series(0.0, index=idx)

    feat = pd.DataFrame(
        {
            "spy_ema21_spread": ema21_spread,
            "spy_ema50_spread": ema50_spread,
            "spy_ema200_spread": ema200_spread,
            "qqq_spy_rel_21d": qqq_spy_rel,
            "vix_pctl_252d": vix_pctl,
            "vix_slope_10d": vix_slope,
            "breadth_ema50": breadth,
            "spy_return_126d": spy_ret_126,
            "credit_spread_21d": credit_spread,
            "tlt_return_21d": tlt_ret21,
            "copper_gold_21d": copper_gold,
            "vix_accel_5d": vix_slope.diff(periods=5).fillna(0.0),
            "credit_spread_momentum": credit_spread.diff(periods=5).fillna(0.0),
            "breadth_acceleration": breadth.diff(periods=5).fillna(0.0),
            "tsy_3m_yield_slope": tsy_3m_slope,
            "fed_funds_futures_slope": pd.Series(0.0, index=idx),
            "yield_curve_slope_10y2y": yc_slope,
            "recession_probability_nyfed": pd.Series(0.0, index=idx),
        },
        index=idx,
    )

    feat = feat.iloc[WARMUP_BARS:].dropna()
    return feat[FEATURE_NAMES]
