"""EMA helpers and small EMA-based utilities.

Includes helpers to compute single EMAs, multiple configured EMAs,
and a simple slope approximation for trend strength.
"""

import pandas as pd
from dataclasses import dataclass
from typing import Dict


@dataclass
class EMAConfig:
    short: int = 21
    mid: int = 50
    long: int = 200
    slope_lookback: int = 20  # for trend slope calculations


def compute_ema(series: pd.Series, length: int) -> pd.Series:
    """
    Compute an Exponential Moving Average (EMA) over `length` periods.

    Parameters
    ----------
    series : pd.Series
        Typically the closing prices.
    length : int
        Lookback length for the EMA.

    Returns
    -------
    pd.Series
        EMA series with the same index as `series`.
    """
    return series.ewm(span=length, adjust=False).mean()


def compute_emas(close: pd.Series, cfg: EMAConfig) -> Dict[str, pd.Series]:
    """
    Compute short, mid, and long EMAs using EMAConfig.

    Returns a dict with keys: 'short', 'mid', 'long'.
    """
    ema_short = compute_ema(close, cfg.short)
    ema_mid = compute_ema(close, cfg.mid)
    ema_long = compute_ema(close, cfg.long)

    return {
        "short": ema_short,
        "mid": ema_mid,
        "long": ema_long,
    }


def compute_ema_slope(ema: pd.Series, lookback: int) -> pd.Series:
    """
    Compute a simple slope of the EMA over a rolling window.

    Slope_t ≈ (EMA_t - EMA_{t-lookback}) / lookback

    This gives you an approximate 'points per bar' slope for trend strength.

    Parameters
    ----------
    ema : pd.Series
        EMA series.
    lookback : int
        Number of bars over which to measure the slope.

    Returns
    -------
    pd.Series
        EMA slope series.
    """
    prev = ema.shift(lookback)
    slope = (ema - prev) / float(lookback)
    return slope
