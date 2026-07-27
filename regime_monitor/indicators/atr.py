"""Average True Range (ATR) utilities.

This module provides helpers to compute True Range (TR) and Average
True Range (ATR) from OHLCV data. Functions accept a pandas DataFrame
that contains at least the `high`, `low` and `close` columns and return
pandas Series aligned to the same index.
"""

import pandas as pd
from typing import Literal


def compute_true_range(ohlcv: pd.DataFrame) -> pd.Series:
    """
    Compute True Range (TR) from OHLCV data.

    Expects columns: 'high', 'low', 'close'.
    Index should be a DatetimeIndex or something sortable.
    """
    high = ohlcv["high"]
    low = ohlcv["low"]
    close = ohlcv["close"]

    prev_close = close.shift(1)

    range_hl = high - low
    range_hc = (high - prev_close).abs()
    range_lc = (low - prev_close).abs()

    tr = pd.concat([range_hl, range_hc, range_lc], axis=1).max(axis=1)
    return tr


def compute_atr(
    ohlcv: pd.DataFrame,
    length: int = 14,
    method: Literal["wilder", "sma"] = "wilder",
) -> pd.Series:
    """
    Compute Average True Range (ATR) from OHLCV data.

    Parameters
    ----------
    ohlcv : pd.DataFrame
        Must contain 'high', 'low', 'close' columns.
    length : int, default 14
        Lookback period for ATR.
    method : {'wilder', 'sma'}, default 'wilder'
        'wilder' -> exponential smoothing with alpha = 1/length
        'sma'    -> simple moving average over the last `length` TR values.

    Returns
    -------
    pd.Series
        ATR values, indexed the same as `ohlcv.index`.
    """
    tr = compute_true_range(ohlcv)

    if method == "wilder":
        # Wilder's ATR uses an EMA with alpha = 1/length
        atr = tr.ewm(alpha=1 / length, adjust=False).mean()
    elif method == "sma":
        atr = tr.rolling(window=length, min_periods=length).mean()
    else:
        raise ValueError(f"Unknown ATR method: {method}")

    return atr
