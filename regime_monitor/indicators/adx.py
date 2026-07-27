"""Average Directional Index (ADX) implementation.

Provides utilities to compute +DM, -DM, True Range and the ADX
indicator using Wilder smoothing. Functions accept a pandas
DataFrame containing 'high','low','close' and return pandas Series.
"""

import numpy as np
import pandas as pd


def _directional_moves(ohlcv: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Compute positive and negative directional movement (+DM, -DM).

    Expects columns: 'high', 'low'.
    """
    high = ohlcv["high"]
    low = ohlcv["low"]

    up_move = high.diff()
    down_move = -low.diff()  # L_{t-1} - L_t

    plus_dm = (up_move.where((up_move > down_move) & (up_move > 0), 0.0)).astype(
        "float64"
    )
    minus_dm = (down_move.where((down_move > up_move) & (down_move > 0), 0.0)).astype(
        "float64"
    )

    return plus_dm, minus_dm


def _true_range(ohlcv: pd.DataFrame) -> pd.Series:
    """
    True Range (TR), identical to what we use for ATR.
    Expects 'high', 'low', 'close' columns.
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


def compute_adx(ohlcv: pd.DataFrame, length: int = 14) -> pd.Series:
    """
    Compute Average Directional Index (ADX) using Wilder's method.

    Parameters
    ----------
    ohlcv : pd.DataFrame
        Must contain 'high', 'low', 'close' columns.
    length : int, default 14
        Lookback period for smoothing TR, +DM, -DM, and DX.

    Returns
    -------
    pd.Series
        ADX values, same index as `ohlcv`.
    """
    tr = _true_range(ohlcv)
    plus_dm, minus_dm = _directional_moves(ohlcv)

    # Wilder smoothing: EMA with alpha = 1/length
    tr_smooth = tr.ewm(alpha=1 / length, adjust=False).mean()
    plus_dm_smooth = plus_dm.ewm(alpha=1 / length, adjust=False).mean()
    minus_dm_smooth = minus_dm.ewm(alpha=1 / length, adjust=False).mean()

    # Avoid division by zero. Using np.nan (not pd.NA) keeps these Series
    # float64 — replacing with pd.NA forces an object-dtype coercion that
    # triggers infinite recursion in pandas 1.5.3's block manager
    # (pandas GH#45725).
    tr_smooth = tr_smooth.mask(tr_smooth == 0, np.nan)

    plus_di = 100 * (plus_dm_smooth / tr_smooth)
    minus_di = 100 * (minus_dm_smooth / tr_smooth)

    di_sum = plus_di + minus_di
    di_diff = (plus_di - minus_di).abs()

    dx = 100 * (di_diff / di_sum.mask(di_sum == 0, np.nan))

    # Final ADX: Wilder-smoothed DX
    adx = dx.ewm(alpha=1 / length, adjust=False).mean()

    return adx


def compute_adx_full(ohlcv: pd.DataFrame, length: int = 14) -> pd.DataFrame:
    """
    Compute ADX with +DI and -DI components.

    Parameters
    ----------
    ohlcv : pd.DataFrame
        Must contain 'high', 'low', 'close' columns.
    length : int, default 14
        Lookback period for smoothing.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'adx', 'plus_di', 'minus_di'.
    """
    tr = _true_range(ohlcv)
    plus_dm, minus_dm = _directional_moves(ohlcv)

    # Wilder smoothing
    tr_smooth = tr.ewm(alpha=1 / length, adjust=False).mean()
    plus_dm_smooth = plus_dm.ewm(alpha=1 / length, adjust=False).mean()
    minus_dm_smooth = minus_dm.ewm(alpha=1 / length, adjust=False).mean()

    tr_smooth = tr_smooth.mask(tr_smooth == 0, np.nan)

    plus_di = 100 * (plus_dm_smooth / tr_smooth)
    minus_di = 100 * (minus_dm_smooth / tr_smooth)

    di_sum = plus_di + minus_di
    di_diff = (plus_di - minus_di).abs()

    dx = 100 * (di_diff / di_sum.mask(di_sum == 0, np.nan))

    adx = dx.ewm(alpha=1 / length, adjust=False).mean()

    return pd.DataFrame({
        'adx': adx,
        'plus_di': plus_di,
        'minus_di': minus_di
    }, index=ohlcv.index)
