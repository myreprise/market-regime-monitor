"""Keltner Channel indicator.

Provides a configurable center line (EMA or SMA) with bands derived from
Average True Range (ATR). The implementation intentionally mirrors common
industry formulas and returns a DataFrame containing middle/upper/lower
and channel width metrics.
"""

import pandas as pd
from dataclasses import dataclass
from typing import Literal

from .atr import compute_atr


@dataclass
class KeltnerConfig:
    ma_length: int = 20          # EMA length for center line
    atr_length: int = 20         # ATR lookback
    atr_mult: float = 2.0        # ATR multiplier for band width
    ma_type: Literal["ema", "sma"] = "ema"  # center-line type


def _moving_average(
    close: pd.Series,
    length: int,
    ma_type: Literal["ema", "sma"] = "ema",
) -> pd.Series:
    if ma_type == "ema":
        return close.ewm(span=length, adjust=False).mean()
    elif ma_type == "sma":
        return close.rolling(window=length, min_periods=length).mean()
    else:
        raise ValueError(f"Unknown ma_type: {ma_type!r}")


def compute_keltner(
    ohlcv: pd.DataFrame,
    cfg: KeltnerConfig = KeltnerConfig(),
) -> pd.DataFrame:
    """
    Compute Keltner Channels.

    Parameters
    ----------
    ohlcv : pd.DataFrame
        Must contain 'high', 'low', 'close' columns.
    cfg : KeltnerConfig, optional
        Configuration for MA length, ATR length, multiplier, and MA type.

    Returns
    -------
    pd.DataFrame
        Columns:
            - 'middle'     : center line (EMA or SMA of close)
            - 'upper'      : middle + atr_mult * ATR
            - 'lower'      : middle - atr_mult * ATR
            - 'channel_width'  : upper - lower
            - 'channel_width_norm' : (upper - lower) / middle
    """
    close = ohlcv["close"]

    middle = _moving_average(
        close,
        length=cfg.ma_length,
        ma_type=cfg.ma_type,
    )

    atr = compute_atr(ohlcv, length=cfg.atr_length, method="wilder")

    upper = middle + cfg.atr_mult * atr
    lower = middle - cfg.atr_mult * atr

    channel_width = upper - lower
    channel_width_norm = channel_width / middle

    return pd.DataFrame(
        {
            "middle": middle,
            "upper": upper,
            "lower": lower,
            "channel_width": channel_width,
            "channel_width_norm": channel_width_norm,
        },
        index=ohlcv.index,
    )


# Backwards-compatible alias expected by older callers
compute_keltner_channels = compute_keltner
