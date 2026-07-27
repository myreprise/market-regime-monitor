"""Bollinger Bands computation.

This module computes the center moving average, upper/lower bands,
and derived metrics such as bandwidth and percent-b (%B).
"""

import pandas as pd
from dataclasses import dataclass


@dataclass
class BollingerConfig:
    length: int = 20      # lookback window for MA and std
    num_std: float = 2.0  # number of standard deviations


def compute_bollinger(
    close: pd.Series,
    cfg: BollingerConfig = BollingerConfig(),
) -> pd.DataFrame:
    """
    Compute Bollinger Bands and derived metrics.

    Parameters
    ----------
    close : pd.Series
        Closing prices.
    cfg : BollingerConfig, optional
        Configuration with `length` and `num_std`.

    Returns
    -------
    pd.DataFrame
        Columns:
            - 'middle'     : moving average
            - 'upper'      : upper band
            - 'lower'      : lower band
            - 'bandwidth'  : (upper - lower) / middle
            - 'percent_b'  : (close - lower) / (upper - lower)
    """
    length = cfg.length
    num_std = cfg.num_std

    ma = close.rolling(window=length, min_periods=length).mean()
    std = close.rolling(window=length, min_periods=length).std()

    upper = ma + num_std * std
    lower = ma - num_std * std

    # Bandwidth: normalized band width
    bandwidth = (upper - lower) / ma

    # %B: where price sits relative to the bands
    range_ = (upper - lower)
    percent_b = (close - lower) / range_
    percent_b = percent_b.clip(lower=0.0, upper=1.0)  # optional clamp

    return pd.DataFrame(
        {
            "middle": ma,
            "upper": upper,
            "lower": lower,
            "bandwidth": bandwidth,
            "percent_b": percent_b,
        },
        index=close.index,
    )


# Backwards-compatible name used elsewhere in the codebase
# Older callers expect `compute_bollinger_bands`.
compute_bollinger_bands = compute_bollinger
