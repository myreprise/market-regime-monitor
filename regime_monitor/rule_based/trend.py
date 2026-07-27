import pandas as pd
from .config import DEFAULT_CONFIG as cfg
from ..indicators.ema import EMAConfig, compute_emas, compute_ema_slope
from ..indicators.regression import RegSlopeConfig, compute_regression_slope_with_r2
from dataclasses import dataclass

@dataclass
class TrendDiagnostics:
    label: str          # 'up', 'down', 'flat'
    slope: float        # regression slope (normalized)
    r2: float           # trend fit quality
    ema_short: float    # last EMA(short)
    ema_mid: float      # last EMA(mid)
    ema_long: float     # last EMA(long)
    ema_stack: str      # 'bull', 'bear', 'mixed'


def _classify_ema_stack(
    ema_short: float,
    ema_mid: float,
    ema_long: float,
    ref_price: float,
) -> str:
    """
    Classify EMA stacking relationship.

    Returns:
        'bull'  -> short > mid > long (by more than tolerance)
        'bear'  -> short < mid < long (by more than tolerance)
        'mixed' -> anything else
    """
    # tolerance as a fraction of current price
    tol = ref_price * cfg.trend.ema_tolerance_pct

    if (ema_short > ema_mid + tol) and (ema_mid > ema_long + tol):
        return "bull"
    if (ema_short < ema_mid - tol) and (ema_mid < ema_long - tol):
        return "bear"
    return "mixed"


def analyze_trend(ohlcv: pd.DataFrame) -> TrendDiagnostics:
    """
    Analyze trend using EMA stack + regression slope.

    Logic:
      1) Compute short/mid/long EMAs.
      2) Compute normalized regression slope + R^2.
      3) Use EMA stack as primary:
            - bull stack + slope >= 0 -> 'up'
            - bear stack + slope <= 0 -> 'down'
         Fallback:
            - if |slope| < slope_min or r2 < r2_min -> 'flat'
            - else sign(slope).
    """
    close = ohlcv["close"]

    # --- 1) EMAs
    ema_cfg = EMAConfig(
        short=cfg.trend.ema_short,
        mid=cfg.trend.ema_mid,
        long=cfg.trend.ema_long,
        slope_lookback=cfg.trend.slope_lookback,
    )
    emas = compute_emas(close, ema_cfg)

    ema_short = float(emas["short"].dropna().iloc[-1])
    ema_mid = float(emas["mid"].dropna().iloc[-1])
    ema_long = float(emas["long"].dropna().iloc[-1])

    ref_price = float(close.iloc[-1])
    ema_stack = _classify_ema_stack(
        ema_short=ema_short,
        ema_mid=ema_mid,
        ema_long=ema_long,
        ref_price=ref_price,
    )

    # --- 2) Regression slope + R^2
    reg_cfg = RegSlopeConfig(
        length=cfg.trend.reg_length,
        use_log=True,
        normalize=True,
    )
    reg_df = compute_regression_slope_with_r2(close, cfg=reg_cfg)
    latest = reg_df.dropna().iloc[-1]
    slope = float(latest["slope"])
    r2 = float(latest["r2"])

    slope_min = cfg.trend.slope_min
    r2_min = cfg.trend.r2_min

    # --- 3) Hybrid trend label

    # Case A: clear EMA stack, regression agrees in direction (or is neutral)
    if ema_stack == "bull" and slope >= 0:
        label = "up"
    elif ema_stack == "bear" and slope <= 0:
        label = "down"
    else:
        # Case B: fall back to pure regression logic
        if abs(slope) < slope_min or r2 < r2_min:
            label = "flat"
        else:
            label = "up" if slope > 0 else "down"

    return TrendDiagnostics(
        label=label,
        slope=slope,
        r2=r2,
        ema_short=ema_short,
        ema_mid=ema_mid,
        ema_long=ema_long,
        ema_stack=ema_stack,
    )



def classify_trend(ohlcv: pd.DataFrame, cfg_override=None) -> tuple[str, float]:
    close = ohlcv["close"]

    ema_cfg = EMAConfig(
        short=cfg.trend.ema_short,
        mid=cfg.trend.ema_mid,
        long=cfg.trend.ema_long,
        slope_lookback=cfg.trend.slope_lookback,
    )

    emas = compute_emas(close, ema_cfg)
    slope = compute_ema_slope(emas["mid"], ema_cfg.slope_lookback)

    latest_slope = float(slope.dropna().iloc[-1])

    if latest_slope > cfg.trend.slope_min:
        label = "up"
    elif latest_slope < -cfg.trend.slope_min:
        label = "down"
    else:
        label = "flat"

    return label, latest_slope