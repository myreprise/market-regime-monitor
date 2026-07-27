import pandas as pd
from .config import ChopConfig
from ..indicators.adx import compute_adx
from dataclasses import dataclass


@dataclass
class ChopAnalysis:
    chop_label: str
    adx: float
    structure: str | None = None



def classify_chop(
    ohlcv: pd.DataFrame,
    cfg: ChopConfig
) -> tuple[str, float]:
    """
    Classify price as 'trending' or 'choppy' using ADX.
    Returns (label, latest_adx_value).
    """
    if ohlcv.empty:
        return "choppy", 0.0

    adx = compute_adx(ohlcv, cfg.adx_lookback)

    latest = float(adx.iloc[-1]) if not adx.dropna().empty else 0.0

    if latest >= cfg.adx_trend_threshold:
        label = "trending"
    else:
        label = "choppy"

    return label, latest


def analyze_chop(ohlcv: pd.DataFrame, cfg: ChopConfig) -> ChopAnalysis:
    """Return richer chop diagnostics including a `structure` hint.

    `structure` can be one of:
      - 'trending' : ADX is high (>= threshold) and slope positive
      - 'coil'     : ADX is below threshold but rising (potential buildup)
      - 'choppy'   : ADX is low/flat or declining

    This is intentionally conservative and only supplies a hint to the
    caller; callers should tolerate None values.
    """
    if ohlcv.empty:
        return ChopAnalysis("choppy", 0.0, None)

    adx = compute_adx(ohlcv, cfg.adx_lookback)
    if adx.dropna().empty:
        return ChopAnalysis("choppy", 0.0, None)

    latest = float(adx.dropna().iloc[-1])
    label = "trending" if latest >= cfg.adx_trend_threshold else "choppy"

    # Estimate recent ADX slope (simple finite-difference over a small window)
    # Use a short window (3) to capture recent direction without being too noisy.
    try:
        window = min(3, len(adx.dropna()))
        recent = adx.dropna().iloc[-window:]
        # slope = last - first over the window
        adx_slope = float(recent.iloc[-1] - recent.iloc[0])
    except Exception:
        adx_slope = 0.0

    if latest >= cfg.adx_trend_threshold and adx_slope > 0:
        structure = "trending"
    elif latest < cfg.adx_trend_threshold and adx_slope > 0:
        structure = "coil"
    else:
        structure = "choppy"

    return ChopAnalysis(label, latest, structure)
