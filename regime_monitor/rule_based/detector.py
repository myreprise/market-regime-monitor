# src/regime/detector.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd

from .config import DEFAULT_CONFIG as cfg
from .trend import analyze_trend, classify_trend, TrendDiagnostics
from .vol import classify_vol, analyze_vol
from .chop import classify_chop, analyze_chop


@dataclass
class VolDiagnostics:
    vol_label: str
    atr_percent: float
    squeeze_state: str | None = None


@dataclass
class ChopDiagnostics:
    chop_label: str
    adx: float
    structure: str | None = None


@dataclass
class RegimeSnapshot:
    """
    Full snapshot of the market regime for a given symbol and time window.
    """
    symbol: str
    start: pd.Timestamp
    end: pd.Timestamp
    timeframe: str  # e.g. "1d"
    regime: str     # e.g. "bull", "bear", "range", "high_vol"
    confidence: float  # 0.0 – 1.0

    trend: TrendDiagnostics
    vol: VolDiagnostics
    chop: ChopDiagnostics

    diagnostics: Dict[str, Any]


def _normalize_timestamp(ts: Optional[pd.Timestamp]) -> Optional[pd.Timestamp]:
    if ts is None:
        return None
    if isinstance(ts, pd.Timestamp):
        return ts
    return pd.Timestamp(ts)


def _combine_regime_label(
    trend: TrendDiagnostics,
    vol: VolDiagnostics,
    chop: ChopDiagnostics,
) -> str:
    """
    Map trend/vol/chop into a composite regime label.

    Priority:
      1) Direction (trend + EMA stack)
      2) Volatility (normal vs high)
      3) Chop → 'range' only when direction is unclear
    """
    trend_label = trend.label          # 'up' | 'down' | 'flat'
    ema_stack = trend.ema_stack        # 'bull' | 'bear' | 'mixed'
    vol_label = vol.vol_label          # 'low' | 'normal' | 'high' | 'extreme'
    chop_label = chop.chop_label       # 'trending' | 'choppy' | 'coil'

    # 1) Decide direction: use EMA stack as a strong hint
    direction: str | None = None

    if trend_label == "up" or ema_stack == "bull":
        direction = "up"
    elif trend_label == "down" or ema_stack == "bear":
        direction = "down"
    else:
        direction = None

    # 2) If we have a clear direction, vol decorates the label
    if direction == "up":
        if vol_label in {"high", "extreme"}:
            return "bull_high_vol"
        else:
            return "bull"

    if direction == "down":
        if vol_label in {"high", "extreme"}:
            return "bear_high_vol"
        else:
            return "bear"

    # 3) No clear direction → use vol + chop
    if vol_label in {"high", "extreme"} and chop_label == "trending":
        # high-vol but directionless chop (rare, but can happen)
        return "high_vol"

    # default: sideways / non-directional
    return "range"



def _compute_confidence(
    trend: TrendDiagnostics,
    vol: VolDiagnostics,
    chop: ChopDiagnostics,
) -> float:
    """
    Heuristic confidence score based on:
      - trend slope & R^2
      - volatility state
      - chop/trend structure

    Everything normalized into [0, 1].
    """

    # --- Trend component
    # Normalize slope strength relative to configured minimum
    slope_min = max(cfg.trend.slope_min, 1e-9)
    slope_strength = min(1.0, abs(trend.slope) / (2.0 * slope_min))

    # Normalize R^2 component
    r2_min = cfg.trend.r2_min
    if trend.r2 <= r2_min:
        r2_quality = 0.0
    else:
        r2_quality = min(1.0, (trend.r2 - r2_min) / (1.0 - r2_min))

    trend_score = 0.6 * slope_strength + 0.4 * r2_quality

    # --- Volatility component
    vol_map = {
        "low": 0.4,
        "normal": 0.7,
        "high": 0.9,
        "extreme": 0.8,
    }
    vol_score = vol_map.get(vol.vol_label, 0.6)

    # --- Structure / chop component
    chop_map = {
        "trending": 1.0,
        "coil": 0.8,
        "choppy": 0.4,
    }
    chop_score = chop_map.get(chop.chop_label, 0.6)

    # Simple average for now — easy to tweak later
    raw_conf = (trend_score + vol_score + chop_score) / 3.0
    return float(max(0.0, min(1.0, raw_conf)))


def detect_regime(
    ohlcv: pd.DataFrame,
    symbol: str = "SPY",
    timeframe: str = "1d",
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
) -> RegimeSnapshot:
    """
    Main entrypoint: detect the market regime over a given OHLCV window.

    Parameters
    ----------
    ohlcv : pd.DataFrame
        Price data with at least 'open', 'high', 'low', 'close', 'volume'.
        Index should be datetime-like and sorted ascending.
    symbol : str, default "SPY"
        Ticker this OHLCV belongs to.
    timeframe : str, default "1d"
        Timeframe / bar size label (informational).
    start : pd.Timestamp or str, optional
        Optional start date for slicing the OHLCV window.
    end : pd.Timestamp or str, optional
        Optional end date for slicing the OHLCV window.

    Returns
    -------
    RegimeSnapshot
        Snapshot object containing regime label, confidence, and diagnostics.
    """

    if ohlcv.empty:
        raise ValueError("detect_regime: received empty OHLCV dataframe")

    ohlcv = ohlcv.sort_index()

    start_ts = _normalize_timestamp(start) or ohlcv.index[0]
    end_ts = _normalize_timestamp(end) or ohlcv.index[-1]

    window = ohlcv.loc[(ohlcv.index >= start_ts) & (ohlcv.index <= end_ts)]
    if window.empty:
        raise ValueError("detect_regime: OHLCV window is empty after slicing")

    # --- Block 1: Trend
    # Allow tests to monkeypatch `classify_trend` to short-circuit heavy analysis
    try:
        ct = classify_trend(window, cfg)
    except TypeError:
        ct = classify_trend(window)

    if isinstance(ct, tuple):
        trend_label = ct[0]
    else:
        trend_label = ct

    # Compute richer diagnostics when possible, but tolerate failures on tiny inputs
    try:
        trend_diag = analyze_trend(window)
    except Exception:
        trend_diag = TrendDiagnostics(
            label=trend_label,
            slope=0.0,
            r2=0.0,
            ema_short=0.0,
            ema_mid=0.0,
            ema_long=0.0,
            ema_stack="mixed",
        )

    # --- Block 2: Volatility
    # Prefer the richer `analyze_vol` diagnostics (includes squeeze_state,
    # band widths, etc.). Fall back to the lightweight `classify_vol`
    # if `analyze_vol` fails for any reason (e.g., tiny inputs).
    try:
        vol_local = analyze_vol(window)
        vol_diag = VolDiagnostics(vol_local.vol_label, vol_local.atr_percent, squeeze_state=vol_local.squeeze_state)
    except Exception:
        vol_label, atrp = classify_vol(window, cfg.vol)
        vol_diag = VolDiagnostics(vol_label, atrp)

    # --- Block 3: Chop / structure
    # Prefer the richer `analyze_chop` diagnostics (includes `structure`).
    try:
        chop_local = analyze_chop(window, cfg.chop)
        chop_diag = ChopDiagnostics(chop_local.chop_label, chop_local.adx, structure=chop_local.structure)
    except Exception:
        chop_label, adx = classify_chop(window, cfg.chop)
        chop_diag = ChopDiagnostics(chop_label, adx)

    # --- Combine into composite regime + confidence
    regime_label = _combine_regime_label(trend_diag, vol_diag, chop_diag)
    confidence = _compute_confidence(trend_diag, vol_diag, chop_diag)

    diagnostics: Dict[str, Any] = {
        "trend": trend_diag.label,
        "trend_label": trend_diag.label,
        "trend_slope": trend_diag.slope,
        "trend_r2": trend_diag.r2,
        "trend_ema_stack": trend_diag.ema_stack,
        "vol_label": vol_diag.vol_label,
        "atr_percent": getattr(vol_diag, "atr_percent", None),
        "squeeze_state": getattr(vol_diag, "squeeze_state", None),
        "chop_label": chop_diag.chop_label,
        "adx": getattr(chop_diag, "adx", None),
        "structure": getattr(chop_diag, "structure", None),
    }

    return RegimeSnapshot(
        symbol=symbol,
        start=start_ts,
        end=end_ts,
        timeframe=timeframe,
        regime=regime_label,
        confidence=confidence,
        trend=trend_diag,
        vol=vol_diag,
        chop=chop_diag,
        diagnostics=diagnostics,
    )


def summarize_regime(snapshot: RegimeSnapshot) -> str:
    """
    Human-readable one-liner summary for CLI / logs.
    """
    return (
        f"{snapshot.symbol} {snapshot.timeframe} "
        f"{snapshot.start.date()} → {snapshot.end.date()} | "
        f"Regime: {snapshot.regime} (conf={snapshot.confidence:.2f}) | "
        f"Trend={snapshot.trend.label} "
        f"(slope={snapshot.trend.slope:.4f}, r2={snapshot.trend.r2:.2f}), "
        f"Vol={snapshot.vol.vol_label}, "
        f"Chop={snapshot.chop.chop_label}"
    )


from .types import MarketSnapshot as _MarketSnapshot


def get_regime(snapshot_or_ohlcv, *args, **kwargs):
    """Backward-compatible wrapper that accepts either a MarketSnapshot
    or an OHLCV DataFrame and returns (label, confidence, diagnostics).
    """
    # If a MarketSnapshot was passed, extract its fields
    if isinstance(snapshot_or_ohlcv, _MarketSnapshot):
        snap = snapshot_or_ohlcv
        rs = detect_regime(snap.ohlcv, symbol=snap.symbol, timeframe=snap.timeframe, *args, **kwargs)
    else:
        # assume an ohlcv DataFrame was provided
        rs = detect_regime(snapshot_or_ohlcv, *args, **kwargs)

    return rs.regime, rs.confidence, rs.diagnostics
