"""Fast rule-based regime history.

``detect_regime`` recomputes every indicator from scratch on each call, so
labelling a multi-year daily history one bar at a time is slow. This module
computes the indicator series **once** over the full price history, then reuses
the detector's real ``_combine_regime_label`` for each bar — so the labels match
the live engine while the whole history is produced in well under a second.
"""

from __future__ import annotations

import pandas as pd

from ..indicators.adx import compute_adx
from ..indicators.atr import compute_atr
from ..indicators.ema import EMAConfig, compute_emas
from ..indicators.regression import RegSlopeConfig, compute_regression_slope_with_r2
from .config import DEFAULT_CONFIG as cfg
from .detector import ChopDiagnostics, VolDiagnostics, _combine_regime_label
from .trend import TrendDiagnostics, _classify_ema_stack


def rule_based_history(spy: pd.DataFrame) -> pd.Series:
    """Return a Series of rule-based composite regime labels indexed by date."""
    close = spy["close"]

    emas = compute_emas(close, EMAConfig(
        short=cfg.trend.ema_short, mid=cfg.trend.ema_mid,
        long=cfg.trend.ema_long, slope_lookback=cfg.trend.slope_lookback,
    ))
    ema_s, ema_m, ema_l = emas["short"], emas["mid"], emas["long"]

    reg = compute_regression_slope_with_r2(
        close, cfg=RegSlopeConfig(length=cfg.trend.reg_length, use_log=True, normalize=True)
    )

    atr = compute_atr(spy, cfg.vol.atr_lookback)
    atr_pct = 100.0 * atr / close
    adx = compute_adx(spy, cfg.chop.adx_lookback)

    slope_min, r2_min = cfg.trend.slope_min, cfg.trend.r2_min
    labels: dict[pd.Timestamp, str] = {}

    # Extract aligned numpy arrays once — per-element Series lookups in a loop
    # are ~100x slower than iterating arrays.
    index = close.index
    a_slope = reg["slope"].to_numpy()
    a_r2 = reg["r2"].to_numpy()
    a_es, a_em, a_el = ema_s.to_numpy(), ema_m.to_numpy(), ema_l.to_numpy()
    a_price = close.to_numpy()
    a_atrp = atr_pct.to_numpy()
    a_adx = adx.to_numpy()

    for i, ts in enumerate(index):
        slope, r2 = a_slope[i], a_r2[i]
        es, em, el, price = a_es[i], a_em[i], a_el[i], a_price[i]
        if pd.isna(slope) or pd.isna(r2) or pd.isna(el):
            continue

        ema_stack = _classify_ema_stack(es, em, el, price)
        # Hybrid trend label — mirrors analyze_trend().
        if ema_stack == "bull" and slope >= 0:
            trend_label = "up"
        elif ema_stack == "bear" and slope <= 0:
            trend_label = "down"
        elif abs(slope) < slope_min or r2 < r2_min:
            trend_label = "flat"
        else:
            trend_label = "up" if slope > 0 else "down"

        # Vol label — mirrors analyze_vol()'s fixed ATR% thresholds.
        ap = float(a_atrp[i])
        if ap < cfg.vol.atr_low:
            vol_label = "low"
        elif ap < cfg.vol.atr_high:
            vol_label = "normal"
        elif ap < cfg.vol.atr_extreme:
            vol_label = "high"
        else:
            vol_label = "extreme"

        adx_val = a_adx[i]
        chop_label = "trending" if (pd.notna(adx_val) and adx_val >= cfg.chop.adx_trend_threshold) else "choppy"

        trend = TrendDiagnostics(
            label=trend_label, slope=float(slope), r2=float(r2),
            ema_short=float(es), ema_mid=float(em), ema_long=float(el), ema_stack=ema_stack,
        )
        vol = VolDiagnostics(vol_label=vol_label, atr_percent=ap)
        chop = ChopDiagnostics(chop_label=chop_label, adx=float(adx_val) if pd.notna(adx_val) else 0.0)
        labels[ts] = _combine_regime_label(trend, vol, chop)

    return pd.Series(labels, name="rule_regime")
