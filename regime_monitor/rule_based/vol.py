import pandas as pd
from .config import VolConfig, DEFAULT_CONFIG as cfg
from ..indicators.atr import compute_atr
from ..indicators.bollinger import compute_bollinger
from ..indicators.keltner import compute_keltner
from dataclasses import dataclass


@dataclass
class VolDiagnostics:
    vol_label: str          # 'low' | 'normal' | 'high' | 'extreme'
    atr_percent: float      # latest ATR% of price
    bb_width: float         # latest BB relative width
    kc_width: float         # latest KC relative width
    squeeze_state: str      # 'compressed' | 'normal' | 'expanding'


def classify_vol(
    ohlcv: pd.DataFrame,
    cfg: VolConfig
) -> tuple[str, float]:
    """
    Classify volatility as 'low', 'normal', or 'high' using ATR% percentiles.
    Returns (label, current_atr_percent).
    """
    if ohlcv.empty:
        return "normal", 0.0

    close = ohlcv["close"]
    atr = compute_atr(ohlcv, cfg.atr_lookback)
    atrp = (atr / close) * 100.0  # ATR as % of price
    # if division yielded a single-column DataFrame, convert to Series
    if isinstance(atrp, pd.DataFrame):
        atrp = atrp.iloc[:, 0]

    # Make sure we have enough data to compute percentiles
    if atrp.dropna().empty:
        return "normal", 0.0

    window = atrp.dropna().iloc[-cfg.atrp_lookback:]
    if len(window) == 0:
        current = float(atrp.dropna().iloc[-1])
        return "normal", current

    current = float(window.iloc[-1])
    rank_pct = float(window.rank(pct=True).iloc[-1] * 100.0)

    if rank_pct >= cfg.atrp_high_pct:
        label = "high"
    elif rank_pct <= cfg.atrp_low_pct:
        label = "low"
    else:
        label = "normal"

    return label, current


def _compute_band_widths(
    ohlcv: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """
    Compute relative Bollinger and Keltner band widths.

    Returns:
        bb_width, kc_width  (both as fractions of price, e.g. 0.05 = 5%)
    """
    close = ohlcv["close"]

    bb = compute_bollinger(
        close=close,
        cfg=type("C", (), {"length": cfg.vol.bb_length, "num_std": cfg.vol.bb_std})(),
    )
    # expect columns: 'middle', 'upper', 'lower'

    kc = compute_keltner(
        ohlcv=ohlcv,
        cfg=type("C", (), {"ma_length": cfg.vol.kc_length, "atr_length": cfg.vol.kc_length, "atr_mult": cfg.vol.kc_atr_mult, "ma_type": "ema"})(),
    )
    # expect columns: 'middle', 'upper', 'lower'

    bb_width = (bb["upper"] - bb["lower"]) / close
    kc_width = (kc["upper"] - kc["lower"]) / close

    bb_width.name = "bb_width"
    kc_width.name = "kc_width"

    return bb_width, kc_width



def _classify_squeeze_state(
    bb_width: pd.Series,
    kc_width: pd.Series,
) -> str:
    """
    Classify volatility structure based on BB vs KC.

    Logic:
      - 'compressed' if BB < squeeze_in_ratio * KC
      - 'expanding' if BB > squeeze_out_ratio * KC and
                     BB width has been increasing recently
      - else 'normal'
    """
    if bb_width.empty or kc_width.empty:
        return "normal"

    bb_last = float(bb_width.iloc[-1])
    kc_last = float(kc_width.iloc[-1])

    if kc_last <= 0:
        return "normal"

    ratio = bb_last / kc_last
    in_thr = cfg.vol.squeeze_in_ratio
    out_thr = cfg.vol.squeeze_out_ratio

    # Check for compression
    if ratio < in_thr:
        return "compressed"

    # Check for expansion
    lookback = min(cfg.vol.expansion_lookback, len(bb_width))
    if lookback >= 2:
        recent = bb_width.iloc[-lookback:]
        # simple test: is width increasing vs its min in this window?
        if ratio > out_thr and bb_last > float(recent.min()):
            return "expanding"

    return "normal"


def analyze_vol(ohlcv: pd.DataFrame) -> VolDiagnostics:
    """
    Analyze volatility regime and squeeze state.

    Uses:
      - ATR% for vol_label
      - Bollinger vs Keltner width for squeeze_state
    """
    close = ohlcv["close"]

    # --- 1) ATR% and vol_label ---
    atr = compute_atr(
        ohlcv=ohlcv,
        length=cfg.vol.atr_lookback,
    )
    # latest ATR value
    atr_last = float(atr.dropna().iloc[-1])

    ref_price = float(close.iloc[-1])
    atr_percent = 100.0 * atr_last / ref_price if ref_price != 0 else 0.0

    # map ATR% to vol label
    if atr_percent < cfg.vol.atr_low:
        vol_label = "low"
    elif atr_percent < cfg.vol.atr_high:
        vol_label = "normal"
    elif atr_percent < cfg.vol.atr_extreme:
        vol_label = "high"
    else:
        vol_label = "extreme"

    # --- 2) Band widths + squeeze_state ---
    bb_width, kc_width = _compute_band_widths(ohlcv)
    squeeze_state = _classify_squeeze_state(bb_width, kc_width)

    bb_last = float(bb_width.dropna().iloc[-1])
    kc_last = float(kc_width.dropna().iloc[-1])

    return VolDiagnostics(
        vol_label=vol_label,
        atr_percent=atr_percent,
        bb_width=bb_last,
        kc_width=kc_last,
        squeeze_state=squeeze_state,
    )
