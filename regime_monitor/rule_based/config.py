from dataclasses import dataclass, field

# regime/config.py (example – adapt to your style)
@dataclass
class TrendConfig:
    ema_short = 21
    ema_mid = 50
    ema_long = 200

    # how much separation between EMAs to consider them clearly stacked
    ema_tolerance_pct = 0.0015  # 0.15% of price

    # regression slope window
    reg_length = 30            # shorter than 50, more responsive
    slope_lookback = 20        # if used in EMAConfig

    # thresholds
    slope_min = 0.00015        # was 0.0005 – too strict
    r2_min = 0.12              # was 0.3 – too strict for SPY



@dataclass
class VolConfig:
    # ATR volatility bands (already roughly in place)
    atr_lookback = 14
    atr_low = 0.8       # example thresholds, tune later
    atr_high = 2.0
    atr_extreme = 4.0

    # Bollinger / Keltner squeeze
    bb_length = 20
    bb_std = 2.0

    kc_length = 20
    kc_atr_mult = 1.5

    # Ratio thresholds: BB width vs KC width
    squeeze_in_ratio = 0.9    # BB < 90% of KC → compressed
    squeeze_out_ratio = 1.1   # BB > 110% of KC → expanding

    # How many bars to look back to decide if expansion is growing
    expansion_lookback = 5

    # ATR-percentile based classification (ATR% lookback + percentile thresholds)
    # Used by `classify_vol` to compare current ATR% vs historical distribution.
    atrp_lookback: int = 60
    atrp_high_pct: float = 65.0
    atrp_low_pct: float = 25.0


@dataclass
class ChopConfig:
    adx_lookback: int = 14
    adx_trend_threshold: float = 20.0  # classic ADX threshold
    range_lookback: int = 40          # how many bars to decide “range”


@dataclass
class RegimeConfig:
    trend: TrendConfig = field(default_factory=TrendConfig)
    vol: VolConfig = field(default_factory=VolConfig)
    chop: ChopConfig = field(default_factory=ChopConfig)

DEFAULT_CONFIG = RegimeConfig()
# Backwards-compatibility alias expected by some modules/tests
cfg = DEFAULT_CONFIG
