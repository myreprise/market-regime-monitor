#!/usr/bin/env python3
"""Market Regime Monitor — data pipeline.

Fetches price data, computes the market regime two ways (a transparent
rule-based trend/vol/chop detector and a 4-state statistical HMM), and emits
the JSON the static site renders:

    docs/data/latest.json    current read from both engines + freshness
    docs/data/history.json   daily regime time series + SPY close (for the ribbon)

Run:
    python regime_pipeline.py                 # fetch fresh + write JSON
    python regime_pipeline.py --cache-only     # offline, use cached prices
    python regime_pipeline.py --history-years 5
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from regime_monitor.data import fetch_ohlcv
from regime_monitor.hmm.features import ALL_TICKERS, build_feature_matrix
from regime_monitor.hmm.model import RegimeHMM
from regime_monitor.rule_based.detector import detect_regime
from regime_monitor.rule_based.history import rule_based_history

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "docs" / "data"

# Rule-based composite labels -> the coarse direction shown alongside the HMM.
_RULE_DIRECTION = {
    "bull": "Risk-On",
    "bull_high_vol": "Risk-On (high vol)",
    "bear": "Risk-Off",
    "bear_high_vol": "Risk-Off (high vol)",
    "high_vol": "High Volatility",
    "range": "Range / Neutral",
}

# Collapse each engine's label to on / off / neutral for an agreement check.
_RULE_COARSE = {
    "bull": "on", "bull_high_vol": "on",
    "bear": "off", "bear_high_vol": "off",
    "range": "neutral", "high_vol": "neutral",
}
_HMM_COARSE = {
    "Risk-On": "on", "Risk-Off": "off",
    "Risk-On-Retreat": "neutral", "Risk-Off-Stable": "neutral",
}


def _load_prices(cache_only: bool, period: str) -> dict[str, pd.DataFrame]:
    prices: dict[str, pd.DataFrame] = {}
    for ticker in ALL_TICKERS:
        df = fetch_ohlcv(ticker, period=period, cache_only=cache_only)
        if df is not None and not df.empty:
            prices[ticker] = df
    if "SPY" not in prices:
        raise SystemExit("No SPY data available — cannot compute regime.")
    return prices


def _rule_based_current(spy: pd.DataFrame) -> dict:
    snap = detect_regime(spy, symbol="SPY", timeframe="1d")
    return {
        "regime": snap.regime,
        "direction": _RULE_DIRECTION.get(snap.regime, snap.regime),
        "confidence": round(float(snap.confidence), 3),
        "trend": snap.trend.label,
        "ema_stack": snap.trend.ema_stack,
        "vol": snap.vol.vol_label,
        "atr_percent": round(float(snap.vol.atr_percent), 2),
        "chop": snap.chop.chop_label,
        "adx": round(float(snap.chop.adx), 1),
    }


def _forward_stats(labels_by_date: dict[str, str], spy: pd.DataFrame,
                   horizons=(5, 21, 63)) -> dict:
    """Forward SPY-return statistics grouped by regime, per horizon.

    For each bar labelled with a regime, measure the SPY return over the next
    `h` trading days, then aggregate by regime. This is an *in-sample,
    descriptive* association (overlapping windows) — it shows whether the
    regimes separate future outcomes, not a forecast.
    """
    closes = spy["close"].to_numpy(dtype=float)
    pos_by_date = {d.strftime("%Y-%m-%d"): i for i, d in enumerate(spy.index)}
    n = len(closes)

    out = {}
    for h in horizons:
        groups: dict[str, list[float]] = {}
        for d, lbl in labels_by_date.items():
            i = pos_by_date.get(d)
            if i is None or i + h >= n:
                continue
            groups.setdefault(lbl, []).append(closes[i + h] / closes[i] - 1.0)
        rows = []
        for lbl, vals in groups.items():
            arr = np.array(vals, dtype=float)
            rows.append({
                "regime": lbl,
                "mean": round(float(arr.mean()) * 100, 2),
                "median": round(float(np.median(arr)) * 100, 2),
                "hit_rate": round(float((arr > 0).mean()) * 100, 1),
                "n": int(arr.size),
            })
        out[str(h)] = rows
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute market regime and emit site JSON.")
    ap.add_argument("--cache-only", action="store_true", help="Use cached prices only (offline).")
    ap.add_argument("--period", default="max", help="yfinance history period to request (default: max).")
    ap.add_argument("--history-years", type=int, default=5, help="Years of history to emit for the ribbon.")
    ap.add_argument("--out", default=str(OUT_DIR), help="Output directory for JSON.")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading prices...")
    prices = _load_prices(args.cache_only, args.period)
    spy = prices["SPY"]
    as_of = spy.index[-1]
    print(f"  SPY through {as_of.date()} ({len(prices)} tickers loaded)")

    # --- Engine 1: transparent rule-based composite ---
    rule_now = _rule_based_current(spy)
    print(f"  rule-based : {rule_now['regime']} (conf {rule_now['confidence']})")

    # --- Engine 2: statistical 4-state HMM ---
    feat = build_feature_matrix(prices)
    hmm_hist = RegimeHMM().predict_history(feat)
    hmm_last = hmm_hist.iloc[-1]
    print(f"  HMM        : {hmm_last['regime']} (conf {hmm_last['confidence']})")

    # Full rule-based history (used for validation stats, then windowed).
    rule_full = rule_based_history(spy)

    # --- Validation: forward-return separation by regime (full history) ---
    hmm_labels = dict(zip(hmm_hist["date"], hmm_hist["regime"]))
    rule_labels = {ts.strftime("%Y-%m-%d"): lbl for ts, lbl in rule_full.items()}
    HORIZONS = [5, 21, 63]
    validation = {
        "as_of": as_of.strftime("%Y-%m-%d"),
        "generated_at": None,  # filled below
        "horizons": HORIZONS,
        "range": {"start": hmm_hist["date"].iloc[0], "end": hmm_hist["date"].iloc[-1]},
        "hmm": _forward_stats(hmm_labels, spy, HORIZONS),
        "rule_based": _forward_stats(rule_labels, spy, HORIZONS),
    }

    # --- History window for the ribbon ---
    cutoff = (as_of - pd.DateOffset(years=args.history_years)).strftime("%Y-%m-%d")
    hmm_window = hmm_hist[hmm_hist["date"] >= cutoff].reset_index(drop=True)

    rule_win = rule_full[rule_full.index >= cutoff]
    rule_hist = {ts.strftime("%Y-%m-%d"): lbl for ts, lbl in rule_win.items()}

    spy_window = spy.loc[spy.index >= cutoff, "close"]
    spy_series = [
        {"date": d.strftime("%Y-%m-%d"), "close": round(float(c), 2)}
        for d, c in spy_window.items()
    ]

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    validation["generated_at"] = generated_at
    stale_days = (pd.Timestamp.now(tz="UTC").normalize() - as_of.tz_localize("UTC")).days

    latest = {
        "as_of": as_of.strftime("%Y-%m-%d"),
        "generated_at": generated_at,
        "spy_close": round(float(spy["close"].iloc[-1]), 2),
        "data_freshness": {
            "last_bar": as_of.strftime("%Y-%m-%d"),
            "days_since_last_bar": int(stale_days),
            "tickers_loaded": len(prices),
        },
        "rule_based": rule_now,
        "hmm": {
            "regime": hmm_last["regime"],
            "confidence": float(hmm_last["confidence"]),
            "transition_risk": hmm_last["transition_risk"],
            "probabilities": {
                "Risk-On": float(hmm_last["risk_on_pct"]),
                "Risk-On-Retreat": float(hmm_last["risk_on_retreat_pct"]),
                "Risk-Off": float(hmm_last["risk_off_pct"]),
                "Risk-Off-Stable": float(hmm_last["risk_off_stable_pct"]),
            },
        },
        "agreement": (
            _RULE_COARSE.get(rule_now["regime"], "neutral")
            == _HMM_COARSE.get(hmm_last["regime"], "neutral")
        ),
    }

    history = {
        "as_of": as_of.strftime("%Y-%m-%d"),
        "generated_at": generated_at,
        # Ribbon only needs date + regime per bar (probabilities live in latest.json).
        "hmm": hmm_window[["date", "regime"]].to_dict(orient="records"),
        "rule_based": [
            {"date": d, "regime": lbl} for d, lbl in rule_hist.items()
        ],
        "spy": spy_series,
    }

    (out_dir / "latest.json").write_text(json.dumps(latest, indent=2))
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    (out_dir / "validation.json").write_text(json.dumps(validation, indent=2))
    print(f"Wrote latest.json, history.json, validation.json → {out_dir}")
    print(f"  ribbon: {len(hmm_window)} HMM bars, {len(rule_hist)} rule-based bars, {len(spy_series)} SPY points")
    for h in HORIZONS:
        rows = {r['regime']: r['mean'] for r in validation['hmm'][str(h)]}
        print(f"  fwd {h}d mean% by HMM regime: {rows}")


if __name__ == "__main__":
    main()
