"""Smoke tests for the regime pipeline.

These run against the on-disk price cache (offline). They validate that both
engines produce well-formed output and that the emitted JSON has the shape the
front-end depends on.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from regime_monitor.data import CACHE_DIR
from regime_monitor.hmm.features import ALL_TICKERS, FEATURE_NAMES, build_feature_matrix
from regime_monitor.hmm.model import RegimeHMM
from regime_monitor.rule_based.detector import detect_regime
from regime_monitor.rule_based.history import rule_based_history

def _reject_constant(x):
    raise AssertionError(f"JSON contains non-finite constant: {x}")


RULE_LABELS = {"bull", "bear", "range", "bull_high_vol", "bear_high_vol", "high_vol"}
HMM_LABELS = {"Risk-On", "Risk-On-Retreat", "Risk-Off", "Risk-Off-Stable"}


def _has_cache() -> bool:
    return (CACHE_DIR / "SPY.parquet").exists()


requires_cache = pytest.mark.skipif(not _has_cache(), reason="no seeded price cache")


def _prices():
    return {
        t: pd.read_parquet(CACHE_DIR / f"{t.replace('^', '_')}.parquet")
        for t in ALL_TICKERS
        if (CACHE_DIR / f"{t.replace('^', '_')}.parquet").exists()
    }


def test_feature_contract_is_18_ordered():
    assert len(FEATURE_NAMES) == 18
    assert FEATURE_NAMES[0] == "spy_ema21_spread"
    assert FEATURE_NAMES[6] == "breadth_ema50"


@requires_cache
def test_rule_based_current():
    spy = pd.read_parquet(CACHE_DIR / "SPY.parquet")
    snap = detect_regime(spy, symbol="SPY", timeframe="1d")
    assert snap.regime in RULE_LABELS
    assert 0.0 <= snap.confidence <= 1.0


@requires_cache
def test_hmm_probabilities_sum_to_100():
    feat = build_feature_matrix(_prices())
    assert list(feat.columns) == FEATURE_NAMES
    hist = RegimeHMM().predict_history(feat)
    last = hist.iloc[-1]
    total = (
        last["risk_on_pct"] + last["risk_on_retreat_pct"]
        + last["risk_off_pct"] + last["risk_off_stable_pct"]
    )
    assert abs(total - 100.0) < 0.5
    assert last["regime"] in HMM_LABELS
    assert last["transition_risk"] in {"Low", "Moderate", "Elevated"}


@requires_cache
def test_rule_based_history_labels_valid():
    spy = pd.read_parquet(CACHE_DIR / "SPY.parquet")
    hist = rule_based_history(spy)
    assert len(hist) > 100
    assert set(hist.unique()).issubset(RULE_LABELS)


@requires_cache
def test_emitted_json_schema(tmp_path):
    import regime_pipeline

    sys.argv = ["regime_pipeline.py", "--cache-only", "--history-years", "2", "--out", str(tmp_path)]
    regime_pipeline.main()

    # Emitted JSON must be strictly valid (no NaN/Infinity — the browser's
    # JSON.parse rejects them and the site would fail to load).
    for name in ("latest.json", "history.json", "validation.json"):
        raw = (tmp_path / name).read_text()
        json.loads(raw, parse_constant=_reject_constant)

    latest = json.loads((tmp_path / "latest.json").read_text())
    for key in ("as_of", "spy_close", "rule_based", "hmm", "data_freshness", "agreement"):
        assert key in latest
    assert set(latest["hmm"]["probabilities"]) == HMM_LABELS

    history = json.loads((tmp_path / "history.json").read_text())
    for key in ("hmm", "rule_based", "spy"):
        assert key in history and len(history[key]) > 0
    # trimmed history: hmm records carry only date + regime
    assert set(history["hmm"][0].keys()) == {"date", "regime"}

    validation = json.loads((tmp_path / "validation.json").read_text())
    assert validation["horizons"] == [5, 21, 63]
    for engine in ("hmm", "rule_based"):
        rows = validation[engine]["21"]
        assert rows and all(
            {"regime", "mean", "median", "hit_rate", "n"} <= set(r) for r in rows
        )
